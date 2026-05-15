"""3D Voronoi cells turned into smooth-edge prism cutters.

For each real seed we:
  1. Compute the 3D Voronoi cell (real + mirror + ghost neighbours).
  2. Project the cell's vertices onto the seed's tangent plane and take the
     2D convex hull. That's the cell silhouette polygon.
  3. Inset the polygon by `strut_thickness / 2` — this is what creates the
     strut between adjacent holes.
  4. Smooth with quadratic Bézier curves anchored at edge midpoints; the
     curve stays inside the inset polygon so the strut gap is preserved.
  5. Extrude the smooth curve along the seed normal into a prism that cuts
     through the shell wall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi, cKDTree

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from voronoizer import progress
from voronoizer.seeding import Seeds


_GHOST_POINT_COUNT = 26
_GHOST_RADIUS_MULT = 3.0

# Prism extrusion: how far past the outer / inner shell surface the cell extends.
_PRISM_OUTWARD_K = 1.0
_PRISM_INWARD_K = 2.0

# Default Bézier sampling: more = smoother boundary / more triangles per cell.
_BEZIER_SAMPLES_PER_EDGE = 6


@dataclass
class CellBuildStats:
    requested: int
    built: int
    shrunk_to_empty: int
    unbounded: int
    degenerate: int


def _fibonacci_sphere(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    return np.stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ], axis=-1)


def _ghost_points(mesh_bounds: np.ndarray) -> np.ndarray:
    center = mesh_bounds.mean(axis=0)
    diag = float(np.linalg.norm(mesh_bounds[1] - mesh_bounds[0]))
    return _fibonacci_sphere(_GHOST_POINT_COUNT) * (_GHOST_RADIUS_MULT * diag) + center


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    return u, v


def _build_neighbor_map(vor: Voronoi) -> dict[int, list[int]]:
    """For each point index, list the indices of its Voronoi neighbours."""
    nm: dict[int, list[int]] = {}
    for i, j in vor.ridge_points:
        nm.setdefault(int(i), []).append(int(j))
        nm.setdefault(int(j), []).append(int(i))
    return nm


def _cell_polygon_at_tangent(
    seed_pos: np.ndarray,
    neighbor_positions: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray | None:
    """2D polygon of the cell sliced by the seed's tangent plane.

    For each neighbour P, the bisector with the seed gives the half-plane
    (P-S)·u·x + (P-S)·v·y ≤ |P-S|² / 2 in tangent (u, v) coordinates. Their
    intersection is the cell's lateral footprint at the seed surface — what
    we want to extrude through the wall."""
    if len(neighbor_positions) == 0:
        return None
    rel = neighbor_positions - seed_pos
    u_j = rel @ u
    v_j = rel @ v
    rel_sq = np.einsum("ij,ij->i", rel, rel)
    # HalfspaceIntersection expects [A_x, A_y, b] with A·x + b ≤ 0.
    eqs = np.column_stack([u_j, v_j, -rel_sq * 0.5])
    try:
        hsi = HalfspaceIntersection(eqs, np.array([0.0, 0.0]))
    except (QhullError, ValueError):
        return None
    pts = np.asarray(hsi.intersections)
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    return pts[hull.vertices]


def _inset_polygon_2d(
    polygon: np.ndarray, offset: float, interior: np.ndarray
) -> np.ndarray | None:
    """Inset (shrink) a 2D convex polygon by `offset`. CCW result or None."""
    if len(polygon) < 3:
        return None
    try:
        hull = ConvexHull(polygon)
    except QhullError:
        return None
    eqs = hull.equations.copy()
    eqs[:, 2] += offset
    try:
        hsi = HalfspaceIntersection(eqs, interior)
    except (QhullError, ValueError):
        return None
    pts = np.asarray(hsi.intersections)
    if len(pts) < 3:
        return None
    try:
        h = ConvexHull(pts)
    except QhullError:
        return None
    return pts[h.vertices]


def _bezier_smooth(polygon: np.ndarray, samples_per_edge: int) -> np.ndarray:
    """Smooth a closed convex polygon with quadratic Bézier curves.

    For each edge V[i]→V[i+1] we use midpoints as endpoints and V[i] as the
    control point. The smooth curve stays inside the convex polygon, so two
    inset cells preserve the `strut_thickness` gap between them.
    """
    N = len(polygon)
    nxt = (np.arange(N) + 1) % N
    mid = (polygon + polygon[nxt]) * 0.5
    prev_mid = np.roll(mid, 1, axis=0)
    ts = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False).reshape(-1, 1)
    chunks: list[np.ndarray] = []
    for i in range(N):
        p0, p1, p2 = prev_mid[i], polygon[i], mid[i]
        chunks.append(((1 - ts) ** 2) * p0 + (2 * (1 - ts) * ts) * p1 + (ts ** 2) * p2)
    return np.vstack(chunks)


def _estimate_local_radius(
    seed: np.ndarray,
    normal: np.ndarray,
    neighbor_pts: np.ndarray,
    neighbor_normals: np.ndarray,
) -> float:
    """Estimate the local radius of curvature from same-patch neighbours.

    Returns +inf for flat regions (cube faces, etc.). Only neighbours whose
    normals roughly agree with the seed's are used — that excludes neighbours
    on the adjacent face across a sharp edge, which would mis-read as huge
    curvature."""
    if len(neighbor_pts) == 0:
        return float("inf")
    deltas = neighbor_pts - seed
    distances = np.linalg.norm(deltas, axis=1)
    cos_angles = np.clip(neighbor_normals @ normal, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    # cos > 0.6 ≈ same-patch neighbours within ~53°. Loose enough to include
    # the typical ~30° gap on a sparsely-seeded sphere but tight enough to
    # exclude cross-face cube neighbours (cos ≈ 0).
    valid = (cos_angles > 0.6) & (angles > 1e-3) & (distances > 1e-6)
    if not np.any(valid):
        return float("inf")
    return float(np.median(distances[valid] / angles[valid]))


def _expand_polygon_centroid(polygon_2d: np.ndarray, offset: float) -> np.ndarray:
    """Move each vertex `offset` mm outward along the centroid->vertex ray.

    Used to build the chamfered cross-section. Roughly equivalent to a polygon
    offset for near-equiaxial convex polygons; exact for circles."""
    centroid = polygon_2d.mean(axis=0)
    rel = polygon_2d - centroid
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    return polygon_2d + (rel / np.maximum(dist, 1e-9)) * offset


# Cone half-angle (cosine) used to decide whether the surface point a polygon
# vertex projects onto belongs to the seed's "local" patch. If the projected
# face normal disagrees with the seed normal by more than this cone, fall back
# to the seed's frame for that vertex — keeps a cell that spans a sharp
# dihedral edge from latching onto a neighbouring face.
_PHASE1_FALLBACK_COS = 0.5


def _surface_frames(
    polygon_2d: np.ndarray,
    seed: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    seed_normal: np.ndarray,
    proximity,
    face_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each polygon vertex, compute (P_i, n_i, d_i).

    P_i is the nearest point on the input mesh to the tangent-plane vertex;
    n_i is the local outward normal there (face normal); d_i is the centroid-
    radial direction taken in seed's tangent frame and re-projected into P_i's
    tangent plane so the chamfer offset stays in-surface.

    Where the projected normal disagrees with the seed normal by more than
    ~60° (`_PHASE1_FALLBACK_COS`), fall back to the seed's own frame for that
    vertex.
    """
    p3d = seed + polygon_2d[:, 0:1] * u + polygon_2d[:, 1:2] * v  # (N, 3)
    closest, _dist, face_ids = proximity.on_surface(p3d)
    P_proj = np.asarray(closest, dtype=float)
    n_proj = np.asarray(face_normals[face_ids], dtype=float)

    # Outward direction at each vertex as the bisector of its two adjacent
    # edges' outward perpendiculars (the standard 2D polygon-offset rule).
    # Earlier versions used a centroid-radial direction, which on highly
    # asymmetric polygons (typical of sparse Voronoi seeds on a sphere)
    # didn't point cleanly perpendicular to either adjacent edge — chamfer
    # offsets in that direction occasionally brought adjacent cells almost
    # to a tangent kiss, producing twin vertices in the boolean output.
    prev_v = np.roll(polygon_2d, 1, axis=0)
    next_v = np.roll(polygon_2d, -1, axis=0)
    e_prev = polygon_2d - prev_v
    e_next = next_v - polygon_2d
    # CCW polygon → outward perpendicular is rotate-by-(-90°): (y, -x).
    perp_prev = np.column_stack([e_prev[:, 1], -e_prev[:, 0]])
    perp_next = np.column_stack([e_next[:, 1], -e_next[:, 0]])
    perp_prev = perp_prev / np.maximum(
        np.linalg.norm(perp_prev, axis=1, keepdims=True), 1e-9
    )
    perp_next = perp_next / np.maximum(
        np.linalg.norm(perp_next, axis=1, keepdims=True), 1e-9
    )
    dir_2d = perp_prev + perp_next
    dir_2d = dir_2d / np.maximum(
        np.linalg.norm(dir_2d, axis=1, keepdims=True), 1e-9
    )
    dir_3d = dir_2d[:, 0:1] * u + dir_2d[:, 1:2] * v  # (N, 3)

    # Fallback mask: use projected frame only when the projected face faces
    # roughly the same way as the seed.
    cos_agree = n_proj @ seed_normal
    use_proj = cos_agree > _PHASE1_FALLBACK_COS

    seed_n_b = np.broadcast_to(seed_normal, n_proj.shape)
    P = np.where(use_proj[:, None], P_proj, p3d)
    n = np.where(use_proj[:, None], n_proj, seed_n_b)
    # Re-orthonormalise the per-vertex normal we just chose.
    n_len = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(n_len, 1e-12)

    # Re-project the centroid-radial direction into each P_i's tangent plane.
    dot = np.einsum("ij,ij->i", dir_3d, n)
    dir_in_tangent = dir_3d - dot[:, None] * n
    proj_len = np.linalg.norm(dir_in_tangent, axis=1, keepdims=True)
    d_out = dir_in_tangent / np.maximum(proj_len, 1e-9)

    return P, n, d_out


def _build_prism_surface_aware(
    polygon_2d: np.ndarray,
    P: np.ndarray,
    n: np.ndarray,
    d_out: np.ndarray,
    seed: np.ndarray,
    seed_normal: np.ndarray,
    R_local: float,
    shell_thickness: float,
    chamfer: float,
    safety: float,
) -> trimesh.Trimesh:
    """Phase-1 surface-aware chamfered prism.

    Each polygon vertex i has its own (P_i, n_i, d_i). Ring positions are
    built per-vertex along the local frame, so the chamfer rings 1/4 always
    sit on the actual outer / inner shell surface and the bevel is visible
    at every part of the hole boundary, even on curved surfaces.

    The top cap centroid is pushed `R_local` mm along the seed's normal on
    curved surfaces (else just `safety` mm on flat ones). A planar
    cap-wheel triangle from a low centroid to a ring0 vertex on a sphere is
    a *chord* across the outer surface — its midpoint lies *inside* the
    sphere whenever the ring0 vertex is more than ~20° off the seed-axis,
    and the chord cuts back through the shell. Putting the centroid at
    `seed + R_local · seed_normal` guarantees every chord midpoint stays
    outside the parallel sphere for polygon spreads up to 90°.

    The bottom cap stays near the seed (inside the sphere is empty space,
    so cap-triangles dipping further inward there are harmless).
    """
    N = len(polygon_2d)

    # On curved surfaces, lift the chamfered rings 0.05 mm into the air
    # outside the shell along each polygon vertex's local normal. That
    # displaces the *exact* chamfered polygon positions off the surface so
    # manifold3d cuts the prism at the surface inside the chamfer
    # transition rather than exactly at the chamfered position. Adjacent
    # cells then don't share a coincident point on the surface and the
    # twin-vertex slivers (~500 non-manifold edges per sphere STL) go away.
    #
    # To keep the visible chamfer width the same, the polygon expansion is
    # widened by `lift` so the boolean's cross-section at altitude 0
    # interpolates back to exactly `chamfer` mm. Bevel angle stays 45°.
    #
    # On flat surfaces (cube faces, cylinder caps) the lift is 0 — there
    # are no near-tangencies to break, and the lift would only reduce the
    # visible chamfer slightly.
    if np.isfinite(R_local) and R_local > 0.0:
        lift = 0.05
        expansion = chamfer + lift
    else:
        lift = 0.0
        expansion = chamfer
    chamf_lat = expansion * d_out  # (N, 3)

    ring0 = P + chamf_lat + safety * n
    ring1 = P + chamf_lat + lift * n
    ring2 = P - chamfer * n
    ring3 = P - (shell_thickness - chamfer) * n
    ring4 = P - shell_thickness * n + chamf_lat - lift * n
    ring5 = ring4 - safety * n

    if np.isfinite(R_local) and R_local > 0.0:
        cap_top_height = max(safety, R_local)
    else:
        cap_top_height = safety
    cap_top_centroid = seed + cap_top_height * seed_normal
    cap_bot_centroid = seed - (shell_thickness + safety) * seed_normal
    verts = np.vstack([ring0, ring1, ring2, ring3, ring4, ring5,
                       cap_top_centroid[None, :], cap_bot_centroid[None, :]])

    n_rings = 6
    faces: list[list[int]] = []
    # Side walls between consecutive rings.
    for r in range(n_rings - 1):
        a = r * N
        b = (r + 1) * N
        for i in range(N):
            j = (i + 1) % N
            faces.append([a + i, a + j, b + j])
            faces.append([a + i, b + j, b + i])
    cap_top_idx = n_rings * N
    cap_bot_idx = n_rings * N + 1
    # Top cap wheel — outward-facing.
    for i in range(N):
        faces.append([cap_top_idx, i, (i + 1) % N])
    # Bottom cap wheel — reversed for outward -normal facing.
    bot_off = (n_rings - 1) * N
    for i in range(N):
        faces.append([cap_bot_idx, bot_off + ((i + 1) % N), bot_off + i])

    mesh_obj = trimesh.Trimesh(
        vertices=verts, faces=np.asarray(faces, dtype=int), process=False
    )
    mesh_obj.fix_normals()
    return mesh_obj


def _build_prism(
    polygon_2d: np.ndarray,
    seed: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    out_height: float,
    in_height: float,
    R_local: float = float("inf"),
    shell_thickness: float = 0.0,
    chamfer: float = 0.0,
) -> trimesh.Trimesh:
    """Extrude a CCW 2D polygon (in seed's tangent frame) into a 3D solid.

    For curved surfaces, each cross-section is scaled by (R + h) / R (h signed,
    positive = outward) so the side walls follow the surface curvature. Rings
    that should land on a shell surface (outer or inner) get per-vertex
    altitude corrections `-d²·s / (2R)` so each vertex sits on the actual
    parallel sphere instead of floating in the seed's flat tangent plane.

    When `chamfer > 0`, the prism gains extra rings that widen the cross-
    section at the outer and inner shell surfaces, producing a 45°-ish bevel
    where the hole meets each surface.
    """
    use_chamfer = chamfer > 0.0 and shell_thickness > 0.0
    chamfered = _expand_polygon_centroid(polygon_2d, chamfer) if use_chamfer else polygon_2d

    N = len(polygon_2d)
    h_max = max(out_height, in_height, shell_thickness) if use_chamfer else max(out_height, in_height)
    use_curvature = np.isfinite(R_local) and R_local > h_max + 1e-6

    if use_chamfer:
        # Six fixed-altitude rings with linear curvature lateral taper. On a
        # curved surface this means the chamfer rings (at altitude 0 and
        # -shell) sit on the seed's tangent plane / parallel plane rather
        # than on the actual curved surface — so the chamfer bevel is
        # visible only near the seed where the surface is close to tangent.
        # The alternative (per-vertex altitudes that follow the actual
        # surface) gives uniform chamfer but turns elongated Voronoi cells
        # into very tall prisms that reach into other cells' territory and
        # leave gaps or uncut shell in the middle of holes. The fixed
        # altitudes keep each prism in its seed's local region.
        if use_curvature:
            s_0 = 1.0 + out_height / R_local
            s_1 = 1.0
            s_2 = 1.0 - chamfer / R_local
            s_3 = 1.0 - (shell_thickness - chamfer) / R_local
            s_4 = 1.0 - shell_thickness / R_local
            s_5 = 1.0 - in_height / R_local
        else:
            s_0 = s_1 = s_2 = s_3 = s_4 = s_5 = 1.0

        rings: list[tuple[np.ndarray, float, np.ndarray]] = [
            (chamfered,  s_0, np.full(N,  out_height)),                  # 0
            (chamfered,  s_1, np.zeros(N)),                              # 1
            (polygon_2d, s_2, np.full(N, -chamfer)),                     # 2
            (polygon_2d, s_3, np.full(N, -(shell_thickness - chamfer))), # 3
            (chamfered,  s_4, np.full(N, -shell_thickness)),             # 4
            (chamfered,  s_5, np.full(N, -in_height)),                   # 5
        ]
    else:
        if use_curvature:
            s_top = 1.0 + out_height / R_local
            s_bot = 1.0 - in_height / R_local
        else:
            s_top = s_bot = 1.0
        rings = [
            (polygon_2d, s_top, np.full(N,  out_height)),
            (polygon_2d, s_bot, np.full(N, -in_height)),
        ]

    ring_verts: list[np.ndarray] = []
    for poly, s, h_per_vertex in rings:
        tang = poly[:, 0:1] * u + poly[:, 1:2] * v  # (N, 3)
        ring_verts.append(seed + s * tang + h_per_vertex[:, None] * normal)
    verts = np.vstack(ring_verts)

    n_rings = len(rings)
    faces: list[list[int]] = []
    # Side walls between consecutive rings.
    for r in range(n_rings - 1):
        a = r * N
        b = (r + 1) * N
        for i in range(N):
            j = (i + 1) % N
            faces.append([a + i, a + j, b + j])
            faces.append([a + i, b + j, b + i])
    # Top cap on ring 0 — fan from vertex 0, CCW in (u, v) viewed from +normal.
    for i in range(1, N - 1):
        faces.append([0, i, i + 1])
    # Bottom cap on the last ring, reversed for outward -normal facing.
    last = (n_rings - 1) * N
    for i in range(1, N - 1):
        faces.append([last, last + i + 1, last + i])

    mesh = trimesh.Trimesh(
        vertices=verts, faces=np.asarray(faces, dtype=int), process=False
    )
    mesh.fix_normals()
    return mesh


def build_shrunken_cells(
    seeds: Seeds,
    mesh: trimesh.Trimesh,
    shell_thickness: float,
    strut_thickness: float,
    mirror_seeds: np.ndarray | None = None,
    bezier_samples_per_edge: int = _BEZIER_SAMPLES_PER_EDGE,
    chamfer: float = 0.0,
) -> tuple[list[trimesh.Trimesh], CellBuildStats]:
    """Build a smooth-edge prism cutter for each seed.

    `mesh` is the surface the seeds were sampled from (full input mesh, or
    the top/bottom submesh in `--top-bottom-only` mode). It's used to project
    polygon vertices onto the actual surface so the chamfer rings sit on the
    real outer / inner shell surface at every vertex (Phase 1 surface-aware
    prism). The non-chamfer path is unchanged.

    `chamfer` (mm) bevels the hole edges where they meet the shell surfaces.
    Clamped to keep struts and the straight wall section non-degenerate.
    """
    # Clamp chamfer: must leave some strut between cells at the surface
    # (gap = strut - 2*chamfer) and a non-empty straight wall in the middle.
    chamfer_in = chamfer
    chamfer = max(0.0, chamfer)
    chamfer = min(chamfer, 0.49 * strut_thickness, 0.49 * shell_thickness)
    if chamfer < chamfer_in - 1e-9:
        progress.warn(
            f"chamfer {chamfer_in:.3f} mm clamped to {chamfer:.3f} mm "
            f"(limit: min(0.49 * strut, 0.49 * shell_thickness))"
        )

    mesh_bounds = mesh.bounds

    # Build a single proximity query for surface projection (chamfer path).
    proximity = None
    face_normals = None
    if chamfer > 0.0:
        from trimesh.proximity import ProximityQuery
        proximity = ProximityQuery(mesh)
        face_normals = np.asarray(mesh.face_normals, dtype=float)

    n_real = len(seeds)
    pieces = [seeds.points]
    n_mirror = 0
    if mirror_seeds is not None and len(mirror_seeds) > 0:
        pieces.append(mirror_seeds)
        n_mirror = len(mirror_seeds)
    pieces.append(_ghost_points(mesh_bounds))
    all_points = np.vstack(pieces)

    with progress.step(
        f"compute 3D Voronoi on {n_real} seeds "
        f"(+ {n_mirror} mirrors, + {_GHOST_POINT_COUNT} ghosts)"
    ):
        vor = Voronoi(all_points)

    neighbor_map = _build_neighbor_map(vor)

    inset = strut_thickness / 2.0
    out_h = _PRISM_OUTWARD_K * shell_thickness
    in_h = _PRISM_INWARD_K * shell_thickness

    # KDTree over real seeds (excluding mirrors/ghosts) for local curvature.
    K_curv = min(7, n_real)
    seed_tree = cKDTree(seeds.points) if n_real >= 2 else None

    cells: list[trimesh.Trimesh] = []
    stats = CellBuildStats(
        requested=n_real, built=0, shrunk_to_empty=0, unbounded=0, degenerate=0
    )

    iterator = progress.progress(range(n_real), desc="build cells", total=n_real)
    for i in iterator:
        nbr_idx = neighbor_map.get(i, [])
        if not nbr_idx:
            stats.unbounded += 1
            continue

        seed = seeds.points[i]
        normal = seeds.normals[i]
        u, v = _tangent_basis(normal)

        if seed_tree is None:
            R_local = float("inf")
        else:
            _, kidx = seed_tree.query(seed, k=K_curv)
            kidx = np.atleast_1d(kidx)
            kidx = kidx[kidx != i][: K_curv - 1]
            R_local = _estimate_local_radius(
                seed, normal, seeds.points[kidx], seeds.normals[kidx]
            )

        polygon = _cell_polygon_at_tangent(seed, all_points[nbr_idx], u, v)
        if polygon is None:
            stats.degenerate += 1
            continue

        centroid = polygon.mean(axis=0)
        inset_poly = _inset_polygon_2d(polygon, inset, centroid)
        if inset_poly is None or len(inset_poly) < 3:
            stats.shrunk_to_empty += 1
            continue

        smooth_poly = _bezier_smooth(inset_poly, bezier_samples_per_edge)

        try:
            if chamfer > 0.0:
                # Phase 1: chamfer path is built per-vertex on the actual
                # surface, with each column getting its own local frame.
                P_pv, n_pv, d_pv = _surface_frames(
                    smooth_poly, seed, u, v, normal, proximity, face_normals
                )
                # Generous cap safety past each surface: the boolean operates
                # better with a thick prism than a sliver, and the part past
                # the shell is clipped away anyway.
                safety = max(1.0, shell_thickness)
                prism = _build_prism_surface_aware(
                    smooth_poly, P_pv, n_pv, d_pv,
                    seed=seed, seed_normal=normal,
                    R_local=R_local,
                    shell_thickness=shell_thickness,
                    chamfer=chamfer,
                    safety=safety,
                )
            else:
                prism = _build_prism(
                    smooth_poly, seed, normal, u, v, out_h, in_h,
                    R_local=R_local,
                    shell_thickness=shell_thickness,
                    chamfer=chamfer,
                )
        except Exception:
            stats.degenerate += 1
            continue
        if len(prism.faces) == 0:
            stats.degenerate += 1
            continue

        cells.append(prism)
        stats.built += 1

    progress.log(
        f"cells built: {stats.built} / {stats.requested} "
        f"(unbounded={stats.unbounded}, "
        f"collapsed-by-strut={stats.shrunk_to_empty}, "
        f"degenerate={stats.degenerate})"
    )
    return cells, stats
