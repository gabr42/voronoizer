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
    mesh_bounds: np.ndarray,
    shell_thickness: float,
    strut_thickness: float,
    mirror_seeds: np.ndarray | None = None,
    bezier_samples_per_edge: int = _BEZIER_SAMPLES_PER_EDGE,
    chamfer: float = 0.0,
) -> tuple[list[trimesh.Trimesh], CellBuildStats]:
    """Build a smooth-edge prism cutter for each seed.

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
