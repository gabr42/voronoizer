"""Prism-builder primitives shared with the geodesic engine.

The geodesic pipeline (`voronoizer.surface_pipeline`) drives cell
construction; the helpers below take care of:

  * `_inset_polygon_2d`    — shrink a 2D convex polygon by an offset
                             (used for the strut gap).
  * `_bezier_smooth`       — smooth a closed polygon with quadratic Béziers
                             anchored at edge midpoints.
  * `_estimate_local_radius` — surface curvature estimate from same-patch
                             neighbours, used to size the cap height.
  * `_build_prism_surface_aware` — extrude per-vertex (P, n, d_out) frames
                             into a cutter prism with optional asymmetric
                             chamfer.
  * `_clamp_chamfer_value` — clamp `--chamfer` / `--inner-chamfer` to keep
                             struts and the straight wall non-degenerate.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from voronoizer import progress


# Default Bézier sampling: more = smoother boundary / more triangles per cell.
_BEZIER_SAMPLES_PER_EDGE = 6


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


def _clamp_chamfer_value(
    value: float, strut_thickness: float, shell_thickness: float, name: str
) -> float:
    """Clamp a chamfer value to keep struts and the straight wall non-
    degenerate. Shared by `--chamfer` and `--inner-chamfer`."""
    original = value
    value = max(0.0, value)
    value = min(value, 0.49 * strut_thickness, 0.49 * shell_thickness)
    if value < original - 1e-9:
        progress.warn(
            f"{name} {original:.3f} mm clamped to {value:.3f} mm "
            f"(limit: min(0.49 * strut, 0.49 * shell_thickness))"
        )
    return value


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
    force_lift: bool = False,
    chamfer_inner: float | None = None,
) -> trimesh.Trimesh:
    """Surface-aware prism cutter with optional asymmetric chamfer.

    Each polygon vertex i has its own (P_i, n_i, d_i). Ring positions are
    built per-vertex along the local frame, so the hole edge always sits on
    the actual shell surface — even on curved surfaces where a single
    seed-tangent prism would intersect the curved surface at a shallow
    angle and produce non-tangent cuts.

    `chamfer` is the OUTER chamfer (where the hole meets the outer shell
    surface). `chamfer_inner` is the inner chamfer; if `None` it defaults
    to `chamfer` for backward compatibility. Set `chamfer_inner=0` with
    a non-zero `chamfer` to chamfer only the outer side and leave the
    inner contour as a clean perpendicular cut.

    Ring count varies with which sides are chamfered:

      * 4 rings  — no chamfer on either side:
          cap-top, outer, inner, cap-bot.
      * 5 rings  — chamfer on one side only:
          cap-top [+widened if outer], outer-chamfered,
          chamfer-end, inner [-widened if inner], cap-bot.
      * 6 rings  — chamfer on both sides:
          cap-top widened, outer-chamfered, chamfer-end,
          inner-chamfer-start, inner-chamfered, cap-bot widened.

    Chamfered rings are lifted 0.05 mm off the surface on curved /
    force_lift inputs so manifold3d's boolean cuts the prism in the
    chamfer transition rather than exactly on the surface; the polygon
    expansion is widened by the same amount so the visible chamfer
    width is unchanged.

    The top cap centroid is pushed `max(safety, R_local)` mm along the
    seed's normal on curved surfaces. A planar cap-wheel triangle from a
    low centroid to a ring0 vertex on a sphere is a *chord* across the
    outer surface — its midpoint lies inside the sphere whenever the
    ring0 vertex is more than ~20° off the seed-axis, and the chord
    would cut back through the shell. Putting the centroid at
    `seed + R_local · seed_normal` guarantees every chord midpoint stays
    outside the parallel sphere for polygon spreads up to 90°. The
    bottom cap stays near the seed (inside the sphere is empty space, so
    cap-triangles dipping further inward there are harmless).
    """
    if chamfer_inner is None:
        chamfer_inner = chamfer
    chamfer_outer = chamfer

    N = len(polygon_2d)
    curved = np.isfinite(R_local) and R_local > 0.0
    apply_lift = curved or force_lift
    lift = 0.05 if apply_lift else 0.0

    co = float(chamfer_outer)
    ci = float(chamfer_inner)
    has_outer = co > 0.0
    has_inner = ci > 0.0

    # Lateral expansions for the chamfered rings.
    chamf_lat_outer = (co + lift) * d_out if has_outer else 0.0
    chamf_lat_inner = (ci + lift) * d_out if has_inner else 0.0

    ring_pos: list[np.ndarray] = []
    # Top side (outer cap and outer surface region).
    if has_outer:
        ring_pos.append(P + chamf_lat_outer + safety * n)           # 0: cap top widened
        ring_pos.append(P + chamf_lat_outer + lift * n)             # 1: outer chamfered
        ring_pos.append(P - co * n)                                 # 2: outer chamfer end
    else:
        ring_pos.append(P + safety * n)                             # 0: cap top
        ring_pos.append(P + lift * n)                               # 1: outer surface
    # Bottom side (inner surface region and inner cap).
    if has_inner:
        ring_pos.append(P - (shell_thickness - ci) * n)             # inner chamfer start
        ring_pos.append(P - shell_thickness * n + chamf_lat_inner - lift * n)              # inner chamfered
        ring_pos.append(P - shell_thickness * n + chamf_lat_inner - lift * n - safety * n) # cap bot widened
    else:
        ring_pos.append(P - shell_thickness * n - lift * n)         # inner surface
        ring_pos.append(P - shell_thickness * n - safety * n)       # cap bot
    n_rings = len(ring_pos)

    cap_top_height = max(safety, R_local) if curved else safety
    cap_top_centroid = seed + cap_top_height * seed_normal
    cap_bot_centroid = seed - (shell_thickness + safety) * seed_normal
    verts = np.vstack(ring_pos + [cap_top_centroid[None, :],
                                  cap_bot_centroid[None, :]])

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
