"""Geodesic Voronoi engine — orchestrator that wires Stages 1–6.

The `tangent` engine in `voronoi_cells.build_shrunken_cells` and the
`geodesic` engine here both produce a `list[trimesh.Trimesh]` of cell
cutters compatible with `perforate.perforate`. Stages 1–6:

  Stage 1 (`surface_voronoi.subdivide_for_geodesic`)
    Subdivide the input mesh until edges are short enough for Dijkstra to
    approximate geodesic distance with sub-strut precision.

  Stage 2 (`surface_voronoi.assign_cell_labels`)
    Multi-source Dijkstra: each mesh vertex is labelled with the
    geodesically-closest seed.

  Stage 3 (`surface_boundary.extract_cell_loops`)
    Pull mesh edges that sit on cell boundaries into closed 3D polylines
    per cell (cell on the left of the walking direction).

  Stage 4 (`surface_boundary.bezier_smooth_on_surface`)
    Quadratic Bézier smoothing followed by surface re-projection — mirrors
    Phase 1's `_bezier_smooth` but the smoothed curve stays on the actual
    surface.

  Stage 5 (`surface_boundary.inset_loop_on_surface`)
    Shift each boundary vertex inward by `strut/2` along
    `surface_normal × forward_tangent` and re-project.

  Stage 6 (`surface_prism.build_prism_from_loop`)
    Reuse Phase 1's `_build_prism_surface_aware` with per-vertex frames
    derived directly from the surface loop.

Stage 7 (boolean subtract) is `perforate.perforate`, unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from voronoizer import progress
from voronoizer.seeding import Seeds
from voronoizer.surface_boundary import (
    Loop,
    _smoothed_normals_at,
    bezier_smooth_on_surface,
    convex_hull_indices_in_tangent,
    convex_polygon_2d_in_tangent,
    extract_cell_loops,
    inset_loop_on_surface,
    project_polygon_2d_to_surface,
    resample_loop_arclen,
)
from voronoizer.surface_prism import build_prism_from_loop
from voronoizer.surface_voronoi import (
    assign_cell_labels,
    face_components as compute_face_components,
    face_labels_from_vertex_labels,
    patch_boundary_vertex_indices,
    patch_is_flat,
    smooth_vertex_normals_within_patches,
    subdivide_for_geodesic,
)
from voronoizer.voronoi_cells import (
    _BEZIER_SAMPLES_PER_EDGE,
    _bezier_smooth,
    _estimate_local_radius,
    _inset_polygon_2d,
)


from scipy.spatial import ConvexHull, HalfspaceIntersection
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError


def _polygon_clip_and_inset(
    polygon_2d: np.ndarray,
    strut_half: float,
    patch_clip_eqs: np.ndarray | None,
    shell_thickness: float,
) -> np.ndarray | None:
    """Inset the cell polygon by `strut_half` AND clip to the patch
    boundary inset by `shell_thickness`, in a single HalfspaceIntersection
    pass.

    Returns CCW vertices of the resulting convex polygon, or None if the
    inset eats it to nothing.
    """
    if len(polygon_2d) < 3:
        return None
    try:
        hull = ConvexHull(polygon_2d)
    except (QhullError, ValueError):
        return None
    cell_eqs = hull.equations.copy()
    cell_eqs[:, 2] += strut_half
    if patch_clip_eqs is not None and len(patch_clip_eqs) > 0:
        clip_eqs = patch_clip_eqs.copy()
        clip_eqs[:, 2] += shell_thickness
        all_eqs = np.vstack([cell_eqs, clip_eqs])
    else:
        all_eqs = cell_eqs
    interior = polygon_2d.mean(axis=0)
    try:
        hsi = HalfspaceIntersection(all_eqs, interior)
    except (QhullError, ValueError):
        return None
    pts = np.asarray(hsi.intersections)
    if len(pts) < 3:
        return None
    try:
        h = ConvexHull(pts)
    except (QhullError, ValueError):
        return None
    return pts[h.vertices]


def _patch_clip_halfplanes(
    mesh: trimesh.Trimesh,
    face_comp: np.ndarray,
    comp_id: int,
    seed: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray | None:
    """Half-plane constraints (M, 3) in 2D tangent (u, v) frame from the
    convex hull of the patch's boundary vertices, projected to tangent
    coords relative to `seed`.

    Returns None when the patch has no boundary (closed mesh component
    with no sharp edges — e.g. a sphere). The constraints are in
    `HalfspaceIntersection` format `(A_x, A_y, c)` with `A·x + c ≤ 0`,
    *before* applying the shell-thickness inset margin; the caller adds
    that to the `c` term.
    """
    boundary_vs = patch_boundary_vertex_indices(mesh, face_comp, comp_id)
    if len(boundary_vs) < 3:
        return None
    rel = mesh.vertices[boundary_vs] - seed
    pts2d = np.column_stack([rel @ u, rel @ v])
    try:
        hull = ConvexHull(pts2d)
    except (QhullError, ValueError):
        return None
    return hull.equations.copy()


@dataclass
class GeodesicCellStats:
    requested: int
    built: int
    no_loop: int          # cell that found no boundary loop at all
    too_few_vertices: int # loops with < 3 verts after resample
    prism_failed: int


def build_geodesic_cells(
    seeds: Seeds,
    mesh: trimesh.Trimesh,
    shell_thickness: float,
    strut_thickness: float,
    chamfer: float,
    target_edge_length: float | None = None,
    resample_step: float | None = None,
    sharp_angle_deg: float = 25.0,
    chamfer_inner: float | None = None,
) -> tuple[list[trimesh.Trimesh], GeodesicCellStats]:
    """Build prism cutters for each seed using the geodesic engine."""
    # Stage 1 — subdivide. Default target = strut/2: edge length 50 % of the
    # strut, which keeps Dijkstra's geodesic-distance error well under
    # strut/4 and stays comfortably under the 500k face cap on typical
    # 50–150 mm prints. The plan's `strut/4` target is finer than realistic
    # printing tolerance and blows the cap on small models.
    if target_edge_length is None:
        target_edge_length = strut_thickness / 2.0
    with progress.step(
        f"geodesic: subdivide mesh (target edge {target_edge_length:.3f} mm)"
    ):
        sub_mesh = subdivide_for_geodesic(mesh, target_edge_length)

    # Pre-compute face components once and reuse: assign_cell_labels needs
    # them for the per-patch Dijkstra, and the per-cell loop downstream
    # needs them to look up patch-boundary clipping half-planes.
    face_comp = compute_face_components(sub_mesh, sharp_angle_deg)

    # Smooth the subdivided mesh's vertex normals within each smooth
    # patch. After uniform subdivision every child face inherits its
    # parent's normal exactly, so trimesh's area-weighted vertex normals
    # are piecewise constant per parent face — and the prism walls
    # downstream pick up those discontinuities as stepped facets,
    # producing visibly jagged hole edges on low-poly inputs. Patch-aware
    # Laplacian smoothing diffuses normals only across same-patch
    # adjacencies, so sharp dihedrals (cube edges) are still preserved
    # exactly while smooth surfaces (sphere, filleted body) get a
    # continuously varying normal field.
    with progress.step("smooth vertex normals (patch-aware)"):
        smoothed_n = smooth_vertex_normals_within_patches(
            sub_mesh, face_comp, iterations=5
        )
        sub_mesh.vertex_normals = smoothed_n

    # Stage 2 — Dijkstra with sharp-edge barriers (Approach A). Edges whose
    # dihedral angle exceeds `sharp_angle_deg` get a high cost multiplier
    # so Voronoi cells stay within smooth patches (one cube face, one
    # filleted segment) rather than wrapping across corners — which would
    # produce skewed prisms manifold3d can't subtract cleanly.
    with progress.step(f"geodesic: Dijkstra on {len(sub_mesh.vertices)} vertices"):
        labels = assign_cell_labels(
            sub_mesh, seeds.points, sharp_angle_deg=sharp_angle_deg
        )
        face_labels = face_labels_from_vertex_labels(sub_mesh, labels)

    # For each seed, find which patch it sits on (used downstream to look
    # up patch-boundary clip half-planes).
    from trimesh.proximity import ProximityQuery as _PQ
    _pq_for_patch = _PQ(sub_mesh)
    _, _, seed_face_for_patch = _pq_for_patch.on_surface(seeds.points)
    seed_patch = face_comp[np.asarray(seed_face_for_patch, dtype=int)]

    # Per-patch maximum intra-patch dihedral. Used to distinguish a
    # "feature-y" patch (a CAD body's single patch with fillets that
    # span 30° of normal range) from a "smoothly curved" patch (a
    # sphere with intra-patch dihedrals under 3°). Wraparound clipping
    # downstream applies only to feature-y patches; without this
    # check the sphere's cells (which legitimately span a small range
    # of normals as the loop walks the great circle) get incorrectly
    # clipped.
    n_patches = int(face_comp.max()) + 1 if len(face_comp) else 0
    patch_max_dihedral_rad = np.zeros(n_patches, dtype=float)
    fa = sub_mesh.face_adjacency
    angles_all = sub_mesh.face_adjacency_angles
    if len(fa) > 0:
        for p_id in range(n_patches):
            same_patch = (
                (face_comp[fa[:, 0]] == p_id)
                & (face_comp[fa[:, 1]] == p_id)
            )
            if same_patch.any():
                patch_max_dihedral_rad[p_id] = float(angles_all[same_patch].max())

    # Stage 3 — extract closed boundary loops per cell.
    with progress.step("geodesic: extract boundary loops"):
        loops_per_cell = extract_cell_loops(sub_mesh, face_labels)
    progress.log(
        f"cells with boundary loops: {len(loops_per_cell)} / {len(seeds)}"
    )

    # Build proximity query once, reused by all loops (smoothing + inset).
    from trimesh.proximity import ProximityQuery
    proximity = ProximityQuery(sub_mesh)

    # Per-seed local curvature radius for prism cap-centroid placement.
    n_real = len(seeds)
    seed_tree = cKDTree(seeds.points) if n_real >= 2 else None
    K_curv = min(7, n_real)

    if resample_step is None:
        # Tuned to give ~10–20 resampled vertices for a typical cell (cube
        # face perimeter ~40 mm, strut 1.5 mm → step 4.5 mm → ~9 vertices,
        # ×6 Bézier = 54). Comparable to Phase 1's polygon size.
        resample_step = max(strut_thickness * 3.0, 2.0)

    stats = GeodesicCellStats(
        requested=n_real, built=0, no_loop=0, too_few_vertices=0, prism_failed=0
    )
    cells: list[trimesh.Trimesh] = []

    safety = max(1.0, shell_thickness)
    inset_distance = strut_thickness / 2.0

    iterator = progress.progress(range(n_real), desc="build cells", total=n_real)
    for s_idx in iterator:
        loops = loops_per_cell.get(s_idx, [])
        if not loops:
            stats.no_loop += 1
            continue
        # If multiple loops (rare, e.g. cell wrapping a handle), pick the
        # one with the largest perimeter — that's the principal boundary.
        loop = _pick_principal_loop(loops)
        if loop is None or len(loop) < 3:
            stats.too_few_vertices += 1
            continue

        seed = seeds.points[s_idx]
        seed_normal = seeds.normals[s_idx]

        loop = resample_loop_arclen(loop, sub_mesh, proximity, target_step=resample_step)
        if len(loop) < 3:
            stats.too_few_vertices += 1
            continue

        # Wraparound loop pre-filter. Applied only when:
        #   * The cell's home patch has no sharp-edge boundary (single-
        #     patch mesh — sphere, body, organic blob), and
        #   * The patch contains intermediate dihedrals (≥ 10°) that
        #     indicate features the loop could wrap around — i.e. it's
        #     a CAD-style mesh with fillets, NOT a smoothly curved
        #     surface (sphere, where intra-patch dihedrals are ≤ 3°).
        #
        # When both apply, the loop's wraparound vertices (loop visiting
        # an adjacent face past a fillet) are removed before the convex
        # hull. Otherwise (multi-patch mesh OR smooth-everywhere patch)
        # the full loop is used unchanged.
        patch_id_pre = int(seed_patch[s_idx])
        patch_has_boundary = len(
            patch_boundary_vertex_indices(sub_mesh, face_comp, patch_id_pre)
        ) >= 3
        patch_is_featured = math.degrees(
            patch_max_dihedral_rad[patch_id_pre]
        ) >= 10.0
        seed_n_arr = np.asarray(seed_normal, dtype=float)
        seed_n_arr = seed_n_arr / max(float(np.linalg.norm(seed_n_arr)), 1e-12)
        if (
            not patch_has_boundary
            and patch_is_featured
            and len(loop) > 0
        ):
            cos_with_seed = loop.normals @ seed_n_arr
            face_aligned = cos_with_seed > 0.5  # within 60°
            seed_3d = np.asarray(seed, dtype=float)
            rel = loop.positions - seed_3d
            tan_rel = rel - (rel @ seed_n_arr)[:, None] * seed_n_arr
            dist_in_tangent = np.linalg.norm(tan_rel, axis=1)
            max_polygon_radius = math.sqrt(
                float(mesh.area) / (max(n_real, 1) * math.pi)
            ) * 2.0
            within_radius = dist_in_tangent <= max_polygon_radius
            combined_keep = face_aligned & within_radius
            if combined_keep.sum() >= 3 and not combined_keep.all():
                loop = Loop(
                    positions=loop.positions[combined_keep],
                    face_ids=loop.face_ids[combined_keep],
                    normals=loop.normals[combined_keep],
                )

        # Polygon construction in the seed's 2D tangent plane (Phase 1's
        # geometry, fed by the geodesic loop). Doing the inset + Bézier in
        # 2D avoids the surface-snap-back bug: a loop vertex on a cube
        # edge, inset in 3D and re-projected via on_surface, lands right
        # back on the edge — leaving zero margin for the corner wall and
        # causing adjacent-face cells to eat into shared corner material.
        # In 2D tangent the inset shrinks the polygon by `strut/2`
        # unconditionally; surface projection happens only once, after
        # smoothing, when the final 2D polygon is lifted to the mesh.
        poly = convex_polygon_2d_in_tangent(loop, seed, seed_normal)
        if poly is None:
            stats.too_few_vertices += 1
            continue
        polygon_2d, u_basis, v_basis = poly

        # Three-branch dispatch on the cell's home patch:
        #
        #   flat (cube face): seed's tangent plane and the surface are
        #     identical, so do everything (convex hull, strut/2 inset,
        #     patch-boundary clip) in 2D.
        #
        #   curved with patch boundary (a fillet between two flat
        #     regions): need the 2D patch-boundary clip to enforce
        #     shell_thickness margin from the sharp edges, but the
        #     strut/2 must be applied on the surface to avoid the
        #     foreshortening that an orthogonal 2D inset would suffer.
        #
        #   curved with no patch boundary (sphere, organic blob): skip
        #     the 2D round-trip entirely. Pick hull vertices in 2D for
        #     selection only, then carry their ORIGINAL surface positions
        #     into the surface-aware inset. This avoids the orthogonal
        #     forward + on_surface backward composition that loses
        #     ~R·(θ − arctan(sin θ)) of cell radius on curved patches
        #     and that's what was making sphere walls visibly wider than
        #     cube walls.
        patch_id = int(seed_patch[s_idx])
        flat = patch_is_flat(sub_mesh, face_comp, patch_id)
        patch_clip = _patch_clip_halfplanes(
            sub_mesh, face_comp, patch_id, seed, u_basis, v_basis
        )
        if flat:
            inset_2d = _polygon_clip_and_inset(
                polygon_2d,
                strut_half=inset_distance,
                patch_clip_eqs=patch_clip,
                shell_thickness=shell_thickness,
            )
            if inset_2d is None or len(inset_2d) < 3:
                stats.too_few_vertices += 1
                continue
            inset_loop = project_polygon_2d_to_surface(
                inset_2d, seed, u_basis, v_basis, sub_mesh, proximity
            )
        elif patch_clip is None:
            # Curved with no patch boundary — sphere-like. Keep the
            # hull vertices' ORIGINAL surface positions; no 2D round-trip.
            hull_idx = convex_hull_indices_in_tangent(loop, seed, seed_normal)
            if hull_idx is None or len(hull_idx) < 3:
                stats.too_few_vertices += 1
                continue
            hull_loop = Loop(
                positions=loop.positions[hull_idx],
                face_ids=loop.face_ids[hull_idx],
                normals=loop.normals[hull_idx],
            )
            inset_loop = inset_loop_on_surface(
                hull_loop, sub_mesh, proximity, inset=inset_distance
            )
        else:
            # Curved with patch boundary (fillet). 2D patch clip only,
            # then project to surface, then surface-aware strut/2 inset.
            clipped_2d = _polygon_clip_and_inset(
                polygon_2d,
                strut_half=0.0,
                patch_clip_eqs=patch_clip,
                shell_thickness=shell_thickness,
            )
            if clipped_2d is None or len(clipped_2d) < 3:
                stats.too_few_vertices += 1
                continue
            clipped_loop = project_polygon_2d_to_surface(
                clipped_2d, seed, u_basis, v_basis, sub_mesh, proximity
            )
            inset_loop = inset_loop_on_surface(
                clipped_loop, sub_mesh, proximity, inset=inset_distance
            )
        if len(inset_loop) < 3:
            stats.too_few_vertices += 1
            continue
        # Bézier smoothing ON THE SURFACE — each sample snapped to the mesh.
        loop = bezier_smooth_on_surface(inset_loop, sub_mesh, proximity)
        if len(loop) < 3:
            stats.too_few_vertices += 1
            continue

        if seed_tree is None:
            R_local = float("inf")
        else:
            _, kidx = seed_tree.query(seed, k=K_curv)
            kidx = np.atleast_1d(kidx)
            kidx = kidx[kidx != s_idx][: K_curv - 1]
            R_local = _estimate_local_radius(
                seed, seed_normal, seeds.points[kidx], seeds.normals[kidx]
            )

        try:
            prism = build_prism_from_loop(
                loop=loop,
                mesh=sub_mesh,
                seed=seed,
                seed_normal=seed_normal,
                R_local=R_local,
                shell_thickness=shell_thickness,
                chamfer=chamfer,
                chamfer_inner=chamfer_inner,
                safety=safety,
            )
        except Exception:
            stats.prism_failed += 1
            continue
        if len(prism.faces) == 0:
            stats.prism_failed += 1
            continue

        cells.append(prism)
        stats.built += 1

    progress.warn(
        f"geodesic cells built: {stats.built} / {stats.requested} "
        f"(no-loop={stats.no_loop}, short={stats.too_few_vertices}, "
        f"prism-failed={stats.prism_failed})"
    )
    return cells, stats


def _pick_principal_loop(loops: list[Loop]) -> Loop | None:
    if not loops:
        return None
    best = max(loops, key=lambda L: _loop_perimeter(L))
    return best


def _loop_perimeter(loop: Loop) -> float:
    return float(np.linalg.norm(
        np.roll(loop.positions, -1, axis=0) - loop.positions, axis=1
    ).sum())
