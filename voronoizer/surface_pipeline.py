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

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from voronoizer import progress
from voronoizer.seeding import Seeds
from voronoizer.surface_boundary import (
    Loop,
    bezier_smooth_on_surface,
    dedupe_loop,
    extract_cell_loops,
    inset_loop_on_surface,
    resample_loop_arclen,
)
from voronoizer.surface_prism import build_prism_from_loop
from voronoizer.surface_voronoi import (
    assign_cell_labels,
    face_labels_from_vertex_labels,
    subdivide_for_geodesic,
)
from voronoizer.voronoi_cells import _estimate_local_radius


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

    # Stage 2 — Dijkstra.
    with progress.step(f"geodesic: Dijkstra on {len(sub_mesh.vertices)} vertices"):
        labels = assign_cell_labels(sub_mesh, seeds.points)
        face_labels = face_labels_from_vertex_labels(sub_mesh, labels)

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

        loop = resample_loop_arclen(loop, sub_mesh, proximity, target_step=resample_step)
        if len(loop) < 3:
            stats.too_few_vertices += 1
            continue
        loop = bezier_smooth_on_surface(loop, sub_mesh, proximity)
        loop = inset_loop_on_surface(loop, sub_mesh, proximity, inset=inset_distance)
        # Drop near-duplicate vertices that Bézier + inset can leave behind
        # near sharp edges (segments as short as 0.004 mm on a cube).
        # Threshold is 5 % of the resample step — generous enough to clean
        # actual slivers without removing meaningful geometry.
        loop = dedupe_loop(loop, sub_mesh, min_segment_length=resample_step * 0.05)
        if len(loop) < 3:
            stats.too_few_vertices += 1
            continue

        seed = seeds.points[s_idx]
        seed_normal = seeds.normals[s_idx]

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

    progress.log(
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
