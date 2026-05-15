"""Top-level pipeline: load → shell → seeds → twins → mirrors → cells → cut → write."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from voronoizer import progress
from voronoizer.meshio import load_stl, save_stl
from voronoizer.mirror_seeds import (
    compute_boundary_mirrors,
    compute_sharp_edge_twins,
)
from voronoizer.perforate import perforate
from voronoizer.seeding import Seeds, sample_seeds
from voronoizer.shell import build_shell
from voronoizer.voronoi_cells import build_shrunken_cells


def _auto_edge_neighbor_dist(
    surface_area: float, count: int, strut_thickness: float
) -> float:
    """Cap on how far a seed may be from an edge and still get a twin/mirror.

    Sized to ~2x the natural cell radius — bisectors with neighbours past
    this distance fall outside the cell anyway and have no effect."""
    if count < 1 or surface_area <= 0:
        return 10.0 * strut_thickness
    cell_radius = math.sqrt(surface_area / count) / 2.0
    return max(strut_thickness * 4.0, cell_radius * 2.5)


def _merge_seeds(a: Seeds, b_points: np.ndarray, b_normals: np.ndarray) -> Seeds:
    return Seeds(
        points=np.vstack([a.points, b_points]) if len(b_points) else a.points,
        normals=np.vstack([a.normals, b_normals]) if len(b_normals) else a.normals,
    )


def run(
    input_path: Path,
    output_path: Path,
    shell_thickness: float,
    holes: int,
    strut_thickness: float,
    top_bottom_only: bool,
    normal_angle_deg: float,
    seed: int | None,
    repair: bool,
    edge_margin: float | None = None,
    chamfer: float = 0.0,
) -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    with progress.step("load STL"):
        mesh = load_stl(input_path, repair=repair)

    with progress.step("build hollow shell"):
        shell = build_shell(mesh, shell_thickness)

    with progress.step("sample seed points"):
        seeds = sample_seeds(
            mesh=mesh,
            count=holes,
            top_bottom_only=top_bottom_only,
            angle_deg=normal_angle_deg,
            rng=rng,
            strut_thickness=strut_thickness,
            edge_margin=edge_margin,
        )

    # Mesh to use for edge analysis: the same one sample_seeds drew from
    # (a submesh in --top-bottom-only mode, otherwise the full mesh).
    edge_source = mesh
    if top_bottom_only:
        from voronoizer.seeding import _top_bottom_face_mask  # type: ignore[attr-defined]
        mask = _top_bottom_face_mask(mesh, normal_angle_deg)
        if mask.any():
            edge_source = mesh.submesh([np.where(mask)[0]], append=True)

    with progress.step("compute twin / mirror seeds"):
        neighbor_dist = _auto_edge_neighbor_dist(
            float(edge_source.area), len(seeds), strut_thickness
        )
        twins = compute_sharp_edge_twins(
            mesh=edge_source,
            seed_points=seeds.points,
            seed_normals=seeds.normals,
            max_twin_dist=neighbor_dist,
        )
        seeds = _merge_seeds(seeds, twins.points, twins.normals)
        mirrors = compute_boundary_mirrors(
            mesh=edge_source,
            seed_points=seeds.points,
            seed_normals=seeds.normals,
            max_mirror_dist=neighbor_dist,
        )
        progress.log(
            f"total real seeds: {len(seeds)} (original {len(seeds) - len(twins)} + twins {len(twins)})"
        )

    with progress.step("build & smooth cells"):
        cells, _stats = build_shrunken_cells(
            seeds=seeds,
            mesh=edge_source,
            shell_thickness=shell_thickness,
            strut_thickness=strut_thickness,
            mirror_seeds=mirrors,
            chamfer=chamfer,
        )

    with progress.step("perforate shell"):
        perforated = perforate(shell, cells)

    with progress.step("write STL"):
        save_stl(perforated, output_path)

    progress.log(f"done in {time.perf_counter() - t0:.2f}s")
