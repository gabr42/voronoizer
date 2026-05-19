"""Top-level pipeline: load → shell → seeds → twins → mirrors → cells → cut → write."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import trimesh

from voronoizer import progress
from voronoizer.meshio import load_stl, save_stl
from voronoizer.mirror_seeds import (
    compute_boundary_mirrors,
    compute_sharp_edge_twins,
)
from voronoizer.perforate import perforate
from voronoizer.seeding import Seeds, sample_seeds, sample_seeds_per_patch
from voronoizer.shell import build_shell
from voronoizer.surface_pipeline import build_geodesic_cells
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


def _clip_cutters_to_bbox(
    cutters: list[trimesh.Trimesh],
    bounds: np.ndarray,
    margin: float,
) -> list[trimesh.Trimesh]:
    """Intersect each cutter with an axis-aligned box around the input
    model (expanded by `margin` on each side). Voronoi cells whose seeds
    sit near sparse-neighbour regions can have polygons that extend far
    past the model in one direction; the resulting cutter is still
    correct (the boolean clips it at the shell anyway) but ugly to look
    at when exported via --cutters. Clipping each cutter to roughly the
    model's bounding box gives a cleaner inspection mesh.
    """
    lo = np.asarray(bounds[0], dtype=float) - margin
    hi = np.asarray(bounds[1], dtype=float) + margin
    center = (lo + hi) / 2.0
    extents = hi - lo
    transform = np.eye(4)
    transform[:3, 3] = center
    box = trimesh.creation.box(extents=extents, transform=transform)

    clipped: list[trimesh.Trimesh] = []
    for cutter in cutters:
        try:
            r = trimesh.boolean.intersection([cutter, box], engine="manifold")
            if isinstance(r, trimesh.Trimesh) and len(r.faces) > 0:
                clipped.append(r)
            else:
                clipped.append(cutter)
        except Exception:
            clipped.append(cutter)
    return clipped


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
    chamfer_inner: float | None = None,
    soft_edge_angle_deg: float = 25.0,
    shell_only: bool = False,
    cutters_only: bool = False,
    engine: str = "geodesic",
    target_edge_length: float | None = None,
) -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    with progress.step("load STL"):
        mesh = load_stl(input_path, repair=repair)

    with progress.step("build hollow shell"):
        shell = build_shell(mesh, shell_thickness)

    if shell_only:
        with progress.step("write STL"):
            save_stl(shell, output_path)
        progress.log(f"done in {time.perf_counter() - t0:.2f}s")
        return

    # Sharp-edge handling differs by engine:
    #   - tangent uses --soft-edge-angle to REJECT seeds near sharp edges.
    #     Sampling is plain Poisson-disk on the whole surface; sharp-edge
    #     rejection then filters out candidates near sharp edges.
    #   - geodesic uses --soft-edge-angle to partition the mesh into smooth
    #     patches. Seeds are distributed PER PATCH (proportional to patch
    #     area, minimum one per patch). The per-patch Voronoi labelling
    #     requires every smooth patch to have at least one seed; plain
    #     Poisson on the whole surface routinely leaves small patches
    #     unseeded, which forced 3D-Euclidean fallback assignments and
    #     produced broken cells that span patch boundaries.
    if engine == "tangent":
        with progress.step("sample seed points"):
            seeds = sample_seeds(
                mesh=mesh,
                count=holes,
                top_bottom_only=top_bottom_only,
                angle_deg=normal_angle_deg,
                rng=rng,
                strut_thickness=strut_thickness,
                edge_margin=edge_margin,
                sharp_edge_angle_deg=soft_edge_angle_deg,
            )
    else:
        with progress.step("sample seed points (per patch)"):
            seeds = sample_seeds_per_patch(
                mesh=mesh,
                count=holes,
                top_bottom_only=top_bottom_only,
                angle_deg=normal_angle_deg,
                rng=rng,
                strut_thickness=strut_thickness,
                sharp_edge_angle_deg=soft_edge_angle_deg,
            )

    # Mesh to use for edge analysis: the same one sample_seeds drew from
    # (a submesh in --top-bottom-only mode, otherwise the full mesh).
    edge_source = mesh
    if top_bottom_only:
        from voronoizer.seeding import _top_bottom_face_mask  # type: ignore[attr-defined]
        mask = _top_bottom_face_mask(mesh, normal_angle_deg)
        if mask.any():
            edge_source = mesh.submesh([np.where(mask)[0]], append=True)

    if engine == "tangent":
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
                f"total real seeds: {len(seeds)} "
                f"(original {len(seeds) - len(twins)} + twins {len(twins)})"
            )

        with progress.step("build & smooth cells (tangent engine)"):
            cells, _stats = build_shrunken_cells(
                seeds=seeds,
                mesh=edge_source,
                shell_thickness=shell_thickness,
                strut_thickness=strut_thickness,
                mirror_seeds=mirrors,
                chamfer=chamfer,
                chamfer_inner=chamfer_inner,
            )
    elif engine == "geodesic":
        with progress.step("build cells (geodesic engine)"):
            cells, _stats = build_geodesic_cells(
                seeds=seeds,
                mesh=edge_source,
                shell_thickness=shell_thickness,
                strut_thickness=strut_thickness,
                chamfer=chamfer,
                chamfer_inner=chamfer_inner,
                target_edge_length=target_edge_length,
                sharp_angle_deg=soft_edge_angle_deg,
            )
    else:
        raise ValueError(
            f"pipeline.run: unknown engine '{engine}' "
            f"(expected 'geodesic' or 'tangent')"
        )

    if cutters_only:
        if not cells:
            raise RuntimeError("no cutters to export")
        with progress.step("clip cutters to model bounds"):
            cells = _clip_cutters_to_bbox(
                cells, mesh.bounds, margin=shell_thickness
            )
        with progress.step("concatenate cutters"):
            cutters_mesh = trimesh.util.concatenate(cells)
        with progress.step("write STL"):
            save_stl(cutters_mesh, output_path)
        progress.log(f"done in {time.perf_counter() - t0:.2f}s")
        return

    with progress.step("perforate shell"):
        perforated = perforate(shell, cells)

    # Post-process: drop disconnected leftover components and fix
    # face winding. On CAD bodies with small fillet radii relative to the
    # shell thickness, the boolean leaves small pockets of wall material
    # un-cut where the cell prism's per-vertex extrusion couldn't fully
    # enclose the fillet's curved wall geometry. Manifold3d additionally
    # emits stray 1-2 face fragments with essentially zero volume as
    # numerical noise. Drop both: components whose volume is < 1 % of the
    # main shell's, AND components with fewer than 4 faces or near-zero
    # volume regardless of ratio.
    with progress.step("clean perforated output"):
        before_faces = len(perforated.faces)
        parts = perforated.split(only_watertight=False)
        if len(parts) > 1:
            parts = sorted(parts, key=lambda p: abs(p.volume), reverse=True)
            main = parts[0]
            main_vol = abs(main.volume)
            keep = [main]
            dropped_vol = 0.0
            dropped_count = 0
            for p in parts[1:]:
                v = abs(p.volume)
                if v < 0.01 * main_vol or len(p.faces) < 4 or v < 1e-6:
                    dropped_vol += v
                    dropped_count += 1
                else:
                    keep.append(p)
            if dropped_count > 0:
                progress.warn(
                    f"dropped {dropped_count} disconnected leftover component(s) "
                    f"({dropped_vol:.3f} mm³ total) — artefacts of the boolean "
                    f"leaving slivers at fillet near-tangencies"
                )
            if len(keep) == 1:
                perforated = keep[0]
            else:
                perforated = trimesh.util.concatenate(keep)
        # Re-orient faces so normals point consistently outward. Manifold3d
        # occasionally emits faces with flipped winding which slicers then
        # report as "facets reversed".
        perforated.fix_normals()
        if len(perforated.faces) != before_faces:
            progress.log(
                f"perforated: {before_faces} → {len(perforated.faces)} faces "
                f"after cleanup"
            )

    with progress.step("write STL"):
        save_stl(perforated, output_path)

    progress.log(f"done in {time.perf_counter() - t0:.2f}s")
