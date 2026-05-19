"""Geodesic Voronoi engine — Stage 6 of the Phase 2 pipeline.

Build a prism cutter from an inset surface loop. Reuses Phase 1's
`_build_prism_surface_aware` — Phase 1 already takes per-vertex
`(P, n, d_out)` triples; the only thing Phase 2 needs to supply is those
triples computed directly from the surface loop rather than projected from
a tangent-plane polygon.

Stage 7 of the pipeline (boolean subtract) is unchanged from Phase 1.
"""

from __future__ import annotations

import numpy as np
import trimesh

from voronoizer.surface_boundary import Loop, _normalize_rows
from voronoizer.voronoi_cells import _build_prism_surface_aware


def build_prism_from_loop(
    loop: Loop,
    mesh: trimesh.Trimesh,
    seed: np.ndarray,
    seed_normal: np.ndarray,
    R_local: float,
    shell_thickness: float,
    chamfer: float,
    safety: float,
    chamfer_inner: float | None = None,
) -> trimesh.Trimesh:
    """Build a surface-aware prism cutter from a Phase-2 inset loop.

    For each loop vertex i:
      * `P_i` = `loop.positions[i]` (already on the mesh surface).
      * `n_i` = outward face normal of the face the vertex sits on.
      * `d_out_i` = local boundary tangent direction rotated outward in the
        surface tangent plane: `t × n` where `t = normalize(V[i+1] - V[i-1])`.
        This points from the cell interior toward where the un-inset boundary
        was, i.e. outward — the direction the chamfer expansion grows the
        ring back along.

    The boundary tangent and the outward direction are both purely surface-
    intrinsic; no tangent-plane projection from the seed is involved. That's
    the whole point of Phase 2 — every prism vertex sits on the actual mesh
    even when the cell spans a curved or sharp-edged region of the surface.
    """
    if len(loop) < 3:
        raise ValueError(
            f"surface_prism.build_prism_from_loop: loop too short "
            f"({len(loop)} vertices)"
        )
    P = loop.positions
    _ = mesh  # kept in signature; no longer used directly here
    # Per-cell normal mode. The geodesic boundary often makes a cell
    # span a large angular range of surface normals — on the user's CAD
    # body, the mean per-cell normal spread is ~47° because each cell
    # naturally crosses a flat face plus a fillet plus another flat
    # face. Using per-vertex smoothed normals there gives prism walls
    # that visibly tilt along the cell perimeter — the "jagged cutter"
    # the user reported.
    #
    # Decision:
    #
    #   * If the loop's normals are tightly clustered (spread ≤ 5°) —
    #     the cell sits inside a flat region — extrude the whole prism
    #     along the SINGLE average normal. Clean perpendicular cut.
    #
    #   * If the spread is moderate-to-large (any spread above 5° on a
    #     mesh with sharp dihedrals at all) — use the SEED's normal for
    #     the whole cell. This is the Phase-1 "tangent plane" behaviour:
    #     the prism is a simple polygonal extrusion perpendicular to
    #     the seed's face, regardless of how the cell's loop wraps. On
    #     a mostly-flat CAD body that gives smooth holes; on a sphere
    #     each cell becomes a flat-cap circular hole (still smooth).
    #     The previous per-vertex approach was trying to follow surface
    #     curvature but the smoothed normals weren't smooth enough to
    #     do it without visible faceting on coarse low-poly inputs.
    avg_n = loop.normals.mean(axis=0)
    avg_n_len = float(np.linalg.norm(avg_n))
    if avg_n_len > 1e-9:
        avg_n = avg_n / avg_n_len
        cos_min = float((loop.normals @ avg_n).min())
        max_deviation_deg = float(
            np.degrees(np.arccos(np.clip(cos_min, -1.0, 1.0)))
        )
    else:
        max_deviation_deg = float("inf")

    if max_deviation_deg < 5.0:
        # Cell sits inside a flat region. Single average normal → clean
        # perpendicular cut.
        n = np.broadcast_to(avg_n, loop.normals.shape).copy()
    elif max_deviation_deg < 30.0:
        # Cell wraps a moderately curved region (typical sphere cell at
        # ~21° spread). Per-vertex normals follow the curvature
        # smoothly enough that the walls look continuous.
        n = loop.normals
    else:
        # Cell straddles a feature boundary (a fillet between flat
        # regions on the body — typical spread is 30-90°). Per-vertex
        # smoothed normals jump abruptly at the parent-face boundaries
        # the smoothing couldn't fully average out, giving the user-
        # visible "jagged cutter" facets. Fall back to extrusion along
        # the SEED's face normal: the prism becomes a clean perpendicular
        # cut through the seed's local face, like Phase 1's tangent-plane
        # extrusion. On a cell crossing a fillet that means the hole
        # follows the seed's face's direction even where the cell
        # extends onto the fillet — but that looks cleaner than the
        # stepped surface-following alternative.
        sn = np.asarray(seed_normal, dtype=float)
        sn = sn / max(float(np.linalg.norm(sn)), 1e-12)
        n = np.broadcast_to(sn, loop.normals.shape).copy()
    # Forward boundary tangent at each vertex (central difference around the
    # closed loop).
    fwd = _normalize_rows(np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0))
    # Outward in the surface tangent plane: t × n points from the cell
    # interior outward (cell is on the LEFT of the walking direction, so the
    # right perpendicular is outward).
    d_out = _normalize_rows(np.cross(fwd, n))

    # Phase 1's prism builder uses `polygon_2d` only for vertex count, so a
    # zeros stand-in is harmless. P/n/d_out drive the actual ring geometry.
    polygon_2d_stub = np.zeros((len(loop), 2), dtype=float)

    # Phase 2 boundary loops can wrap around sharp surface edges where two
    # adjacent loop vertices have very different normals. Even on a flat
    # input mesh (cube), the prism wall then passes exactly along the
    # cube's corner edge and manifold3d emits twin vertices unless the
    # outer/inner rings are lifted slightly off the surface. We force the
    # lift on for the geodesic engine regardless of `R_local`.
    return _build_prism_surface_aware(
        polygon_2d=polygon_2d_stub,
        P=P,
        n=n,
        d_out=d_out,
        seed=np.asarray(seed, dtype=float),
        seed_normal=np.asarray(seed_normal, dtype=float),
        R_local=R_local,
        shell_thickness=shell_thickness,
        chamfer=chamfer,
        safety=safety,
        force_lift=True,
        chamfer_inner=chamfer_inner,
    )
