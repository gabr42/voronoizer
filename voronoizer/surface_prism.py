"""Prism cutter construction from an inset surface loop.

`_build_prism_surface_aware` takes per-vertex `(P, n, d_out)` triples; this
module's job is to derive those triples directly from a surface loop and
hand them off.
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
    """Build a surface-aware prism cutter from an inset loop.

    For each loop vertex i:
      * `P_i` = `loop.positions[i]` (already on the mesh surface).
      * `n_i` = outward face normal of the face the vertex sits on.
      * `d_out_i` = local boundary tangent direction rotated outward in the
        surface tangent plane: `t × n` where `t = normalize(V[i+1] - V[i-1])`.
        This points from the cell interior toward where the un-inset boundary
        was, i.e. outward — the direction the chamfer expansion grows the
        ring back along.

    The boundary tangent and the outward direction are both purely surface-
    intrinsic; every prism vertex sits on the actual mesh even when the
    cell spans a curved or sharp-edged region of the surface.
    """
    if len(loop) < 3:
        raise ValueError(
            f"surface_prism.build_prism_from_loop: loop too short "
            f"({len(loop)} vertices)"
        )
    P = loop.positions
    _ = mesh  # kept in signature; no longer used directly here
    # Per-cell normal mode. The surface boundary often makes a cell span
    # a large angular range of surface normals — on the CAD body fixture
    # the mean per-cell normal spread is ~47° because each cell naturally
    # crosses a flat face plus a fillet plus another flat face. Using
    # per-vertex smoothed normals there gives prism walls that visibly
    # tilt along the cell perimeter.
    #
    # Decision:
    #
    #   * If the loop's normals are tightly clustered (spread ≤ 5°) —
    #     the cell sits inside a flat region — extrude the whole prism
    #     along the SINGLE average normal. Clean perpendicular cut.
    #
    #   * Moderate spread (5°–30°) — use per-vertex normals; they follow
    #     curvature smoothly enough that walls stay continuous.
    #
    #   * Large spread (> 30°) — cell straddles a feature boundary.
    #     Fall back to the seed's face normal for the whole cell so the
    #     prism becomes a clean perpendicular cut through the seed's
    #     local face. On a cell crossing a fillet the hole follows the
    #     seed face's direction even where the cell extends onto the
    #     fillet, but that looks cleaner than the stepped surface-
    #     following alternative.
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
        # Cell straddles a feature boundary; fall back to the seed's
        # face normal (see the decision comment above).
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

    # `_build_prism_surface_aware` uses `polygon_2d` only for vertex count,
    # so a zeros stand-in is harmless. P/n/d_out drive the actual ring geometry.
    polygon_2d_stub = np.zeros((len(loop), 2), dtype=float)

    # Boundary loops can wrap around sharp surface edges where two
    # adjacent loop vertices have very different normals. Even on a flat
    # input mesh (cube), the prism wall then passes exactly along the
    # cube's corner edge and manifold3d emits twin vertices unless the
    # outer/inner rings are lifted slightly off the surface. We force the
    # lift on regardless of `R_local`.
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
