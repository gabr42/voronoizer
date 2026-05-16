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
    # Use raw face normals at each loop vertex. Smoothed (barycentric) normals
    # were tried for sharp-edge cases but did not reduce manifold3d twin-vertex
    # artefacts on a cube — and slightly hurt on the sphere. Keeping face
    # normals matches the plan's "each boundary vertex already has a surface
    # position and a face normal".
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    n = face_normals[loop.face_ids]
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
    )
