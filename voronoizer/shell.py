"""Hollow-shell construction by per-vertex offset.

For meshes with sharp dihedrals (a cube), each vertex's inner position
is the least-squares intersection of every incident face's offset
plane — three orthogonal planes at a cube corner solve uniquely to the
correct inner corner, so the cube's inner volume matches the analytic
formula to floating-point precision.

For meshes with no sharp dihedrals (a sphere, a CAD body with only
fillets), the LSQ approach is replaced by *subdivision plus
smoothed-vertex-normal offset*: subdivide the input mesh down to roughly
shell_thickness edge length, smooth the vertex normals (Laplacian
iterations diffused only across same-patch edges so any sharp features
that do exist are preserved), then offset each subdivided vertex by
shell_thickness along its smoothed normal. The resulting inner mesh has
many small triangles whose orientations track a smooth offset surface,
which eliminates the per-parent-face kinks that prism cuts inherit when
the inner mesh is a 168-triangle piecewise-flat offset of a low-poly
input.
"""

from __future__ import annotations

import numpy as np
import trimesh

from voronoizer import progress


def _offset_vertices_inward_lsq(
    mesh: trimesh.Trimesh, thickness: float
) -> np.ndarray:
    """Per-vertex least-squares offset.

    For each vertex V incident to faces with normals (n_1, ..., n_k) we
    solve

        argmin_x  Σ_i  ( n_i · x − (n_i · V − thickness) )²

    by ordinary least-squares. Well-determined for vertices on a smooth
    surface (lots of nearly-parallel constraints project the vertex
    along the average normal) and at sharp corners (3 linearly-
    independent constraints solve uniquely).
    """
    F = mesh.faces
    V = mesh.vertices
    fn = mesh.face_normals
    NV = len(V)

    vf: list[list[int]] = [[] for _ in range(NV)]
    for f_idx, face in enumerate(F):
        for v_i in face:
            vf[int(v_i)].append(f_idx)

    new_v = V.copy()
    for vi in range(NV):
        faces_idx = vf[vi]
        if not faces_idx:
            continue
        n = fn[faces_idx]
        b = (n * V[vi]).sum(axis=1) - thickness
        x, *_ = np.linalg.lstsq(n, b, rcond=None)
        new_v[vi] = x
    return new_v


def _build_shell_smooth(
    mesh: trimesh.Trimesh, thickness: float, sharp_angle_deg: float
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Subdivide + smoothed-vertex-normal offset for a smooth inner surface.

    Use when the input has no sharp dihedrals: the LSQ offset on a coarse
    mesh would give a piecewise-flat inner surface (one flat triangle per
    input face) and prism cuts on the inner side would inherit the kinks
    at every original face boundary. Subdividing to ~thickness/2 edges
    and offsetting each vertex along its (patch-aware Laplacian-smoothed)
    vertex normal produces an inner mesh whose triangle orientations
    track a smooth offset surface, so the cuts come out smooth.

    Returns `(outer, inner)`: the outer is the *subdivided* input (no
    geometry change, just denser tessellation) and the inner is offset
    inward by `thickness`.
    """
    from voronoizer.surface_voronoi import (
        face_components,
        smooth_vertex_normals_within_patches,
        subdivide_for_geodesic,
    )

    # Coarser than the cell-engine subdivision: just enough that the inner
    # surface looks smooth, without inflating the shell's face count to
    # the point where manifold3d's boolean against it starts producing
    # many near-tangency artefacts. Empirically thickness × 2 keeps the
    # body case in the same scale as the cell pipeline (~10k shell faces
    # vs ~170k for the cell mesh) while still erasing the per-parent-
    # face piecewise-flat kinks the user reported.
    target_edge = max(thickness * 2.0, 1.0)
    sub = subdivide_for_geodesic(mesh, target_edge_length=target_edge)

    fc = face_components(sub, sharp_angle_deg=sharp_angle_deg)
    smoothed_n = smooth_vertex_normals_within_patches(sub, fc, iterations=10)

    inner_v = sub.vertices - thickness * smoothed_n
    inner = trimesh.Trimesh(vertices=inner_v, faces=sub.faces, process=False)
    return sub, inner


def build_shell(
    mesh: trimesh.Trimesh,
    thickness: float,
    sharp_angle_deg: float = 25.0,
) -> trimesh.Trimesh:
    """Return `mesh` hollowed into a shell of the given wall thickness.

    `sharp_angle_deg` controls the dihedral threshold used to decide whether
    the input is multi-patch (LSQ offset on the original mesh) or
    single-patch (subdivide + smoothed-normal offset). Must match the
    pipeline-wide `--soft-edge-angle` so the shell's patch decomposition
    agrees with the cell engine's downstream.
    """
    if thickness <= 0:
        raise ValueError("thickness must be > 0")
    if len(mesh.faces) == 0:
        raise RuntimeError("build_shell: input mesh has no faces")

    # Path selection: if the mesh has any sharp dihedrals (multi-patch),
    # use per-vertex LSQ on the input mesh — that solves cube corners
    # exactly. If the mesh is fully smooth (one patch), use subdivision
    # plus smoothed-vertex-normal offset, so the inner surface is dense
    # and tracks a smooth offset (avoids visibly jagged hole edges on the
    # inside when cells perforate a low-poly smooth body).
    from voronoizer.surface_voronoi import face_components
    fc = face_components(mesh, sharp_angle_deg=sharp_angle_deg)
    n_patches = int(fc.max()) + 1 if len(fc) else 0

    if n_patches >= 2:
        with progress.step("offset inner cavity (LSQ; sharp features)"):
            inner_vertices = _offset_vertices_inward_lsq(mesh, thickness)
            inner = trimesh.Trimesh(
                vertices=inner_vertices, faces=mesh.faces, process=False
            )
        outer = mesh
    else:
        with progress.step("offset inner cavity (smoothed normals; no sharp edges)"):
            outer, inner = _build_shell_smooth(mesh, thickness, sharp_angle_deg)

    if not inner.is_watertight:
        progress.warn(
            "build_shell: inner offset surface is non-watertight "
            "(likely a feature thinner than 2 × shell_thickness, or a "
            "concavity whose neighbourhood self-intersects). The boolean "
            "subtract still runs, but the resulting wall thickness may be "
            "uneven near those features."
        )

    if abs(inner.volume) <= 0:
        progress.warn(
            "shell thickness exceeds the thinnest feature of the input; "
            "the object will be kept solid."
        )
        return mesh.copy()

    with progress.step("subtract inner cavity"):
        shell = trimesh.boolean.difference([outer, inner], engine="manifold")

    if not isinstance(shell, trimesh.Trimesh) or len(shell.faces) == 0:
        raise RuntimeError("shell construction produced an empty mesh")

    progress.log(f"shell: {len(shell.vertices)} verts, {len(shell.faces)} faces")
    return shell
