"""Hollow-shell construction by per-vertex offset.

For each vertex we find the point that lies `thickness` mm inside *every*
incident face plane, in a least-squares sense. On a smooth surface this
collapses to a vertex-normal offset; on a cube corner (three orthogonal
face planes) it solves uniquely to the correct inner corner; on a body
mixing flat patches and fillets each vertex gets the right answer for
its local geometry. The result is exact on simple shapes (cube inner
volume matches the analytic formula to floating-point precision) and
smooth everywhere else — none of the voxel-grid stairsteps the previous
binary-erosion implementation produced.
"""

from __future__ import annotations

import numpy as np
import trimesh

from voronoizer import progress


def _offset_vertices_inward(
    mesh: trimesh.Trimesh, thickness: float
) -> np.ndarray:
    """Compute new vertex positions offset inward by `thickness` mm.

    For each vertex V incident to faces with normals (n_1, ..., n_k) and
    a point on each face plane, the corresponding offset planes share
    the same normals and pass `thickness` mm *inside* the originals. We
    solve

        argmin_x  Σ_i  ( n_i · x − (n_i · V − thickness) )²

    by ordinary least-squares. The system is well-determined for
    vertices on a smooth surface (lots of nearly-parallel constraints
    project the vertex along the average normal) and at sharp corners
    (3 linearly-independent constraints solve uniquely).
    """
    F = mesh.faces
    V = mesh.vertices
    fn = mesh.face_normals
    NV = len(V)

    # Group face indices by vertex.
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
        # Target distance for each plane: n · x = n · V[vi] − thickness.
        b = (n * V[vi]).sum(axis=1) - thickness
        # lstsq handles both well-conditioned and over-/under-determined cases.
        x, *_ = np.linalg.lstsq(n, b, rcond=None)
        new_v[vi] = x
    return new_v


def build_shell(mesh: trimesh.Trimesh, thickness: float) -> trimesh.Trimesh:
    """Return `mesh` hollowed into a shell of the given wall thickness."""
    if thickness <= 0:
        raise ValueError("thickness must be > 0")

    if len(mesh.faces) == 0:
        raise RuntimeError("build_shell: input mesh has no faces")

    with progress.step("offset inner cavity"):
        inner_vertices = _offset_vertices_inward(mesh, thickness)
        inner = trimesh.Trimesh(
            vertices=inner_vertices, faces=mesh.faces, process=False
        )
    if not inner.is_watertight:
        # Highly non-convex inputs can self-intersect when offset by more
        # than the local feature size. We let the manifold boolean cope —
        # it usually still produces a sensible shell — but warn so the
        # user knows the result may have unexpected geometry.
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
        shell = trimesh.boolean.difference([mesh, inner], engine="manifold")

    if not isinstance(shell, trimesh.Trimesh) or len(shell.faces) == 0:
        raise RuntimeError("shell construction produced an empty mesh")

    progress.log(f"shell: {len(shell.vertices)} verts, {len(shell.faces)} faces")
    return shell
