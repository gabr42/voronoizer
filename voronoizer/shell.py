"""Hollow-shell construction by voxel erosion + boolean subtract."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import trimesh

from voronoizer import progress


# Soft cap on voxel grid size; ~27M voxels ≈ ~27 MB for bool mask, fine on most machines.
_MAX_VOXELS_PER_AXIS = 300


def _choose_pitch(mesh: trimesh.Trimesh, thickness: float) -> float:
    """Pick voxel pitch so the grid stays under the cap and resolves the thickness."""
    extents = mesh.extents
    desired = thickness / 4.0  # 4 voxels across the wall = decent quality
    max_dim = float(np.max(extents))
    min_pitch_for_cap = max_dim / _MAX_VOXELS_PER_AXIS
    pitch = max(desired, min_pitch_for_cap)
    if pitch > thickness / 2.0:
        progress.warn(
            f"voxel pitch raised to {pitch:.3f} mm for a {max_dim:.1f} mm model; "
            f"shell interior may be slightly coarser than the requested "
            f"{thickness:.2f} mm thickness."
        )
    return pitch


def _voxelize_filled(mesh: trimesh.Trimesh, pitch: float) -> trimesh.voxel.VoxelGrid:
    vox = mesh.voxelized(pitch=pitch)
    vox = vox.fill()
    progress.log(f"voxel grid: pitch={pitch:.3f} mm, shape={tuple(vox.matrix.shape)}")
    return vox


def _erode_to_inner_mesh(
    vox: trimesh.voxel.VoxelGrid, thickness: float, pitch: float
) -> trimesh.Trimesh | None:
    """Erode the solid voxel mask by `thickness` and return the inner cavity mesh.

    Returns None when erosion removes everything (object thinner than `thickness`).
    """
    iterations = max(1, int(round(thickness / pitch)))
    mask = vox.matrix
    eroded = ndi.binary_erosion(mask, iterations=iterations)
    if not eroded.any():
        return None
    inner_grid = trimesh.voxel.VoxelGrid(eroded, transform=vox.transform.copy())
    inner = inner_grid.marching_cubes
    if len(inner.faces) == 0:
        return None
    # marching_cubes returns vertices in voxel-index space; map them back into
    # world coordinates using the grid's transform.
    inner.apply_transform(vox.transform)
    return inner


def build_shell(mesh: trimesh.Trimesh, thickness: float) -> trimesh.Trimesh:
    """Return `mesh` hollowed into a shell of the given wall thickness."""
    if thickness <= 0:
        raise ValueError("thickness must be > 0")

    pitch = _choose_pitch(mesh, thickness)
    with progress.step("voxelize input"):
        vox = _voxelize_filled(mesh, pitch)

    with progress.step("erode inward"):
        inner = _erode_to_inner_mesh(vox, thickness, pitch)

    if inner is None:
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
