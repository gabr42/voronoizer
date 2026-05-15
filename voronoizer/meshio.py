"""STL load/save with optional best-effort repair."""

from __future__ import annotations

from pathlib import Path

import trimesh

from voronoizer import progress


def load_stl(path: Path, repair: bool = False) -> trimesh.Trimesh:
    """Load an STL. If repair=True, attempt to make it watertight."""
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"file did not load as a single mesh: {path}")
    if len(mesh.faces) == 0:
        raise ValueError(f"mesh has no faces: {path}")

    progress.log(
        f"loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
        f"watertight={mesh.is_watertight}, winding_consistent={mesh.is_winding_consistent}"
    )

    if repair:
        with progress.step("repair input mesh"):
            mesh.merge_vertices()
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.fix_normals()
            trimesh.repair.fill_holes(mesh)
            mesh.process(validate=True)
        progress.log(
            f"after repair: watertight={mesh.is_watertight}, "
            f"winding_consistent={mesh.is_winding_consistent}"
        )

    if not mesh.is_watertight:
        progress.warn(
            "input mesh is not watertight; boolean results may be unreliable. "
            "Consider passing --repair."
        )

    return mesh


def save_stl(mesh: trimesh.Trimesh, path: Path) -> None:
    """Save as binary STL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="stl")
    progress.log(f"wrote {path} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
