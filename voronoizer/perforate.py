"""Boolean subtraction of Voronoi cells from the shell."""

from __future__ import annotations

import trimesh

from voronoizer import progress


_DEFAULT_BATCH = 25


def perforate(
    shell: trimesh.Trimesh,
    cells: list[trimesh.Trimesh],
    batch_size: int = _DEFAULT_BATCH,
) -> trimesh.Trimesh:
    """Subtract `cells` from `shell`, in batches via the manifold backend."""
    if not cells:
        progress.warn("no Voronoi cells to subtract; returning shell as-is.")
        return shell

    result = shell
    n_batches = (len(cells) + batch_size - 1) // batch_size
    progress.log(
        f"subtracting {len(cells)} cells in {n_batches} batches of up to {batch_size}"
    )

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        batch = cells[start:start + batch_size]
        with progress.step(f"batch {batch_idx + 1}/{n_batches} ({len(batch)} cells)"):
            result = trimesh.boolean.difference(
                [result, *batch], engine="manifold"
            )

    if not isinstance(result, trimesh.Trimesh) or len(result.faces) == 0:
        raise RuntimeError("perforation produced an empty mesh")
    return result


def cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean up a perforated mesh before writing it to disk.

    On curved-surface chamfered inputs the manifold backend produces a
    valid 2-manifold that uses *twin vertices* (two distinct topological
    vertices at the same 3D location) wherever adjacent Voronoi cells are
    near-tangent. The mesh is watertight in memory, but slicers and other
    consumers that dedupe vertices on load see the twins as coincident and
    end up with broken topology.

    This routine mirrors what a 3D-printing slicer does internally:

      1. merge_vertices()                — collapse coincident twins;
      2. update_faces(nondegenerate)     — drop the now-zero-area triangles
                                            produced by the merge;
      3. update_faces(unique_faces)      — drop any duplicate triangles;
      4. fill_holes()                    — patch small holes opened by 2/3;
      5. fix_normals()                   — restore consistent outward winding.

    On already-clean inputs (flat-face chamfer, no chamfer at all) the
    sequence is a no-op.
    """
    f0, v0 = len(mesh.faces), len(mesh.vertices)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    progress.log(
        f"cleanup: vertices {v0} -> {len(mesh.vertices)} "
        f"(merged {v0 - len(mesh.vertices)}), faces {f0} -> {len(mesh.faces)} "
        f"(removed {f0 - len(mesh.faces)})"
    )
    return mesh
