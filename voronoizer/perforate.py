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
