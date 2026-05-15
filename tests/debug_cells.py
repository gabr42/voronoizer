"""Diagnostic: build cells and report each one's bbox / wall coverage."""

from pathlib import Path

import numpy as np
import trimesh

from voronoizer.meshio import load_stl
from voronoizer.seeding import sample_seeds
from voronoizer.shell import build_shell
from voronoizer.voronoi_cells import build_shrunken_cells

THICKNESS = 2.0
HOLES = 60
STRUT = 1.5
SEED = 42
INPUT = Path(r"H:\RAZVOJ\voronoizer\tests\data\cube_100mm.stl")
OUT_DIR = INPUT.parent

mesh = load_stl(INPUT)
shell = build_shell(mesh, THICKNESS)
rng = np.random.default_rng(SEED)
seeds = sample_seeds(mesh, HOLES, top_bottom_only=False, angle_deg=30.0, rng=rng)
cells, stats = build_shrunken_cells(
    seeds=seeds,
    mesh=mesh,
    shell_thickness=THICKNESS,
    strut_thickness=STRUT,
)
print(f"built {len(cells)} cells")

# Per-cell: extent along the seed's outward normal.
suspicious = []
for i, cell in enumerate(cells):
    n = seeds.normals[i]
    p = seeds.points[i]
    v = np.asarray(cell.vertices)
    proj = v @ n  # signed distance along seed normal
    outward_extent = proj.max() - float(np.dot(p, n))
    inward_extent = float(np.dot(p, n)) - proj.min()
    bbox_extent = cell.extents
    if outward_extent < THICKNESS * 0.5 or inward_extent < THICKNESS * 1.0:
        suspicious.append((i, outward_extent, inward_extent, bbox_extent, p, n))

print(f"suspicious (cell may not span wall): {len(suspicious)}")
for s in suspicious[:10]:
    print(f"  seed#{s[0]:>2d}  out={s[1]:+.2f}  in={s[2]:+.2f}  bbox={s[3]}  "
          f"p={s[4]}  n={s[5]}")

# Export union of cells for visual inspection.
union = trimesh.util.concatenate(cells)
union.export(OUT_DIR / "cube_voronoi_cells_union.stl")
print(f"wrote {OUT_DIR / 'cube_voronoi_cells_union.stl'}")

# Export shell + cells overlaid as two-color (we'll just dump as separate STLs).
shell.export(OUT_DIR / "cube_voronoi_shell_only.stl")
print(f"wrote {OUT_DIR / 'cube_voronoi_shell_only.stl'}")
