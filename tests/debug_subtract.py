"""Subtract cells one at a time and report which ones change the result."""

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

mesh = load_stl(INPUT)
shell = build_shell(mesh, THICKNESS)
rng = np.random.default_rng(SEED)
seeds = sample_seeds(mesh, HOLES, top_bottom_only=False, angle_deg=30.0,
                     rng=rng, strut_thickness=STRUT)
cells, _ = build_shrunken_cells(
    seeds=seeds,
    mesh=mesh,
    shell_thickness=THICKNESS,
    strut_thickness=STRUT,
)
print(f"shell volume start: {shell.volume:.2f}")

cur = shell
removed = []
for i, cell in enumerate(cells):
    new = trimesh.boolean.difference([cur, cell], engine="manifold")
    if not isinstance(new, trimesh.Trimesh) or len(new.faces) == 0:
        print(f"  cell {i}: SUBTRACT PRODUCED EMPTY MESH (skipped)")
        continue
    dv = cur.volume - new.volume
    removed.append((i, dv, cell.volume, seeds.points[i], seeds.normals[i]))
    cur = new

removed.sort(key=lambda r: r[1])
print(f"{'idx':>3}  {'dV_mm3':>10}  {'cell_vol':>10}  {'seed_xyz':>22}  normal")
for r in removed[:15]:
    print(f"{r[0]:>3}  {r[1]:>10.2f}  {r[2]:>10.2f}  ({r[3][0]:6.2f},{r[3][1]:6.2f},{r[3][2]:6.2f})  {r[4]}")
print("...")
for r in removed[-5:]:
    print(f"{r[0]:>3}  {r[1]:>10.2f}  {r[2]:>10.2f}  ({r[3][0]:6.2f},{r[3][1]:6.2f},{r[3][2]:6.2f})  {r[4]}")
print(f"\nfinal volume: {cur.volume:.2f}")
cur.export(Path(r"H:\RAZVOJ\voronoizer\tests\data\cube_voronoi_seq.stl"))
