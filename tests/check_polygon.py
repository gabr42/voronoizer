import numpy as np
from voronoizer.meshio import load_stl
from voronoizer.seeding import sample_seeds
from voronoizer.voronoi_cells import build_shrunken_cells
import voronoizer.voronoi_cells as vc

polys = []
orig = vc._build_prism
def spy(polygon_2d, *a, **kw):
    polys.append(polygon_2d.copy())
    return orig(polygon_2d, *a, **kw)
vc._build_prism = spy

m = load_stl(r"H:\RAZVOJ\voronoizer\tests\data\sphere_r20mm.stl")
rng = np.random.default_rng(3)
seeds = sample_seeds(m, 30, top_bottom_only=False, angle_deg=30.0,
                    rng=rng, strut_thickness=1.5)
cells, _ = build_shrunken_cells(seeds, m, 1.0, 1.5)

for i in (0, 5, 10, 15, 20):
    p = polys[i]
    diag = np.linalg.norm(p.max(axis=0) - p.min(axis=0))
    print(f"poly {i}: verts={len(p)} bbox-diag={diag:.2f} max-radius={np.linalg.norm(p, axis=1).max():.2f}")
print(f"avg max radius: {np.mean([np.linalg.norm(p, axis=1).max() for p in polys]):.2f}")
print(f"avg cell vol:   {np.mean([c.volume for c in cells]):.1f}")
