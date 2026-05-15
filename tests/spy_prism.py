import numpy as np
from voronoizer.meshio import load_stl
from voronoizer.seeding import sample_seeds
from voronoizer.voronoi_cells import build_shrunken_cells
import voronoizer.voronoi_cells as vc

calls = []
orig = vc._build_prism

def spy(*args, **kwargs):
    calls.append(kwargs.get("R_local"))
    return orig(*args, **kwargs)

vc._build_prism = spy

m = load_stl(r"H:\RAZVOJ\voronoizer\tests\data\sphere_r20mm.stl")
rng = np.random.default_rng(3)
seeds = sample_seeds(m, 30, top_bottom_only=False, angle_deg=30.0,
                    rng=rng, strut_thickness=1.5)
cells, _ = build_shrunken_cells(seeds, m, 1.0, 1.5)
print(f"_build_prism called {len(calls)} times")
shown = [f"{r:.2f}" if isinstance(r, float) and np.isfinite(r) else "inf"
        for r in calls[:8]]
print("first R_local values:", shown)
fin = sum(1 for r in calls if isinstance(r, float) and np.isfinite(r))
inf = sum(1 for r in calls if isinstance(r, float) and not np.isfinite(r))
print(f"finite: {fin}, inf: {inf}")
