"""Build cells with and without taper, run full perforation, compare."""
import numpy as np
import trimesh
from voronoizer.meshio import load_stl
from voronoizer.shell import build_shell
from voronoizer.seeding import sample_seeds
from voronoizer.voronoi_cells import build_shrunken_cells
from voronoizer.perforate import perforate
import voronoizer.voronoi_cells as vc


def run(use_taper):
    m = load_stl(r"H:\RAZVOJ\voronoizer\tests\data\sphere_r20mm.stl")
    shell = build_shell(m, 1.0)
    rng = np.random.default_rng(3)
    seeds = sample_seeds(m, 30, top_bottom_only=False, angle_deg=30.0,
                        rng=rng, strut_thickness=1.5)

    if use_taper:
        cells, _ = build_shrunken_cells(seeds, m.bounds, 1.0, 1.5)
    else:
        # Override _build_prism with one that ignores R_local
        orig = vc._build_prism
        def flat(*args, **kwargs):
            kwargs["R_local"] = float("inf")
            return orig(*args, **kwargs)
        vc._build_prism = flat
        try:
            cells, _ = build_shrunken_cells(seeds, m.bounds, 1.0, 1.5)
        finally:
            vc._build_prism = orig

    cells_vol = sum(c.volume for c in cells)
    out = perforate(shell, cells)
    print(f"{'taper' if use_taper else 'flat ':<5}: "
          f"shell_vol={shell.volume:.1f} cells_total_vol={cells_vol:.1f} "
          f"out_vol={out.volume:.1f} wt={out.is_watertight}")


run(False)
run(True)
