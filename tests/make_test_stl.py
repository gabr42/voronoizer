"""Generate small reference STLs for smoke-testing voronoizer."""

from pathlib import Path

import trimesh


def main() -> None:
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    cube = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
    cube.export(out_dir / "cube_100mm.stl")

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
    sphere.export(out_dir / "sphere_r50mm.stl")

    cyl = trimesh.creation.cylinder(radius=30.0, height=120.0, sections=64)
    cyl.export(out_dir / "cyl_r30_h120mm.stl")

    print(f"wrote test STLs to {out_dir}")


if __name__ == "__main__":
    main()
