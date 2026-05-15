# voronoizer

Command-line tool that turns an STL file into a Voronoi-perforated shell — a
hollow wall with organic, smooth-edged holes whose layout follows a 3D
Voronoi tessellation of points sampled evenly across the input surface.

```
voronoizer model.stl perforated.stl -t 2 -n 120 -s 1.5
```

What it does, in order:

1. Loads the STL.
2. Hollows the body into a shell of the requested wall thickness (voxelize →
   erode inward → marching cubes → boolean-subtract the inner volume).
3. Samples N seed points evenly across the surface (Poisson-disk sampling,
   with rejection near sharp dihedral edges and open boundaries).
4. Builds a 3D Voronoi tessellation around the seeds and turns each cell
   into a smooth-edged prism cutter, smoothed with quadratic Bézier curves
   so hole boundaries look organic rather than polygonal.
5. Boolean-subtracts the cutters from the shell to produce the perforated
   output STL.

Sharp edges and open boundaries are handled with mirror / twin seeds so that
holes wrap cleanly around cube edges and cylinder cap rims without leaving
flat patches.

Typical input range: 30–300 mm in any dimension.

## Installation

Requires Python 3.10 or newer. Use a project-local virtual environment:

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

bash (Linux / macOS):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The editable install (`-e .`) builds the entry-point script and links the
package to the source tree so code changes take effect without re-installing.

After installing you can invoke the tool either as `voronoizer ...` or
`python -m voronoizer ...`.

### Dependencies

Pulled in automatically by `pip install -e .`:

- `trimesh` + `manifold3d` — mesh I/O and boolean operations
- `numpy`, `scipy` — Voronoi tessellation, KD-trees, half-space intersection
- `scikit-image` — marching cubes for the shell construction
- `networkx`, `tqdm` — graph utilities and progress bars

## Usage

```
voronoizer INPUT.stl OUTPUT.stl [options]
```

`OUTPUT.stl` may contain the macro `{seed}`, which is replaced with the
random seed actually used. Combined with `--seed N` (or letting the tool
pick one) this lets you keep the seed visible in the filename — for
example `voronoizer in.stl "out_{seed}.stl"` produces `out_1734528.stl`
and re-running with `--seed 1734528` reproduces the same pattern.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `-t, --shell-thickness MM` | 2.0 | Wall thickness of the hollow shell. |
| `-n, --holes INT` | 100 | Approximate number of holes to create. |
| `-s, --strut-thickness MM` | 1.5 | Width of the strut (webbing) left between adjacent holes. |
| `--top-bottom-only` | off | Perforate only roughly horizontal faces — useful for cylinders, vases, etc. where you want the side walls intact. |
| `--normal-angle DEG` | 30 | With `--top-bottom-only`, the maximum angle from ±Z a face can have and still count as "top/bottom". |
| `--edge-margin MM` | auto | Minimum distance seeds must keep from sharp mesh edges or open boundaries. Default is auto-scaled from seed density. |
| `--soft-edge-angle DEG` | 25 | Dihedral angle above which a mesh edge counts as "sharp" for the seed-rejection logic. The default works for boxy / faceted models with crisp edges. Lower it (e.g. 10) for CAD-like models with smooth filleted edges where holes near the fillets would otherwise look chewed up. |
| `--chamfer MM` | 0.0 | Bevel the hole edges where they meet the shell surfaces. Works well on flat faces (cube, cylinder caps); on highly curved surfaces (spheres) the bevel is only visible near the centre of each hole — see *Limitations* below. Clamped to `min(0.49·strut, 0.49·shell)`. |
| `--seed N` | random | Random seed for reproducible hole layouts. The seed actually used is always printed to stderr. |
| `--repair` | off | Best-effort repair of non-manifold or non-watertight input meshes. |
| `-v, --verbose` | off | Progress bars and per-step timing. |
| `--version` | — | Print version and exit. |

### Examples

A standard perforated cube:

```
voronoizer cube_100mm.stl cube_voronoi.stl -t 2 -n 80
```

A vase-like cylinder with holes only on top and bottom caps:

```
voronoizer cylinder.stl cylinder_holes.stl -t 2 -n 40 --top-bottom-only
```

A reproducible run that records its seed in the filename:

```
voronoizer model.stl "model_{seed}.stl" -t 2 -n 100 -s 1.5
```

A chamfered cube — note the `{seed}` macro for re-running with the same
pattern later:

```
voronoizer cube_100mm.stl "cube_chamfer_{seed}.stl" -t 2 -n 60 -s 2 --chamfer 0.5
```

## Limitations

- **Chamfer on highly curved surfaces.** The chamfer bevel is built in
  each seed's tangent frame; on a strongly curved wall (e.g. a sphere) the
  bevel is most visible at the centre of each hole and fades out toward
  the hole boundary. On flat faces (cubes, cylinder caps) the chamfer is
  uniform.
- **Very thin shells or thin features.** If `--shell-thickness` exceeds
  the thinnest cross-section of the input, the shell construction can
  produce empty or degenerate results. Keep shell thickness comfortably
  smaller than the model's thinnest feature.
- **Non-manifold input.** `--repair` is best-effort. Pathologically broken
  STLs may need pre-processing in a dedicated repair tool (e.g. MeshLab,
  Blender) before voronoization.

## Authors

- Primož Gabrijelčič — <gabr42@gmail.com>
- Claude Opus 4.7 (Anthropic)

## License

MIT.
