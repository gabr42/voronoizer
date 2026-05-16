# Phase 2 — Geodesic Voronoi: Implementation Plan

## Goal

Replace the tangent-plane Voronoi tessellation (Phase 1) with a **geodesic
Voronoi on the actual mesh surface**. Cell boundaries become real curves
that follow the surface; they're gap-free by construction and don't
suffer from polygon vertices projecting into neighbours' territory.

This is the proper structural fix for the cases Phase 1 handles only
partially:

- CAD-like models with smooth filleted edges (the `--soft-edge-angle 10`
  workaround case).
- Very low-poly input meshes.
- Sparse seeds on highly curved surfaces.

When Phase 2 lands, `--soft-edge-angle` becomes irrelevant (the
boundaries respect the actual surface, no matter how soft the edges).

## What stays from Phase 1

- `shell.py` — voxel hollowing is unchanged.
- `seeding.py` — Poisson-disk seed sampling is unchanged (we just don't
  need the sharp-edge rejection anymore — geodesic Voronoi handles edges
  naturally).
- `meshio.py`, `progress.py` — unchanged.
- `perforate.py` — boolean subtraction is unchanged.
- The **per-vertex prism builder** from Phase 1 (`_build_prism_surface_aware`)
  — we reuse it, fed by a different ring polyline.
- The **lift trick** for chamfered rings — keeps manifold3d clean. We
  proved this works even with arbitrarily many polygon vertices.
- The **cap centroid placement** at `seed + max(safety, R_local)·normal` —
  still needed (still produces cap-wheel triangles that must stay
  outside the parallel sphere).

## What changes / goes away

| | Phase 1 (current) | Phase 2 |
|---|---|---|
| Cell polygon | half-plane intersection in seed's tangent frame | mesh-edge loop where adjacent triangles have different seed labels |
| Cell boundary | straight line segments in 2D, projected to surface | already a 3D polyline on the surface |
| Sharp-edge twins | `mirror_seeds.py` generates twin seeds for each sharp edge | not needed — geodesic distance already respects edges |
| Boundary refinement | quadratic Bézier in 2D tangent frame | quadratic Bézier on the surface (control points re-projected after each step) |
| Strut inset | shrink 2D polygon by `strut/2`, then project | shift each surface vertex inward by `strut/2` perpendicular to the local boundary tangent, then re-project to surface |
| Per-vertex frame source | nearest-surface-point lookup for each tangent vertex | natural — each boundary vertex already has a surface position and a face normal |

The `mirror_seeds.py` module and the `--soft-edge-angle` CLI flag both
become dead code. Keep them around behind a `--legacy-tangent-engine`
fallback for one release, then remove.

## Algorithm

### Stage 1 — Mesh preparation

The input STL may be too coarse for Dijkstra to produce smooth Voronoi
boundaries. Subdivide so the boundaries have enough mesh edges to follow.

- **Target edge length**: `strut_thickness / 4` is a good rule of thumb.
  At `strut=2 mm` that means ~0.5 mm edges, ample resolution for the
  boundary to look smooth after a single Bézier pass.
- **Subdivision method**: trimesh's `mesh.subdivide_to_size(target_edge_length)`
  recursively splits long edges. O(F) for typical meshes.
- **Cost ceiling**: cap the subdivided face count at maybe 500k. Warn
  if the input would need more (typical 3D-print STL won't).

Add a CLI flag `--target-edge-length MM`, default `strut_thickness / 4`,
to let users override.

### Stage 2 — Multi-source Dijkstra

For each input seed, run Dijkstra on the subdivided mesh's edge graph,
all sources at once.

- Initial frontier: each seed gets added as a "virtual" mesh vertex
  connected to the three vertices of the face it landed on, with edge
  weights = distance from seed to each of those face vertices.
- Edge weights for mesh edges: Euclidean edge length.
- Output: per-mesh-vertex, the nearest seed id and the geodesic distance.

This is approximate (Dijkstra on the edge graph is an upper bound on
true geodesic distance) but it's accurate enough for boundary
construction at our resolution. `O(V log V + E)`. Sub-second for 100k
vertex meshes.

If quality turns out to be insufficient on a real model, upgrade path:
the **heat method** (`potpourri3d.MeshHeatMethodDistanceSolver`) gives
smooth geodesic distances. Drop-in replacement.

### Stage 3 — Boundary edge extraction

For every mesh edge whose two adjacent triangles have different seed
labels, that edge is part of the Voronoi boundary.

- Build a `(seed_a, seed_b) -> [edges]` map.
- For each cell `s`, collect the edges where `s` appears as one of
  `(seed_a, seed_b)`.
- Walk the edges to form closed loops:
  - Build a per-cell vertex-to-edges adjacency table.
  - Start from any boundary vertex; follow edges keeping the cell on
    the left until back to start.
  - Repeat for unvisited starts (cells with multiple disconnected
    boundary components — can happen on torus-like topology but not
    for our box/sphere/cylinder inputs).

Each cell ends up with one or more closed 3D polylines on the surface.

### Stage 4 — Boundary smoothing

The raw polyline follows mesh edges and looks pixelated. Smooth it
while keeping vertices on the surface.

- **One Bézier pass**, mirroring Phase 1's `_bezier_smooth`: for each
  edge `V[i] → V[i+1]`, anchor a quadratic Bézier at edge midpoints,
  use `V[i]` as the control point. Sample `_BEZIER_SAMPLES_PER_EDGE`
  points along the curve.
- After Bézier, each new vertex is in 3D space *near* the surface.
  Re-project each onto the closest mesh face (trimesh
  `ProximityQuery.on_surface`). Negligible drift.

Reuse `_BEZIER_SAMPLES_PER_EDGE = 6` from Phase 1 unless empirically
problematic.

### Stage 5 — Surface inset (strut/2)

For each boundary vertex `V_i`:

1. Local boundary tangent `t_i = normalize(V[i+1] − V[i−1])`.
2. Local surface normal `n_i` from the underlying mesh face.
3. Inward direction in the surface tangent plane: `d_i = normalize(n_i × t_i)`,
   sign-flipped so it points away from the *outside* of the cell.
   ("Outside the cell" = the seed of the *other* side of this boundary
   edge, easy to track from Stage 3.)
4. Move `V_i` by `d_i · strut/2`.
5. Re-project onto the mesh surface.

Result: an inset 3D polyline that's `≈ strut/2` (geodesic) inward of
the cell boundary, still on the surface. Two cells sharing a boundary
both inset away from it, giving a `strut` gap between their inset
polylines.

### Stage 6 — Per-vertex prism construction

Feed the inset polyline directly into Phase 1's
`_build_prism_surface_aware`. For each boundary vertex `i`:

- `P_i` = surface position (already on the surface — no projection
  step needed).
- `n_i` = surface normal at that position (face normal of the face
  `P_i` sits on, or barycentric-weighted vertex normals for smoother
  results).
- `d_out_i` = inward-pointing direction (the same `d_i` from Stage 5,
  but negated — chamfer expansion points *outward* from the cell, so
  back toward the original (un-inset) boundary).

Chamfer rings, lift trick, cap centroids — all already in place from
Phase 1.

### Stage 7 — Boolean subtract from shell

Unchanged from Phase 1. `perforate.perforate(shell, cells, batch_size=25)`.

## Module structure

- New: `voronoizer/surface_voronoi.py`
  - `assign_cell_labels(mesh, seeds) -> labels` (multi-source Dijkstra).
  - `boundary_edges(mesh, labels) -> {(s_a, s_b): [(v_a, v_b), ...]}`.
- New: `voronoizer/surface_boundary.py`
  - `extract_cell_loops(mesh, boundary_edges, cell_id) -> [Loop]`
    where `Loop` is a list of (vertex_idx, position, in_face_idx).
  - `bezier_smooth_on_surface(loop, mesh, samples_per_edge) -> Loop`.
  - `inset_loop(loop, mesh, strut_half, outside_cell_lookup) -> Loop`.
- New: `voronoizer/surface_prism.py`
  - `build_prism_from_loop(loop, seed, seed_normal, R_local, shell, chamfer, safety) -> Trimesh`.
  - Internally calls Phase 1's `_build_prism_surface_aware` (or a
    pared-down twin).
- Modified: `voronoizer/pipeline.py`
  - New "subdivide input mesh" step.
  - Replace cell-build step with the four new stages above.
- Modified: `voronoizer/cli.py`
  - Add `--target-edge-length MM` (default `strut_thickness / 4`).
  - Add `--engine {geodesic,tangent}` (default `geodesic`). `tangent`
    keeps Phase 1 for regression comparison.
  - Mark `--soft-edge-angle` as deprecated for geodesic engine
    (still functional under `--engine tangent`).
- Unchanged: `shell.py`, `meshio.py`, `perforate.py`, `progress.py`,
  `seeding.py`, the entire `voronoi_cells.py` module (kept under the
  `tangent` engine).
- Removed by default (still callable under `tangent`):
  `mirror_seeds.py`. Geodesic Voronoi handles sharp edges naturally.

## CLI changes

```
voronoizer INPUT.stl OUTPUT.stl [options]

  ... existing options ...

  --engine {geodesic,tangent}    Voronoi tessellation engine.
                                 geodesic (default): cell boundaries
                                 follow the actual mesh surface;
                                 robust on CAD models with fillets and
                                 on low-poly inputs.
                                 tangent: Phase 1's tangent-plane
                                 Voronoi; kept for regression and
                                 special cases.
  --target-edge-length MM        For --engine geodesic: subdivide the
                                 input mesh until all edges are at
                                 most this long. Default: strut/4.
```

`--soft-edge-angle` is silently ignored when `--engine geodesic`.

## Testing

Run all six standard cases under both engines and confirm
geodesic is at least as good:

| | tangent (current) | geodesic (target) |
|---|---|---|
| cube ch=0 | wt=True nm=0 | wt=True nm=0 |
| cube ch=0.6 | wt=True nm=0 | wt=True nm=0 |
| cylinder top-bot ch=0 | wt=True nm=0 | wt=True nm=0 |
| cylinder top-bot ch=0.5 | wt=True nm=0 | wt=True nm=0 |
| sphere ch=0 | wt=True (after lift) | wt=True nm=0 |
| sphere ch=0.5 | wt=True nm=0 | wt=True nm=0 |

Plus two cases Phase 1 only partially handles:

| | tangent | geodesic |
|---|---|---|
| body LR (168 faces) full perforation | 33 nm + visible artefacts | wt=True nm=0 expected |
| body HR full perforation | 2 nm + visible artefacts near fillets | wt=True nm=0 expected |

Add a regression test script in `tests/` that runs all the above and
checks the result against expected (vol, watertight, nm count).

## Open questions / risks

1. **Boundary smoothing quality**. A single Bézier pass after Dijkstra
   may not be smooth enough — the underlying loop is mesh-edge-jagged,
   and Bézier control points sit on the jagged path. Mitigations: a
   second smoothing pass; Laplacian smoothing on the polyline; or
   resample to constant arc length before the Bézier.
2. **Strut consistency on highly curved surfaces**. Insetting by
   `strut/2` is in geodesic distance. The Euclidean distance between
   two cells' inset polylines may differ slightly on tight curves —
   probably acceptable.
3. **Loop extraction edge cases**. Cells touching the mesh boundary
   produce open polylines (only matters for open-mesh inputs — our
   shells are always closed, so this shouldn't fire, but the code
   should handle it gracefully).
4. **Subdivision blow-up**. A pathological input could explode the
   subdivided mesh size. Add a hard cap (e.g. 500k faces) and a clear
   error message.
5. **Performance**. Multi-source Dijkstra on 100k vertices: well under
   a second. Boundary extraction: O(E). Boolean stays the bottleneck.
6. **Cells with very small angular footprint** (e.g. forced by a
   dense `--holes` value on a tiny model). Boundary loops may have
   too few vertices to smooth meaningfully. Could fall back to a
   minimum number of Bézier samples per loop (e.g. ensure ≥ 12
   vertices per cell after smoothing, even if the raw loop had fewer
   mesh edges).
7. **Mesh quality assumption**. Geodesic Voronoi assumes the input
   mesh is 2-manifold and consistently oriented. We already require
   this for the boolean. `--repair` is the user's escape hatch.

## Scope and effort

Realistic estimate: **1–2 weeks** of focused work.

| Stage | Effort | Risk |
|---|---|---|
| Mesh subdivision | half day | low (trimesh has it) |
| Multi-source Dijkstra | 1 day | low |
| Boundary edge extraction | 1 day | low |
| Loop walking | 1–2 days | medium (edge cases) |
| Surface Bézier smoothing | 1–2 days | medium (off-surface drift) |
| Surface inset | 1–2 days | medium (sign conventions, drift) |
| Prism wiring (reuse Phase 1) | half day | low |
| CLI / pipeline integration | half day | low |
| Test sweep + regression | 1 day | low |
| Polish / iteration | 2–4 days | depends on the above |

Most of the time goes into the loop-walking and surface-inset details
— Phase 1 taught us those geometric subtleties always take longer than
expected. The Phase 1 prism builder, lift trick, and boolean pipeline
are reusable as-is.

## Recommendation

Start Phase 2 only when there's a real model that Phase 1 +
`--soft-edge-angle` can't satisfy. The CAD body case is the canonical
trigger — if you find more models in the same family that need it,
that's the cue.

When you do start, do it in a long-lived branch and ship under
`--engine geodesic` (off by default) for one release cycle so users
can compare. Default-flip once it's verified on a handful of real
prints.
