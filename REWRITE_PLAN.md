# Surface-Aware Voronoi Perforation — Rewrite Plan

## Goal

Make chamfer (and hole shape in general) follow the **actual surface** at every
point of every hole boundary, so the bevel is uniform on curved surfaces
(spheres, organic shapes) instead of fading out toward the hole edges.

## Why the current approach fails

Every cell's geometry is built in **one** tangent frame anchored at the seed:

- The Voronoi polygon comes from intersecting bisector half-planes in that
  single tangent plane.
- The prism (cutter) is extruded along that single seed normal.

On a curved surface, polygon vertices far from the seed correspond to surface
points whose true position and true normal are very different from the
tangent-plane assumption. Result: the chamfer is built in the wrong place at
the hole edges.

## Phase 1 — Surface-aware prism (incremental)

**Keep** the existing tangent-plane Voronoi polygon. **Change** the prism
construction so each polygon-vertex column uses *its own* local surface frame.

For each polygon vertex `(u_i, v_i)` in seed S's tangent frame:

1. Form the 3D ray `seed + u_i·u + v_i·v` (project the tangent vertex onto
   seed's tangent plane in 3D).
2. Find the **nearest point on the mesh surface** to that 3D point
   (trimesh ProximityQuery).
3. Take the mesh face normal at that surface point → `n_i`.
4. Store the projected surface point `P_i` and normal `n_i` for this column.

Then build the chamfered prism column at vertex i using `P_i` as the surface
anchor and `n_i` as the local normal:

- Ring 1 (outer surface) at `P_i + chamfer · d_i` where `d_i` is the local
  in-surface outward direction.
- Ring 2 (chamfer end) at `P_i − chamfer · n_i`, no lateral offset.
- Ring 4 (inner surface) at `P_i − shell · n_i + chamfer · d_i`.
- Cap rings above ring 1 / below ring 4 by a small safety margin along each
  column's own `n_i`.

**Side walls connect ring r vertex i to ring r vertex (i+1)** — these are
twisted quads in 3D (each pair of columns has slightly different normals), but
the topology stays clean.

**Fallback for projections that land on the wrong face.** When a polygon
vertex projects onto a face whose normal disagrees sharply with the seed
normal (`n_i · seed_normal < 0.5`, ~60° cone), fall back to the seed frame for
that vertex: use the 3D tangent point as `P_i`, `seed_normal` as `n_i`, and
the seed-tangent radial direction as `d_i`. This avoids artefacts when a cell
spans a sharp dihedral edge.

**Expected fix**: chamfer is visible uniformly around each hole on moderately
curved surfaces. The bevel "follows" the surface at each vertex.

**Expected remaining issues**: cells with elongated tangent polygons (one
spike to large `d_i`) still have one vertex on the far side of the sphere
where the projected surface point is in another cell's territory — these will
produce weird artifacts at that single direction. Phase 1 fixes the typical
case, not the pathological case.

**Files touched**: `voronoi_cells.py` (`_build_prism` and
`build_shrunken_cells`), `pipeline.py` (pass mesh into `build_shrunken_cells`),
test scripts that call `build_shrunken_cells` with the old signature.
Single-day change, low risk. The non-chamfer code path is unaffected.

## Phase 2 — Geodesic surface Voronoi (full rewrite)

Replace tangent-plane Voronoi entirely with **discrete geodesic Voronoi on the
mesh**. This is the right fix for sparse seeds on highly curved surfaces.

### Pipeline changes

```
load STL → build shell → sample seeds
        ↓
NEW: discrete geodesic Voronoi
  - multi-source Dijkstra on the mesh's edge graph: each mesh vertex
    gets a nearest-seed label and a geodesic distance
  - extract cell-boundary edge loops (mesh edges where the two adjacent
    triangles belong to different cells)
        ↓
NEW: boundary refinement
  - resample / smooth the raw mesh-edge polyline so the boundary doesn't
    look pixelated (snap to geodesic line between cell corners, or short-
    arc Bézier on the surface)
  - re-project to the actual surface after smoothing
        ↓
NEW: surface inset
  - for each boundary vertex, take the in-surface inward direction
    (perpendicular to the boundary, in the local tangent plane)
  - shift by strut/2; re-project to surface
        ↓
NEW: per-vertex prism (same idea as Phase 1, but driven by the geodesic
      boundary rather than the tangent polygon)
        ↓
boolean subtract from shell  (unchanged)
```

### Key technical decisions

- **Geodesic distance**: start with **edge-graph Dijkstra** from all seeds
  (multi-source). It's `O(V log V + E)` and works on any mesh; quality is
  mesh-resolution-dependent but adequate for typical 3D-print STLs
  (10k–100k triangles). If quality is poor, upgrade to the **heat method**
  (`potpourri3d`) or **exact geodesics** (vendored from libigl).
- **Mesh resolution**: low-poly inputs need pre-subdivision so Dijkstra has
  enough vertices for clean boundaries. Add an optional
  `--subdivide-target-edge-length` flag, default = `strut_thickness / 2`.
- **Loop extraction**: walk the dual graph of "boundary edges" to extract one
  closed loop per cell. Watch for cells that touch a mesh boundary (open
  mesh) — those produce open polylines.
- **Surface inset**: at each boundary vertex, compute the in-surface inward
  direction as `bitangent × normal`, where the bitangent is the local
  boundary tangent. Project the offset point back to the surface via
  closest-point query (occasional small drift is fine).
- **Strut consistency**: insetting by `strut/2` from each side of a shared
  boundary gives strut `≈ strut` in geodesic distance, but the Euclidean
  distance between adjacent struts may differ slightly on highly curved
  areas. Probably acceptable; can compensate later if needed.

### Files

- New: `surface_voronoi.py` (multi-source Dijkstra, cell labels, boundary
  edges).
- New: `surface_boundary.py` (loop extraction, smoothing, surface inset).
- New: `surface_prism.py` (per-vertex prism construction shared with Phase 1).
- Rewrite: `pipeline.py` (calls the new modules).
- Keep: `voronoi_cells.py` available behind a flag (`--legacy-voronoi`) for
  regression comparisons during development.
- Unchanged: `shell.py`, `seeding.py`, `meshio.py`, `perforate.py`. Mirror/twin
  seeds (`mirror_seeds.py`) may be dropped — the geodesic Voronoi handles
  sharp edges naturally if seeds are sampled correctly.

### Scope

Realistically 1–2 weeks of focused work. Most of the time goes into:

- Boundary loop extraction edge cases (cells touching mesh boundaries, very
  small cells, non-2-manifold spots).
- Smoothing without introducing self-intersections.
- Tuning the per-vertex prism so booleans stay watertight when adjacent cells
  have wildly different local normals.

## Risks and unknowns

1. **Boolean robustness with twisted side walls**. Manifold3d is generally
   robust, but per-vertex-normal prisms will have non-planar side quads. May
   need to split each quad along its shorter diagonal to keep faces sane.
2. **Mesh density vs. cell size**. If `holes` is large relative to mesh
   triangle count, individual cells span only a few triangles and the
   boundary is unusable without subdivision.
3. **Performance**. Multi-source Dijkstra on a 100k-vertex mesh runs in well
   under a second. The bottleneck will still be the boolean subtraction step
   (unchanged).
4. **Phase 1 may be enough.** Worth trying before committing to Phase 2 — if
   Phase 1 fixes the chamfer for typical inputs (cube, sphere with reasonable
   seed density, organic shapes), Phase 2 is a 5× effort for marginal extra
   polish.

## Recommendation

**Do Phase 1 first.** It's a 1-day, low-risk, surgical change to
`voronoi_cells.py` that should resolve the chamfer-on-sphere issue for any
seed density where polygon vertices stay within ~half the sphere's radius.
Only commit to Phase 2 if Phase 1's residual artifacts (the elongated-polygon
pathological case) actually bother in practice.

## Phase 1 outcome

- ✅ **Flat surfaces** (cube, cylinder caps): chamfer is bit-identical to the
  pre-rewrite simple-chamfer path. No regression.
- ✅ **In-memory correctness**: the boolean result is watertight on the
  sphere too. The Phase 1 prism puts each chamfer ring on the *actual*
  surface, so the bevel is uniform around every hole.
- ✅ **No more shell islands inside holes**: an earlier draft anchored the
  cap centroid at `mean(cap_top_ring)`, which on a sphere with wide-angular
  cells fell *inside* the sphere (average of points on a sphere is
  interior). The cap wheel then formed a cone tip pointing inward and the
  boolean left small islands of shell material inside each hole. Fixed by
  anchoring the cap centroid at `seed ± k·seed_normal`, which is
  guaranteed to lie outside the shell along the seed's outward direction.
- ⚠️ **STL / PLY / OBJ round-trip**: on highly curved surfaces with sparse
  seeds, the boolean output contains ~250 *twin vertices* (two distinct
  topological vertices at the same 3D location), produced by manifold3d
  where adjacent Voronoi cells are very nearly tangent on the curved
  surface. Standard mesh loaders dedupe these twins on load, which breaks
  the manifold — the loaded file reports ~500 non-manifold edges.

  - Edge-perpendicular polygon offset (instead of centroid-radial) helped
    slightly (487 vs 509 nm edges).
  - Reducing the chamfer size did **not** help — the issue isn't chamfer
    magnitude, it's that surface-aware prisms with diverging per-vertex
    normals produce sliver-prone boolean output regardless.
  - Some slicers may still handle the twin-vertex geometry gracefully; this
    needs in-print validation.

**The twin-vertex issue is a Phase 2 problem to properly fix.** A geodesic
Voronoi tessellation produces cell boundaries that are gap-free on the
surface by construction, so the boolean won't need to manufacture twin
vertices at near-tangencies.
