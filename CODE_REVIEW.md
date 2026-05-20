# Code Review Checklist

Findings from the 2026-05-20 code review. Status legend:

- [ ] not yet verified
- [x] verified + fixed
- [~] verified, decision: not a bug / won't fix (with note)
- [L] deferred to "do later" list

## Real bugs

- [x] **1. `--chamfer` clamping is documented but never executed.** `_clamp_chamfer_value` exists but is never called. CLI help and README promise clamping to `min(0.49·strut, 0.49·shell)`.
- [x] **2. `shell.py` ignores `--soft-edge-angle`.** Hardcodes 25° in two `face_components` calls instead of taking the user's value.
- [x] **3. Stale CLI flag in error message.** `surface_voronoi.py:96` says "use --engine tangent" — that flag was removed.
- [x] **4. `progress.step` label lies.** `surface_pipeline.py:213` says "Dijkstra" but the implementation is per-patch Euclidean cKDTree.

## Stale / dead code

- [x] **5. Dead helpers in `surface_voronoi.py`.** Removed `_SHARP_EDGE_COST_MULT`, `_sharp_edge_keys`, `_build_component_adjacency`, `_vertices_in_component`, `_faces_in_component` (inlined), `face_labels_from_vertex_labels` shim, `GeodesicLabels` dataclass, `face_distances` computation, leftover `heapq` / `dataclass` imports, and the redundant `face_components` wrapper.
- [x] **6. Dead imports in `surface_pipeline.py`.** `_BEZIER_SAMPLES_PER_EDGE`, `_bezier_smooth`, `_inset_polygon_2d` from `voronoi_cells`; `_smoothed_normals_at` from `surface_boundary`.
- [x] **7. Duplicated constant `_BEZIER_SAMPLES_PER_EDGE = 6`** — removed from `voronoi_cells.py` (it had no live users there). The constant now lives in only `surface_boundary.py`. Also removed dead `_bezier_smooth` and `_inset_polygon_2d` helpers from voronoi_cells.py.
- [x] **8. Stale docstrings.** Module-level docstrings, comments, and progress labels referring to "Dijkstra", "Phase 1/2", "tangent engine", `build_shrunken_cells` rewritten to describe what the code actually does.
- [~] **9. Repo clutter.** Top-level STLs, `_phase2_outputs/`, `REWRITE_PLAN.md`, `PHASE_2_PLAN.md`. User decision: leave alone.

## Code smells / correctness concerns (minor)

- [x] **10. Private cross-module import via `# type: ignore`.** Promoted `_top_bottom_face_mask` to public `top_bottom_face_mask`; dropped the `# type: ignore`.
- [x] **11. Seeding overshoot when `count < n_eligible_patches`.** When `--holes < n_eligible_patches`, now allocates one seed to each of the `count` largest patches and warns the user; smaller patches go unseeded and fall through to the global-nearest-seed fallback in `assign_cell_labels`. Verified: cube with `--holes 3` produces exactly 3 seeds (was 6 before).
- [x] **12. `_polygon_clip_and_inset` interior point may end up outside.** Now tries the seed (0,0) first (guaranteed inside the original Voronoi cell), then falls back to the polygon centroid.
- [x] **13. `dedupe_loop` is defined but never called.** Deleted. Regression suite passed without it; if its concern resurfaces in practice, easier to re-add deliberately than to keep dead code.
- [x] **14. Double-construction of `ProximityQuery`** — collapsed into a single instance reused for seed-to-patch lookup and per-cell snapping.
- [x] **15. Name collision: function `face_components` vs parameter `face_components`** in `surface_voronoi.py`. Parameter renamed to `face_comp` everywhere. Also dropped the no-op `face_components` wrapper around `_face_components` (renamed `_face_components` → `face_components` directly).
- [x] **16. Magic constants in pre-filter.** Promoted 10° / 0.5 thresholds to `_FEATURE_PATCH_MIN_DIHEDRAL_DEG` and `_LOOP_FACE_ALIGN_COS_MIN` with explanatory comments.
- [x] **17. Redundant `np.where(keep)[0]` in dedupe_loop close-loop block.** Moot — function deleted in #13.
- [~] **18. `cli.py:155` seed entropy.** Not a bug. Standard numpy behavior; the `default_rng()` no-arg form is documented as OS-entropy-seeded. No comment needed.
- [x] **19. Test fixtures don't auto-generate.** `regression.py` now runs `make_test_stl.py` automatically when any of the three generated fixtures (cube/sphere/cylinder) is missing.
- [x] **20. `seeding.py` rng-derived seed `+1` can edge-case to INT_MAX.** Replaced `seed_int + 1` with a fresh `rng.integers(0, 2**31 - 1)` draw — no arithmetic, no boundary risk.
- [~] **21. `meshio.load_stl` silently merges multi-body STLs** via `force="mesh"`. Not really a bug — the load log already reports vertex/face counts and watertight status; `--repair` exists for any cleanup. Adding an explicit "merged N bodies" warning would require pre-loading as a Scene to count, doubling I/O cost for a corner case.

## Documentation / spec mismatches

- [~] **22. README "width of strut" wording.** Not actually wrong. Two adjacent cells each inset by `strut/2`, so the gap between them IS `strut`. README description is correct.
- [~] **23. README on `--target-edge-length`.** The `--cutters` description already says "Build the shell and compute the Voronoi cell cutters" — implicitly tells users it does the full cell pipeline. Adding "this still pays subdivision cost" would be redundant.

## Do-later list

(empty)
