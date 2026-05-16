"""Geodesic Voronoi engine — Stages 3, 4 & 5 of the Phase 2 pipeline.

Given per-face cell labels from Stage 2, we:
  * extract boundary half-edges (Stage 3),
  * walk them per-cell into closed 3D polylines on the mesh surface,
  * smooth each polyline with a quadratic Bézier pass + surface re-projection
    (Stage 4),
  * inset each polyline inward by `strut / 2` along the surface tangent
    perpendicular to the local boundary direction (Stage 5).

Each `Loop` carries per-vertex 3D positions and the face index each position
sits on — that face's normal becomes the local outward surface normal used by
Stage 5 and Stage 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


# Bézier smoothing matches Phase 1's tangent-frame smoothing.
_BEZIER_SAMPLES_PER_EDGE = 6


@dataclass
class Loop:
    """A closed 3D polyline on the mesh surface, owned by one cell.

    `positions[i]` is the 3D position of the i-th vertex of the loop.
    `face_ids[i]` is the index of the mesh face that position sits on.
    `normals[i]` is the smoothed surface normal at that position
    (barycentric interpolation of the face's vertex-normals); using this
    rather than the raw face normal keeps the per-vertex frame continuous
    across sharp surface edges, which is what manifold3d needs to avoid
    twin-vertex artefacts when a cell wraps around a cube corner / fillet.

    The loop is closed by convention — `positions[-1]` connects back to
    `positions[0]`. The walking convention keeps the cell on the LEFT when
    viewed from outside (n × forward_tangent points inward into the cell).
    """
    positions: np.ndarray   # (N, 3) float
    face_ids: np.ndarray    # (N,) int
    normals: np.ndarray     # (N, 3) float, unit length

    def __len__(self) -> int:
        return len(self.positions)


def _smoothed_normals_at(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    face_ids: np.ndarray,
) -> np.ndarray:
    """Barycentric interpolation of mesh vertex-normals at each position.

    Smoothed normals are continuous across the surface — at a sharp cube
    edge, a vertex sitting on the edge gets the average of the two
    adjacent face normals (i.e. a diagonal), not one or the other.
    `mesh.vertex_normals` is trimesh's area-weighted vertex normal field.
    """
    faces = mesh.faces[face_ids]                    # (N, 3) int
    v0 = mesh.vertices[faces[:, 0]]                 # (N, 3)
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]
    # Barycentric coords by area subdivision.
    n_tri = np.cross(v1 - v0, v2 - v0)
    area_tri = np.linalg.norm(n_tri, axis=1, keepdims=True)
    inv_area = 1.0 / np.maximum(area_tri, 1e-12)
    p = positions
    a0 = np.linalg.norm(np.cross(v1 - p, v2 - p), axis=1, keepdims=True) * inv_area
    a1 = np.linalg.norm(np.cross(v2 - p, v0 - p), axis=1, keepdims=True) * inv_area
    a2 = np.linalg.norm(np.cross(v0 - p, v1 - p), axis=1, keepdims=True) * inv_area
    bary = np.hstack([a0, a1, a2])
    bary = bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
    vn = mesh.vertex_normals[faces]                 # (N, 3, 3)
    smoothed = (bary[:, :, None] * vn).sum(axis=1)  # (N, 3)
    smoothed = smoothed / np.maximum(
        np.linalg.norm(smoothed, axis=1, keepdims=True), 1e-12
    )
    return smoothed


def _boundary_half_edges_per_cell(
    mesh: trimesh.Trimesh, face_labels: np.ndarray
) -> dict[int, list[tuple[int, int, int]]]:
    """For each cell, list its directed boundary half-edges.

    Each entry is `(v_from, v_to, owning_face)`. The owning face is one of
    the cell's faces — its normal gives the local surface normal for the
    half-edge, and its winding fixes the half-edge direction so the cell
    is on the left.
    """
    F = mesh.faces
    NF = len(F)

    # For each undirected edge, list the (face, local_edge_index) half-edges
    # incident to it. A manifold internal edge has exactly two; boundary or
    # non-manifold edges have one or more — those are skipped.
    edge_he: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for fi in range(NF):
        f0, f1, f2 = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        for k, (a, b) in enumerate(((f0, f1), (f1, f2), (f2, f0))):
            key = (a, b) if a < b else (b, a)
            edge_he.setdefault(key, []).append((fi, k))

    out: dict[int, list[tuple[int, int, int]]] = {}
    for hes in edge_he.values():
        if len(hes) != 2:
            continue
        (fA, kA), (fB, kB) = hes
        lA = int(face_labels[fA])
        lB = int(face_labels[fB])
        if lA == lB:
            continue
        # Half-edge on face fA goes from F[fA, kA] to F[fA, (kA+1)%3], with
        # cell lA on the left (assuming consistent CCW-from-outside winding).
        aA = int(F[fA, kA]); bA = int(F[fA, (kA + 1) % 3])
        out.setdefault(lA, []).append((aA, bA, fA))
        aB = int(F[fB, kB]); bB = int(F[fB, (kB + 1) % 3])
        out.setdefault(lB, []).append((aB, bB, fB))
    return out


def _walk_into_loops(
    half_edges: list[tuple[int, int, int]],
) -> list[list[tuple[int, int]]]:
    """Walk directed half-edges into closed loops.

    Each input entry is `(v_from, v_to, owning_face)`. Output: list of loops,
    each loop a list of `(v, face)` pairs (one entry per loop vertex, in
    walking order, closing back to start without repeating the first vertex).
    """
    outgoing: dict[int, list[int]] = {}  # v_from -> list of edge indices
    for idx, (a, _b, _f) in enumerate(half_edges):
        outgoing.setdefault(a, []).append(idx)

    used = [False] * len(half_edges)
    loops: list[list[tuple[int, int]]] = []

    for seed in range(len(half_edges)):
        if used[seed]:
            continue
        a0, _b0, _f0 = half_edges[seed]
        loop: list[tuple[int, int]] = []
        cur_edge = seed
        while True:
            used[cur_edge] = True
            a, b, f = half_edges[cur_edge]
            loop.append((a, f))
            if b == a0:
                break
            # Next half-edge: any outgoing from b that we haven't used yet.
            # In well-behaved cases there's exactly one. At junction vertices
            # (3+ cells meeting at a point) there can be multiple — but each
            # cell still has exactly one out-half-edge from any of its
            # boundary vertices, so the unused-outgoing pick is unique.
            nxts = [e for e in outgoing.get(b, ()) if not used[e]]
            if not nxts:
                # Open chain (e.g. boundary touches an open mesh edge). Save
                # what we have and stop walking this component.
                loop.append((b, f))
                break
            cur_edge = nxts[0]
        loops.append(loop)
    return loops


def extract_cell_loops(
    mesh: trimesh.Trimesh, face_labels: np.ndarray
) -> dict[int, list[Loop]]:
    """Build closed 3D boundary loops for every cell.

    Returns `cell_id -> [Loop, ...]`. Cells with no boundary half-edges
    (the only cell on a connected mesh component, or a degenerate empty
    cell) get an empty list.
    """
    cell_he = _boundary_half_edges_per_cell(mesh, face_labels)
    result: dict[int, list[Loop]] = {}
    for cell_id, hes in cell_he.items():
        loops_idx = _walk_into_loops(hes)
        cell_loops: list[Loop] = []
        for loop in loops_idx:
            if len(loop) < 3:
                continue
            v_idx = np.asarray([v for v, _f in loop], dtype=np.int64)
            f_idx = np.asarray([f for _v, f in loop], dtype=np.int64)
            positions = mesh.vertices[v_idx].astype(float)
            normals = _smoothed_normals_at(mesh, positions, f_idx)
            cell_loops.append(Loop(
                positions=positions,
                face_ids=f_idx,
                normals=normals,
            ))
        result[int(cell_id)] = cell_loops
    return result


# ---------------------------------------------------------------------------
# Loop resampling (preprocessing for Stage 4).
# ---------------------------------------------------------------------------


# Floor on the resampled vertex count before Bézier — guards against very
# tiny cells degenerating to a triangle that the Bézier pass can't smooth
# meaningfully. Phase 1 typically produced 6–10 inset polygon vertices; a
# higher floor here gives Phase 2 enough resolution to follow surface
# curvature even on a small cell.
_MIN_RESAMPLED_VERTICES = 12


def _polyline_arclengths(positions: np.ndarray) -> np.ndarray:
    """Cumulative arc-length along the closed polyline.

    Returns shape (N+1,) with `t[0] == 0` and `t[-1] == perimeter`.
    """
    seg = np.linalg.norm(np.roll(positions, -1, axis=0) - positions, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def resample_loop_arclen(
    loop: Loop,
    mesh: trimesh.Trimesh,
    proximity,
    target_step: float,
    min_vertices: int = _MIN_RESAMPLED_VERTICES,
) -> Loop:
    """Resample the loop to roughly uniform arc-length spacing.

    The raw Dijkstra loop follows mesh edges and can have hundreds of
    vertices when Stage 1 over-subdivides for boundary quality. Downstream
    prism construction wants a much coarser polygon (Phase 1 used 6–10
    vertices); resampling here decouples loop density from mesh density.
    """
    if len(loop) < 3:
        return loop
    cum = _polyline_arclengths(loop.positions)
    perim = float(cum[-1])
    if perim <= 0:
        return loop
    n_target = max(min_vertices, int(round(perim / max(target_step, 1e-6))))
    # Uniform sample positions in arc-length, dropping the closing duplicate.
    ts = np.linspace(0.0, perim, n_target, endpoint=False)
    # For each ts[j], find which segment it falls in and the interpolation t.
    # cum is monotonically increasing of length N+1; segment i covers
    # [cum[i], cum[i+1]).
    seg_idx = np.searchsorted(cum, ts, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(loop) - 1)
    seg_start = cum[seg_idx]
    seg_len = cum[seg_idx + 1] - seg_start
    seg_len = np.where(seg_len > 1e-12, seg_len, 1.0)
    alpha = ((ts - seg_start) / seg_len).reshape(-1, 1)
    a = loop.positions[seg_idx]
    b = loop.positions[(seg_idx + 1) % len(loop)]
    resampled = (1.0 - alpha) * a + alpha * b
    snapped, _dist, face_ids = proximity.on_surface(resampled)
    snapped = np.asarray(snapped, dtype=float)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    return Loop(
        positions=snapped,
        face_ids=face_ids,
        normals=_smoothed_normals_at(mesh, snapped, face_ids),
    )


# ---------------------------------------------------------------------------
# Stage 4 — Bézier smoothing on the surface.
# ---------------------------------------------------------------------------


def _bezier_samples(
    polyline: np.ndarray, samples_per_edge: int
) -> np.ndarray:
    """Quadratic Bézier smoothing of a closed 3D polyline.

    Mirrors Phase 1's `_bezier_smooth`: for each edge V[i] -> V[i+1] take the
    midpoints as endpoints and V[i] as the control point. Returns the
    concatenated sample points (`N * samples_per_edge` rows).
    """
    N = len(polyline)
    nxt = (np.arange(N) + 1) % N
    mid = (polyline + polyline[nxt]) * 0.5
    prev_mid = np.roll(mid, 1, axis=0)
    ts = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False).reshape(-1, 1)
    chunks: list[np.ndarray] = []
    for i in range(N):
        p0, p1, p2 = prev_mid[i], polyline[i], mid[i]
        chunks.append(((1 - ts) ** 2) * p0
                      + (2 * (1 - ts) * ts) * p1
                      + (ts ** 2) * p2)
    return np.vstack(chunks)


def bezier_smooth_on_surface(
    loop: Loop,
    mesh: trimesh.Trimesh,
    proximity,
    samples_per_edge: int = _BEZIER_SAMPLES_PER_EDGE,
) -> Loop:
    """One Bézier smoothing pass on the loop, with surface re-projection.

    `proximity` is a `trimesh.proximity.ProximityQuery` over `mesh`; it's
    accepted as a parameter so callers can build it once per pipeline run
    rather than once per cell.
    """
    if len(loop) < 3:
        return loop
    sampled = _bezier_samples(loop.positions, samples_per_edge)
    snapped, _dist, face_ids = proximity.on_surface(sampled)
    snapped = np.asarray(snapped, dtype=float)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    return Loop(
        positions=snapped,
        face_ids=face_ids,
        normals=_smoothed_normals_at(mesh, snapped, face_ids),
    )


# ---------------------------------------------------------------------------
# Stage 5 — Surface inset.
# ---------------------------------------------------------------------------


def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


def dedupe_loop(
    loop: Loop, mesh: trimesh.Trimesh, min_segment_length: float
) -> Loop:
    """Drop loop vertices whose edge to the next vertex is too short.

    After Bézier smoothing followed by inset + surface re-projection, two
    consecutive loop vertices can land at near-coincident 3D positions —
    especially on inputs with sharp edges, where one face's inset
    projection lands near another face's projection. The resulting
    near-zero-length segment makes a degenerate prism wall sliver that
    manifold3d turns into twin vertices in the final boolean.
    """
    if len(loop) < 4:
        return loop
    P = loop.positions
    N = len(P)
    keep = np.ones(N, dtype=bool)
    # Greedy: walk forward, drop next vertex if too close, keep doing so
    # until a far-enough vertex is found.
    last_kept = 0
    for i in range(1, N):
        d = float(np.linalg.norm(P[i] - P[last_kept]))
        if d < min_segment_length:
            keep[i] = False
        else:
            last_kept = i
    # Close the loop: if the last kept vertex is too close to the first,
    # drop it too (don't break the polygon if doing so leaves < 3 verts).
    while keep.sum() > 3 and (
        np.linalg.norm(P[np.where(keep)[0][-1]] - P[np.where(keep)[0][0]])
        < min_segment_length
    ):
        keep[np.where(keep)[0][-1]] = False
    if keep.sum() < 3:
        return loop  # would degenerate; leave as-is
    return Loop(
        positions=P[keep],
        face_ids=loop.face_ids[keep],
        normals=loop.normals[keep],
    )


def inset_loop_on_surface(
    loop: Loop,
    mesh: trimesh.Trimesh,
    proximity,
    inset: float,
) -> Loop:
    """Shift each loop vertex inward by `inset` mm along the surface tangent.

    At each vertex V_i:
      * forward tangent t_i = normalize(V[i+1] - V[i-1])
      * outward surface normal n_i = face_normals[loop.face_ids[i]]
      * inward direction d_i = normalize(n_i × t_i)  (cell on the left of the
        walking direction, so n × t points into the cell)
      * V_i' = V_i + inset * d_i, then re-projected onto the mesh.

    The cross-product convention follows trimesh's outward-CCW face winding;
    if a mesh has inconsistent winding the boolean step downstream will
    already have failed, so we don't sign-correct per vertex here.
    """
    if len(loop) < 3:
        return loop
    P = loop.positions
    nxt = np.roll(P, -1, axis=0)
    prv = np.roll(P,  1, axis=0)
    tangents = _normalize_rows(nxt - prv)
    # For the inset DIRECTION specifically we use the smoothed normal — it's
    # continuous across sharp edges, so the inset displacement direction
    # transitions smoothly there. (The prism builder downstream uses the
    # discontinuous face normal for the ring frames — see surface_prism.)
    inward = _normalize_rows(np.cross(loop.normals, tangents))
    shifted = P + inset * inward
    snapped, _dist, face_ids = proximity.on_surface(shifted)
    snapped = np.asarray(snapped, dtype=float)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    return Loop(
        positions=snapped,
        face_ids=face_ids,
        normals=_smoothed_normals_at(mesh, snapped, face_ids),
    )
