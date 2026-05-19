"""Geodesic Voronoi engine — Stages 1 & 2 of the Phase 2 pipeline.

Stage 1: subdivide the input mesh so Dijkstra's edge-graph approximation of
geodesic distance has enough resolution for smooth cell boundaries.

Stage 2: multi-source Dijkstra on the subdivided mesh's edge graph. Each input
seed is connected as a virtual source vertex to the three vertices of the face
it landed on; from there, edge weights are Euclidean edge lengths. Output is a
per-mesh-vertex (label, distance) pair: the index of the closest seed and the
geodesic distance to it.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import trimesh

from voronoizer import progress


# Multiplier applied to mesh-edge weights that cross a "sharp" dihedral.
# Needs to be large enough that the cheapest "go around" path on the
# original face is preferred over crossing a sharp edge — for a 40 mm
# cube with 0.75 mm subdivision, the maximum on-face geodesic between
# two seeds is ~100 mm and one sharp-edge step is 0.75 mm, so the
# multiplier must comfortably exceed 100 mm / 0.75 mm ≈ 130×. A
# generous 10 000× makes the barrier effectively unbreachable for any
# realistic mesh size while still leaving the graph connected (cells
# *can* spill across a sharp edge if their face has no seed at all).
_SHARP_EDGE_COST_MULT = 10000.0


# Hard cap on the subdivided face count. Real prints stay well under this; the
# cap exists to make pathological inputs fail fast with a clear message rather
# than silently churn for minutes and exhaust memory.
_DEFAULT_FACE_CAP = 500_000


def subdivide_for_geodesic(
    mesh: trimesh.Trimesh,
    target_edge_length: float,
    face_cap: int = _DEFAULT_FACE_CAP,
) -> trimesh.Trimesh:
    """Return a copy of `mesh` whose every edge is at most `target_edge_length`.

    Uses trimesh's `subdivide_to_size` which recursively splits long edges.
    Skips subdivision entirely if the mesh is already fine enough — that's
    the common case for already-subdivided CAD output or high-poly prints.

    Raises `ValueError` if the projected face count would exceed `face_cap`.
    """
    if target_edge_length <= 0:
        raise ValueError(
            f"surface_voronoi.subdivide_for_geodesic: target_edge_length "
            f"must be > 0, got {target_edge_length}"
        )

    edges = mesh.edges_unique
    edge_lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
    )
    max_edge = float(edge_lengths.max()) if len(edge_lengths) else 0.0

    if max_edge <= target_edge_length:
        progress.log(
            f"geodesic: mesh already fine ({len(mesh.faces)} faces, "
            f"max edge {max_edge:.3f} mm <= target {target_edge_length:.3f} mm)"
        )
        return mesh

    # Rough projection of face growth: each edge longer than the target gets
    # split log2(L/target) times, splitting one triangle into ~4 each time.
    # This is a loose upper bound — typically the real count is lower because
    # neighbouring triangles share split edges.
    long_edges = edge_lengths > target_edge_length
    if long_edges.any():
        ratios = edge_lengths[long_edges] / target_edge_length
        splits = np.ceil(np.log2(ratios)).astype(int).clip(min=1)
        projected_faces = int(len(mesh.faces) + (4 ** splits).sum())
    else:
        projected_faces = len(mesh.faces)

    if projected_faces > face_cap * 4:
        # The 4x slack accounts for the bound being loose; if even that is over
        # the cap, refuse outright.
        raise ValueError(
            f"surface_voronoi.subdivide_for_geodesic: input mesh would "
            f"subdivide to ~{projected_faces} faces at target edge length "
            f"{target_edge_length:.3f} mm, well over the {face_cap} cap. "
            f"Increase --target-edge-length or use --engine tangent."
        )

    progress.log(
        f"geodesic: subdividing {len(mesh.faces)} faces, max edge "
        f"{max_edge:.3f} mm -> target {target_edge_length:.3f} mm"
    )
    new_vertices, new_faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=target_edge_length
    )

    if len(new_faces) > face_cap:
        raise ValueError(
            f"surface_voronoi.subdivide_for_geodesic: subdivided mesh has "
            f"{len(new_faces)} faces, over the {face_cap} cap. Increase "
            f"--target-edge-length or use --engine tangent."
        )

    sub = trimesh.Trimesh(
        vertices=new_vertices, faces=new_faces, process=False
    )
    # Preserve face winding from the input; subdivide_to_size keeps orientation
    # but we run merge_vertices to drop duplicate vertices that recursive
    # splitting can introduce on shared edges.
    sub.merge_vertices()
    progress.log(
        f"geodesic: subdivided to {len(sub.faces)} faces, "
        f"{len(sub.vertices)} vertices"
    )
    return sub


@dataclass
class GeodesicLabels:
    """Output of multi-source Dijkstra on the mesh edge graph.

    `face_labels[f]` is the index of the seed (in the input seeds array)
    whose Voronoi cell claims face f. We label faces, not vertices: a
    vertex on a sharp cube edge belongs to multiple smooth patches at once
    and would need a different label per patch — face-level labelling
    avoids the ambiguity (every face sits in exactly one patch).

    `face_distances[f]` is the geodesic distance from face f's nearest
    vertex to the claiming seed. Sharp-edge barriers prevent Dijkstra
    paths from crossing patch boundaries except where there is no other
    option (a patch with no seed in it).
    """
    face_labels: np.ndarray     # (F,) int64
    face_distances: np.ndarray  # (F,) float64


def _sharp_edge_keys(
    mesh: trimesh.Trimesh, sharp_angle_deg: float
) -> set[tuple[int, int]]:
    """Set of `(min(v_a, v_b), max(v_a, v_b))` keys for mesh edges whose
    dihedral angle exceeds the threshold.

    `mesh.face_adjacency_angles` gives the angle at each internal edge; edges
    on the open boundary (1-face edges) aren't sharp by definition (no
    dihedral to compute).
    """
    if sharp_angle_deg >= 179.5:
        return set()
    angles = mesh.face_adjacency_angles
    if len(angles) == 0:
        return set()
    sharp = angles > math.radians(sharp_angle_deg)
    if not sharp.any():
        return set()
    sharp_edges = mesh.face_adjacency_edges[sharp]
    return {
        (int(min(a, b)), int(max(a, b)))
        for a, b in sharp_edges
    }


def _face_components(
    mesh: trimesh.Trimesh, sharp_angle_deg: float
) -> np.ndarray:
    """Partition mesh faces into "patches" — connected components when
    sharp-edge face-adjacencies are removed.

    Returns shape (NF,) int array where `comp[f]` is the patch id of face f.
    On a 40 mm cube with default threshold this gives 6 patches (one per
    cube face). On a sphere / fillet (no sharp edges) this gives 1 patch.

    Implemented as a union-find over face-adjacency pairs whose dihedral
    angle is below the threshold.
    """
    NF = len(mesh.faces)
    if NF == 0:
        return np.zeros(0, dtype=np.int64)
    parent = np.arange(NF, dtype=np.int64)

    def find(x: int) -> int:
        # Iterative path compression.
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = root, int(parent[x])
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    fa = mesh.face_adjacency
    angles = mesh.face_adjacency_angles
    if len(fa) > 0:
        thr = math.radians(sharp_angle_deg) if sharp_angle_deg < 179.5 else math.pi
        smooth_mask = angles <= thr
        for (i, j) in fa[smooth_mask]:
            union(int(i), int(j))

    # Compress and re-label to contiguous component ids.
    roots = np.array([find(f) for f in range(NF)], dtype=np.int64)
    _, comp = np.unique(roots, return_inverse=True)
    return comp.astype(np.int64)


def patch_boundary_vertex_indices(
    mesh: trimesh.Trimesh, face_components: np.ndarray, comp_id: int
) -> np.ndarray:
    """Mesh vertex indices on the boundary of patch `comp_id`.

    Boundary = mesh edges where one adjacent face is in the patch and the
    other is in a different patch (or the edge is on the open mesh
    boundary, in which case the single adjacent face is in this patch).

    Used downstream to compute 2D clipping half-planes: a cell's tangent
    polygon must stay at least `shell_thickness` away from these boundary
    vertices, otherwise the cell's inner ring extends into the adjacent
    patch's wall material and the boolean subtraction "eats" the
    neighbouring face's shell.
    """
    F = mesh.faces
    fa = mesh.face_adjacency
    fa_edges = mesh.face_adjacency_edges

    boundary_v: set[int] = set()
    for (f0, f1), (va, vb) in zip(fa, fa_edges):
        c0 = int(face_components[f0])
        c1 = int(face_components[f1])
        if (c0 == comp_id) != (c1 == comp_id):
            boundary_v.add(int(va))
            boundary_v.add(int(vb))

    NF = len(F)
    if NF > 0:
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        face_per_edge = np.repeat(np.arange(NF), 3)
        edges_sorted = np.sort(edges, axis=1)
        unique, inv, counts = np.unique(
            edges_sorted, axis=0, return_inverse=True, return_counts=True
        )
        boundary_unique_idx = np.where(counts == 1)[0]
        for u_idx in boundary_unique_idx:
            first = int(np.where(inv == u_idx)[0][0])
            f = int(face_per_edge[first])
            if int(face_components[f]) == comp_id:
                boundary_v.add(int(unique[u_idx, 0]))
                boundary_v.add(int(unique[u_idx, 1]))

    return np.array(sorted(boundary_v), dtype=np.int64)


def face_components(
    mesh: trimesh.Trimesh, sharp_angle_deg: float
) -> np.ndarray:
    """Public wrapper for `_face_components` so callers outside this
    module (e.g. the pipeline orchestrator) can compute the partition
    once and pass it around without recomputing it inside
    `assign_cell_labels`.
    """
    return _face_components(mesh, sharp_angle_deg)


def _build_component_adjacency(
    mesh: trimesh.Trimesh, face_components: np.ndarray, comp_id: int
) -> dict[int, list[tuple[int, float]]]:
    """Per-vertex neighbour list restricted to faces in `comp_id`.

    Only mesh edges whose BOTH adjacent faces are in component `comp_id`
    are included. Edges on the patch boundary (one face in, one face out)
    are NOT included — that's the barrier: paths can't leave the patch.
    Open boundary edges (one adjacent face) are included if that face is
    in the patch.
    """
    F = mesh.faces
    NF = len(F)
    edge_lengths = {}  # (a, b) sorted -> length; computed lazily
    adj: dict[int, list[tuple[int, float]]] = {}

    def add(v_a: int, v_b: int, length: float) -> None:
        adj.setdefault(v_a, []).append((v_b, length))
        adj.setdefault(v_b, []).append((v_a, length))

    # Walk each face once; for each of its 3 edges, add the edge to `adj`
    # only if the other face across the edge is also in `comp_id` (or the
    # edge is on the open boundary). Mesh.face_adjacency gives the pairs
    # of internal-edge faces; complement set gives boundary edges.
    fa = mesh.face_adjacency
    fa_edges = mesh.face_adjacency_edges
    seen: set[tuple[int, int]] = set()
    for (f0, f1), (va, vb) in zip(fa, fa_edges):
        c0 = int(face_components[f0])
        c1 = int(face_components[f1])
        if c0 != comp_id and c1 != comp_id:
            continue
        if c0 != c1:
            # Cross-patch edge — that's the barrier; skip.
            continue
        a_i = int(va); b_i = int(vb)
        key = (min(a_i, b_i), max(a_i, b_i))
        if key in seen:
            continue
        seen.add(key)
        L = float(np.linalg.norm(mesh.vertices[a_i] - mesh.vertices[b_i]))
        add(a_i, b_i, L)

    # Open-boundary edges (faces used by exactly one triangle): include
    # if that triangle is in comp_id.
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    face_per_edge = np.repeat(np.arange(NF), 3)
    edges_sorted = np.sort(edges, axis=1)
    unique, inv, counts = np.unique(
        edges_sorted, axis=0, return_inverse=True, return_counts=True
    )
    boundary_unique_idx = np.where(counts == 1)[0]
    for u_idx in boundary_unique_idx:
        # Find the one face using this edge.
        first = int(np.where(inv == u_idx)[0][0])
        f = int(face_per_edge[first])
        if int(face_components[f]) != comp_id:
            continue
        a_i, b_i = int(unique[u_idx, 0]), int(unique[u_idx, 1])
        key = (min(a_i, b_i), max(a_i, b_i))
        if key in seen:
            continue
        seen.add(key)
        L = float(np.linalg.norm(mesh.vertices[a_i] - mesh.vertices[b_i]))
        add(a_i, b_i, L)

    return adj


def _faces_in_component(
    face_components: np.ndarray, comp_id: int
) -> np.ndarray:
    return np.where(face_components == comp_id)[0]


def _vertices_in_component(
    mesh: trimesh.Trimesh, face_components: np.ndarray, comp_id: int
) -> np.ndarray:
    faces_idx = _faces_in_component(face_components, comp_id)
    return np.unique(mesh.faces[faces_idx].ravel())


def assign_cell_labels(
    mesh: trimesh.Trimesh,
    seed_points: np.ndarray,
    sharp_angle_deg: float = 25.0,
    sharp_multiplier: float = _SHARP_EDGE_COST_MULT,
) -> GeodesicLabels:
    """Per-face Voronoi labelling by component-restricted Euclidean
    nearest-seed.

    The mesh is partitioned into smooth patches by `_face_components` —
    each patch is a maximal set of faces connected through dihedral angles
    ≤ `sharp_angle_deg`. For every face we assign the label of the
    geodesically-closest seed *within that face's patch*, approximated by
    3D Euclidean distance from the face centroid to each in-patch seed.

    Why not edge-graph Dijkstra: on a regular triangulated grid the
    in-mesh graph has 4 axis-aligned edges plus 2 diagonal edges per
    vertex, with the diagonals running in only ONE direction. Shortest
    paths in that graph collapse toward Manhattan distance whenever the
    target is in the orientation the diagonals don't help with — biasing
    Voronoi cells by ~35 % on sparse seedings. For a flat patch (a cube
    face) 3D Euclidean from a face centroid equals the true 2D in-plane
    distance, so it's exact. For curved patches (sphere, fillet) the
    Euclidean approximation is within the cell-radius / patch-radius
    error band, comparable to Phase 1's tangent-plane approximation.

    Faces in patches that contain no seed at all fall back to the
    nearest seed by 3D Euclidean from any patch — keeps the labelling
    defined everywhere.

    The `sharp_multiplier` argument is accepted for API compatibility
    but is unused; the patch decomposition is the actual barrier.
    """
    _ = sharp_multiplier
    if len(seed_points) == 0:
        raise ValueError(
            "surface_voronoi.assign_cell_labels: no seeds supplied"
        )
    NF = len(mesh.faces)
    seed_points = np.asarray(seed_points, dtype=float)
    N_seeds = len(seed_points)

    face_components = _face_components(mesh, sharp_angle_deg)
    n_comp = int(face_components.max()) + 1 if len(face_components) else 0
    progress.log(
        f"geodesic: mesh partitions into {n_comp} smooth patch(es) "
        f"at sharp-edge threshold {sharp_angle_deg:.1f}°"
    )

    # Snap each seed onto the mesh; record landing face and patch.
    from trimesh.proximity import ProximityQuery
    from scipy.spatial import cKDTree
    pq = ProximityQuery(mesh)
    closest, _dist, seed_face_ids = pq.on_surface(seed_points)
    seed_face_ids = np.asarray(seed_face_ids, dtype=int)
    closest = np.asarray(closest, dtype=float)
    seed_comp = face_components[seed_face_ids]

    INF = float("inf")
    face_labels = np.full(NF, -1, dtype=np.int64)
    face_distances = np.full(NF, INF, dtype=np.float64)

    centroids = mesh.vertices[mesh.faces].mean(axis=1)

    seeds_by_comp: dict[int, list[int]] = {}
    for s_idx in range(N_seeds):
        seeds_by_comp.setdefault(int(seed_comp[s_idx]), []).append(s_idx)

    for comp_id in range(n_comp):
        comp_seeds = seeds_by_comp.get(comp_id, [])
        if not comp_seeds:
            continue
        comp_faces = _faces_in_component(face_components, comp_id)
        if len(comp_faces) == 0:
            continue
        seed_idx_arr = np.asarray(comp_seeds, dtype=np.int64)
        seed_pts = closest[seed_idx_arr]
        tree = cKDTree(seed_pts)
        d, local = tree.query(centroids[comp_faces])
        face_labels[comp_faces] = seed_idx_arr[local]
        face_distances[comp_faces] = d

    # Fallback for patches with no seed.
    unl = face_labels < 0
    if unl.any():
        tree = cKDTree(seed_points)
        d, idx = tree.query(centroids[unl])
        face_labels[unl] = idx.astype(np.int64)
        face_distances[unl] = d
        progress.warn(
            f"geodesic: {int(unl.sum())} faces in patches without any seed "
            f"fell back to global nearest-seed assignment"
        )

    return GeodesicLabels(
        face_labels=face_labels, face_distances=face_distances
    )


def face_labels_from_vertex_labels(
    mesh: trimesh.Trimesh, labels: GeodesicLabels
) -> np.ndarray:
    """Compatibility shim — labels are now per-face natively."""
    _ = mesh  # unused; kept for call-site compatibility
    return labels.face_labels
