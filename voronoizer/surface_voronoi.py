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
from dataclasses import dataclass

import numpy as np
import trimesh

from voronoizer import progress


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

    `labels[v]` is the index of the seed (in the input seeds array) closest in
    geodesic distance to mesh vertex `v`. `distances[v]` is that distance.
    Unreachable vertices (disconnected components) get label == -1, distance ==
    inf.
    """
    labels: np.ndarray     # (V,) int64
    distances: np.ndarray  # (V,) float64


def _build_vertex_adjacency(
    mesh: trimesh.Trimesh,
) -> list[list[tuple[int, float]]]:
    """Per-vertex neighbour list: vertex -> [(neighbour, edge_length), ...]."""
    edges = mesh.edges_unique
    edge_lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
    )
    adj: list[list[tuple[int, float]]] = [[] for _ in range(len(mesh.vertices))]
    for (a, b), L in zip(edges, edge_lengths):
        a_i = int(a); b_i = int(b); L_f = float(L)
        adj[a_i].append((b_i, L_f))
        adj[b_i].append((a_i, L_f))
    return adj


def assign_cell_labels(
    mesh: trimesh.Trimesh,
    seed_points: np.ndarray,
) -> GeodesicLabels:
    """Multi-source Dijkstra labelling of mesh vertices.

    For each mesh vertex, find the index of the geodesically-closest seed
    and the distance to it. Each seed is treated as a virtual source vertex
    connected to the 3 vertices of the face it sits on; from there, edge
    relaxation uses Euclidean edge lengths.

    This approximates true geodesic distance from above (path is constrained
    to mesh edges). After Stage 1 subdivision the approximation is fine for
    boundary extraction — the boundary is then smoothed in Stage 4.
    """
    if len(seed_points) == 0:
        raise ValueError(
            "surface_voronoi.assign_cell_labels: no seeds supplied"
        )
    V = len(mesh.vertices)
    N_seeds = len(seed_points)

    adj = _build_vertex_adjacency(mesh)

    # Snap each seed onto the mesh and record which face it landed on.
    from trimesh.proximity import ProximityQuery
    pq = ProximityQuery(mesh)
    closest, _dist, face_ids = pq.on_surface(np.asarray(seed_points, dtype=float))
    face_ids = np.asarray(face_ids, dtype=int)
    closest = np.asarray(closest, dtype=float)
    faces = mesh.faces

    INF = float("inf")
    labels = np.full(V, -1, dtype=np.int64)
    distances = np.full(V, INF, dtype=np.float64)

    # Heap entries: (distance, vertex_idx, seed_idx). When a vertex is popped
    # with a strictly smaller distance than already recorded, it inherits the
    # popping seed's label; stale entries with larger distance are skipped.
    heap: list[tuple[float, int, int]] = []
    for s_idx in range(N_seeds):
        f = int(face_ids[s_idx])
        cp = closest[s_idx]
        for v_idx in faces[f]:
            v_i = int(v_idx)
            d0 = float(np.linalg.norm(mesh.vertices[v_i] - cp))
            heapq.heappush(heap, (d0, v_i, s_idx))

    while heap:
        d, v, s = heapq.heappop(heap)
        if d >= distances[v]:
            continue
        distances[v] = d
        labels[v] = s
        for nb, L in adj[v]:
            nd = d + L
            if nd < distances[nb]:
                heapq.heappush(heap, (nd, nb, s))

    unreached = int((labels < 0).sum())
    if unreached:
        progress.warn(
            f"geodesic: {unreached} mesh vertices unreachable from any seed "
            f"(disconnected component?)"
        )

    return GeodesicLabels(labels=labels, distances=distances)


def face_labels_from_vertex_labels(
    mesh: trimesh.Trimesh, labels: GeodesicLabels
) -> np.ndarray:
    """Project per-vertex labels onto per-face labels.

    Each face is assigned the label of its vertex with the smallest geodesic
    distance — i.e., the seed whose Voronoi cell most strongly "claims" the
    face. This gives a clean per-face partition; the boundary lies on mesh
    edges shared between faces with different labels.
    """
    faces = mesh.faces
    vd = labels.distances[faces]            # (F, 3)
    vl = labels.labels[faces]               # (F, 3)
    nearest = np.argmin(vd, axis=1)         # (F,)
    rows = np.arange(len(faces))
    return vl[rows, nearest].astype(np.int64)
