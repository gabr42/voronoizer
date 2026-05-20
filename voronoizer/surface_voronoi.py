"""Surface Voronoi engine — mesh subdivision and per-face cell labelling.

Subdivide the input mesh until edges are short enough for smooth cell
boundaries, partition faces into smooth patches at sharp dihedral edges,
then assign each face the label of its closest in-patch seed by 3D
Euclidean distance from the face centroid.
"""

from __future__ import annotations

import math

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

    Uses *uniform* subdivision: every face is split into four (midpoint
    subdivision) per iteration, until the longest edge falls under the
    target. `trimesh.remesh.subdivide_to_size` would be cheaper because it
    only splits LONG edges — but on a non-uniformly-tessellated input it
    creates T-junctions (one face's midpoint isn't shared with its
    neighbour) which break the manifold property: ~1.7 % of the
    subdivided edges end up as boundary or non-manifold, the patch
    partition fragments (one CAD body went from 1 patch to 38 patches
    after subdivide_to_size), and the labelling pass can't find seeds for
    the spurious patches. Uniform subdivision preserves
    manifoldness at the cost of a denser mesh (~2.5× more faces on
    irregularly-tessellated input; identical on uniformly-tessellated
    cubes).

    Raises `ValueError` if the projected face count would exceed
    `face_cap`.
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

    # Each uniform subdivision pass quadruples the face count and halves
    # the maximum edge length. iters = ceil(log2(max_edge / target)).
    iters_needed = max(1, int(np.ceil(np.log2(max_edge / target_edge_length))))
    projected_faces = len(mesh.faces) * (4 ** iters_needed)
    if projected_faces > face_cap:
        raise ValueError(
            f"surface_voronoi.subdivide_for_geodesic: input mesh would "
            f"subdivide to {projected_faces} faces in {iters_needed} "
            f"uniform pass(es) (target edge {target_edge_length:.3f} mm), "
            f"over the {face_cap} cap. Increase --target-edge-length or "
            f"shrink the input model."
        )

    progress.log(
        f"geodesic: uniformly subdividing {len(mesh.faces)} faces, max edge "
        f"{max_edge:.3f} mm -> target {target_edge_length:.3f} mm "
        f"({iters_needed} pass(es))"
    )
    v = mesh.vertices
    f = mesh.faces
    for _ in range(iters_needed):
        v, f = trimesh.remesh.subdivide(v, f)
    sub = trimesh.Trimesh(vertices=v, faces=f, process=False)
    sub.merge_vertices()
    progress.log(
        f"geodesic: subdivided to {len(sub.faces)} faces, "
        f"{len(sub.vertices)} vertices, watertight={sub.is_watertight}"
    )
    return sub


def face_components(
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
    mesh: trimesh.Trimesh, face_comp: np.ndarray, comp_id: int
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
        c0 = int(face_comp[f0])
        c1 = int(face_comp[f1])
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
            if int(face_comp[f]) == comp_id:
                boundary_v.add(int(unique[u_idx, 0]))
                boundary_v.add(int(unique[u_idx, 1]))

    return np.array(sorted(boundary_v), dtype=np.int64)


def smooth_vertex_normals_within_patches(
    mesh: trimesh.Trimesh,
    face_comp: np.ndarray,
    iterations: int = 5,
) -> np.ndarray:
    """Laplacian-smooth vertex normals across each smooth patch.

    Trimesh's default `mesh.vertex_normals` is an area-weighted average
    of incident face normals. After uniform subdivision, every subdivided
    child face inherits its parent's normal exactly, so vertex normals
    are PIECEWISE CONSTANT per parent face — with discontinuities at the
    original face boundaries that survive subdivision. Those
    discontinuities translate to stepped prism walls in the geodesic
    engine and produce visible jagged-edge holes on low-poly CAD bodies.

    This function diffuses the vertex normals iteratively: each vertex's
    new normal is the unweighted average of its 1-ring same-patch
    neighbours' normals, renormalised. Cross-patch edges (sharp
    dihedrals) are NOT used as diffusion bridges, so sharp features
    are preserved exactly — a cube's face-interior vertex normals stay
    axis-aligned even after many iterations.

    Returns an (NV, 3) array of unit-length smoothed normals.
    """
    NV = len(mesh.vertices)
    if NV == 0:
        return mesh.vertex_normals.copy()
    normals = mesh.vertex_normals.copy()
    if iterations <= 0:
        return normals

    # Build intra-patch adjacency. Mesh.edges_unique enumerates each edge
    # once; we filter to edges whose two adjacent faces lie in the same
    # patch (so cross-patch sharp-edge connections aren't bridges for
    # normal diffusion).
    fa = mesh.face_adjacency
    fa_edges = mesh.face_adjacency_edges
    intra_edges: list[tuple[int, int]] = []
    for (f_a, f_b), (v_a, v_b) in zip(fa, fa_edges):
        if face_comp[f_a] == face_comp[f_b]:
            intra_edges.append((int(v_a), int(v_b)))

    if not intra_edges:
        return normals

    from scipy.sparse import csr_matrix
    rows = np.concatenate(
        [np.array([a for a, _ in intra_edges]), np.array([b for _, b in intra_edges])]
    )
    cols = np.concatenate(
        [np.array([b for _, b in intra_edges]), np.array([a for _, a in intra_edges])]
    )
    data = np.ones(len(rows), dtype=float)
    A = csr_matrix((data, (rows, cols)), shape=(NV, NV))
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    # Diagonal scaling factor: 1 / row_sum for vertices with neighbours, 0 otherwise.
    scale = np.where(row_sums > 0, 1.0 / np.maximum(row_sums, 1e-12), 0.0)
    # Vertices with no intra-patch neighbours (isolated in their patch) keep
    # their original normal across iterations.
    has_nbrs = row_sums > 0

    for _ in range(iterations):
        # avg[i] = mean of normals[j] for j in neighbours of i
        avg = A.dot(normals) * scale[:, None]
        # Half-step blend keeps the result close to the starting normal
        # field while still propagating smoothing — works better than a
        # full replacement on coarsely tessellated input.
        new_normals = np.where(has_nbrs[:, None], 0.5 * normals + 0.5 * avg, normals)
        norm = np.linalg.norm(new_normals, axis=1, keepdims=True)
        normals = new_normals / np.where(norm > 1e-12, norm, 1.0)
    return normals


def patch_is_flat(
    mesh: trimesh.Trimesh,
    face_comp: np.ndarray,
    patch_id: int,
    tol_deg: float = 1.0,
) -> bool:
    """True iff every face in patch `patch_id` has its normal within
    `tol_deg` of the patch's mean normal.

    A flat patch (every cube face) can have its cell inset done in the
    seed's 2D tangent plane — the projection is exact, so the 2D inset
    distance equals the surface distance and the patch-boundary clipping
    (cube-edge `shell_thickness` margin) lines up naturally.

    A curved patch (the whole sphere, a filleted region of a CAD body)
    must have its inset done on the surface itself. The orthogonal
    tangent-plane projection that we use to build the convex hull
    foreshortens points far from the seed — `R·sin(θ)` instead of
    `R·θ` — so a `strut/2` 2D inset under-shrinks the cell on the
    surface, leaving visibly wide walls between adjacent holes. Doing
    the strut/2 inset per-vertex along the surface (each vertex shifts
    `strut/2` along `n × t`, then snaps back to the mesh) gives the
    correct geodesic strut width regardless of curvature.
    """
    faces_idx = np.where(face_comp == patch_id)[0]
    if len(faces_idx) == 0:
        return True
    normals = mesh.face_normals[faces_idx]
    mean = normals.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-9:
        return False  # normals all over the place → definitely curved
    mean_unit = mean / norm
    cos_dev = normals @ mean_unit
    max_angle_rad = float(np.arccos(np.clip(cos_dev, -1.0, 1.0)).max())
    return math.degrees(max_angle_rad) < tol_deg


def assign_cell_labels(
    mesh: trimesh.Trimesh,
    seed_points: np.ndarray,
    sharp_angle_deg: float = 25.0,
) -> np.ndarray:
    """Per-face Voronoi labelling by patch-restricted Euclidean nearest-seed.

    The mesh is partitioned into smooth patches by `face_components` —
    each patch is a maximal set of faces connected through dihedral angles
    ≤ `sharp_angle_deg`. For every face we assign the label of the
    closest in-patch seed by 3D Euclidean distance from the face centroid.

    On a flat patch (a cube face) 3D Euclidean from a face centroid equals
    the true 2D in-plane distance, so it's exact. On curved patches
    (sphere, fillet) the Euclidean approximation is within the
    cell-radius / patch-radius error band.

    Faces in patches that contain no seed at all fall back to the global
    nearest seed — keeps the labelling defined everywhere.

    Returns shape (NF,) int64 with `result[f]` the index (into
    `seed_points`) of the seed that claims face `f`.
    """
    if len(seed_points) == 0:
        raise ValueError(
            "surface_voronoi.assign_cell_labels: no seeds supplied"
        )
    NF = len(mesh.faces)
    seed_points = np.asarray(seed_points, dtype=float)
    N_seeds = len(seed_points)

    fc = face_components(mesh, sharp_angle_deg)
    n_comp = int(fc.max()) + 1 if len(fc) else 0
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
    seed_comp = fc[seed_face_ids]

    face_labels = np.full(NF, -1, dtype=np.int64)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)

    seeds_by_comp: dict[int, list[int]] = {}
    for s_idx in range(N_seeds):
        seeds_by_comp.setdefault(int(seed_comp[s_idx]), []).append(s_idx)

    for comp_id in range(n_comp):
        comp_seeds = seeds_by_comp.get(comp_id, [])
        if not comp_seeds:
            continue
        comp_faces = np.where(fc == comp_id)[0]
        if len(comp_faces) == 0:
            continue
        seed_idx_arr = np.asarray(comp_seeds, dtype=np.int64)
        seed_pts = closest[seed_idx_arr]
        tree = cKDTree(seed_pts)
        _, local = tree.query(centroids[comp_faces])
        face_labels[comp_faces] = seed_idx_arr[local]

    # Fallback for patches with no seed.
    unl = face_labels < 0
    if unl.any():
        tree = cKDTree(seed_points)
        _, idx = tree.query(centroids[unl])
        face_labels[unl] = idx.astype(np.int64)
        progress.log(
            f"geodesic: {int(unl.sum())} faces in patches without any seed "
            f"fell back to global nearest-seed assignment"
        )

    return face_labels
