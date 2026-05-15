"""Surface seed sampling for Voronoi hole placement.

Seeds are sampled with Poisson-disk-ish even surface sampling. Candidates that
land within `edge_margin` of a sharp mesh edge (large dihedral angle) are
rejected — those produce Voronoi cells that mostly extend off-face and end up
as ugly edge notches rather than proper holes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from voronoizer import progress


# Faces meeting at a dihedral angle larger than this are treated as a "sharp
# edge" and seeds are kept clear of the edge curve.
_SHARP_EDGE_ANGLE_DEG = 25.0
# How densely we sample points along each sharp edge for the rejection KDTree.
_EDGE_SAMPLE_STEP_FACTOR = 0.5

_MAX_REJECT_ROUNDS = 6


@dataclass
class Seeds:
    points: np.ndarray   # (N, 3)
    normals: np.ndarray  # (N, 3), unit length

    def __len__(self) -> int:
        return len(self.points)


def _top_bottom_face_mask(mesh: trimesh.Trimesh, angle_deg: float) -> np.ndarray:
    threshold = math.cos(math.radians(angle_deg))
    nz = np.abs(mesh.face_normals[:, 2])
    return nz >= threshold


def _boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """(B, 2) vertex pairs for edges used by exactly one face. Empty for a
    closed/watertight mesh; non-empty for submeshes (e.g. caps after a
    top/bottom-only filter)."""
    if len(mesh.faces) == 0:
        return np.zeros((0, 2), dtype=int)
    faces = mesh.faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _sample_sharp_edge_points(
    mesh: trimesh.Trimesh, step: float, sharp_edge_angle_deg: float
) -> np.ndarray:
    """Return dense points along mesh edges that seeds should keep clear of.

    Two kinds count as "sharp":
      - internal edges with dihedral angle above `sharp_edge_angle_deg`
        (cube corners at 90°, or smoothly-filleted "soft" edges if the
        threshold is lowered);
      - open-boundary edges, i.e. edges used by exactly one face (these appear
        on a submesh, like the perimeter of a top/bottom-only filter).
    """
    edge_pairs: list[np.ndarray] = []

    if mesh.face_adjacency_angles is not None and len(mesh.face_adjacency_angles) > 0:
        sharp = mesh.face_adjacency_angles > math.radians(sharp_edge_angle_deg)
        if sharp.any():
            edge_pairs.append(mesh.face_adjacency_edges[sharp])

    boundary = _boundary_edges(mesh)
    if len(boundary) > 0:
        edge_pairs.append(boundary)

    if not edge_pairs:
        return np.zeros((0, 3))

    edges = np.vstack(edge_pairs)
    v0 = mesh.vertices[edges[:, 0]]
    v1 = mesh.vertices[edges[:, 1]]
    seg = v1 - v0
    lengths = np.linalg.norm(seg, axis=1)
    chunks: list[np.ndarray] = []
    for a, d, L in zip(v0, seg, lengths):
        n = max(2, int(math.ceil(L / step)) + 1)
        ts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        chunks.append(a + ts * d)
    return np.vstack(chunks)


def _auto_edge_margin(
    surface_area: float, count: int, strut_thickness: float
) -> float:
    """Default edge margin: just enough to keep seeds off the edge for
    numerical hygiene. Mirror seeds (see voronoizer.mirror_seeds) bound the
    Voronoi cells at sharp/open edges, so we no longer need a margin sized to
    the natural cell radius."""
    _ = surface_area, count  # kept for forward compatibility / future tuning
    return float(strut_thickness)


def sample_seeds(
    mesh: trimesh.Trimesh,
    count: int,
    top_bottom_only: bool,
    angle_deg: float,
    rng: np.random.Generator,
    strut_thickness: float,
    edge_margin: float | None = None,
    sharp_edge_angle_deg: float = _SHARP_EDGE_ANGLE_DEG,
) -> Seeds:
    """Sample ~`count` Poisson-disk seeds on the surface, keeping them clear of
    sharp mesh edges. `edge_margin` autosizes from seed density if None.
    `sharp_edge_angle_deg` controls which dihedrals count as "sharp" and get
    edge-rejected — lower it to push seeds away from smooth fillet curves on
    a CAD-like model whose corners aren't truly sharp."""
    src = mesh
    if top_bottom_only:
        mask = _top_bottom_face_mask(mesh, angle_deg)
        kept = int(mask.sum())
        if kept == 0:
            raise ValueError(
                f"no faces qualify as top/bottom at angle threshold {angle_deg}°; "
                "try a larger --normal-angle or drop --top-bottom-only"
            )
        progress.log(f"top/bottom faces: {kept} / {len(mesh.faces)}")
        src = mesh.submesh([np.where(mask)[0]], append=True)

    if edge_margin is None:
        edge_margin = _auto_edge_margin(float(src.area), count, strut_thickness)
    progress.log(f"edge margin: {edge_margin:.2f} mm")

    edge_step = max(0.1, edge_margin * _EDGE_SAMPLE_STEP_FACTOR)
    edge_pts = _sample_sharp_edge_points(
        src, step=edge_step, sharp_edge_angle_deg=sharp_edge_angle_deg
    )
    edge_tree = cKDTree(edge_pts) if len(edge_pts) > 0 else None
    if edge_tree is None:
        progress.log("no sharp edges detected; skipping edge rejection")
    else:
        progress.log(f"sharp-edge sample points: {len(edge_pts)}")

    accepted_points: list[np.ndarray] = []
    accepted_face_idx: list[np.ndarray] = []
    accumulated = 0

    for round_idx in range(_MAX_REJECT_ROUNDS):
        if accumulated >= count:
            break
        needed = count - accumulated
        # Over-request so rejection-sampling converges in a few rounds.
        over = max(int(needed * 2.5), needed + 16)
        seed = int(rng.integers(0, 2**31 - 1))
        pts, fidx = trimesh.sample.sample_surface_even(src, over, seed=seed)
        if len(pts) == 0:
            continue
        if edge_tree is not None:
            d, _ = edge_tree.query(pts)
            keep = d >= edge_margin
            pts = pts[keep]
            fidx = fidx[keep]
        accepted_points.append(np.asarray(pts))
        accepted_face_idx.append(np.asarray(fidx))
        accumulated += len(pts)

    if accumulated == 0:
        raise RuntimeError(
            "surface sampling produced 0 seeds after edge rejection; "
            "try lowering --holes, --strut-thickness, or --edge-margin"
        )

    all_pts = np.vstack(accepted_points)
    all_idx = np.concatenate(accepted_face_idx)
    if len(all_pts) > count:
        sel = rng.choice(len(all_pts), size=count, replace=False)
        all_pts = all_pts[sel]
        all_idx = all_idx[sel]
    elif accumulated < count:
        progress.warn(
            f"only {accumulated} seeds placed after edge rejection "
            f"(requested {count}); the surface may be too small or "
            "--edge-margin too large."
        )

    normals = src.face_normals[all_idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms > 0, norms, 1.0)

    progress.log(f"seeded {len(all_pts)} points (requested {count})")
    return Seeds(points=np.asarray(all_pts, dtype=float),
                 normals=np.asarray(normals, dtype=float))
