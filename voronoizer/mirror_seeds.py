"""Edge-neighbour generation for Voronoi cells.

Two separate concepts share this module:

* **Twins** (for sharp dihedral edges between two faces). For each near-edge
  real seed we rotate it around the edge onto the adjacent face's plane and
  add the result as a *real* seed (with the adjacent face's normal). The
  Voronoi bisector between a seed and its twin lands exactly on the edge, so
  the seed's cell and the twin's cell together form a single hole that
  wraps the edge.

* **Boundary mirrors** (for open-boundary edges — edges used by exactly one
  face, e.g. the rim of a cylinder cap submesh). There is no adjacent face
  to fold onto, so we add a *virtual* Voronoi neighbour by reflecting the
  seed across the edge line. The virtual neighbour bounds the seed's cell
  at the boundary without creating a cell of its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from voronoizer import progress


# Seed's surface normal must match an adjacent face's normal within this
# cosine to count that face as the seed's host face. cos(45°) ≈ 0.707.
_NORMAL_MATCH_COS = 0.7
_DEDUP_DECIMALS = 4


@dataclass
class TwinSeeds:
    points: np.ndarray   # (T, 3)
    normals: np.ndarray  # (T, 3)

    def __len__(self) -> int:
        return len(self.points)


def _round_key(p: np.ndarray) -> tuple[float, float, float]:
    return (
        round(float(p[0]), _DEDUP_DECIMALS),
        round(float(p[1]), _DEDUP_DECIMALS),
        round(float(p[2]), _DEDUP_DECIMALS),
    )


def _sharp_edges(
    mesh: trimesh.Trimesh, angle_deg: float
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Sharp dihedral edges as (v0, v1, n_a, n_b)."""
    if mesh.face_adjacency_angles is None or len(mesh.face_adjacency_angles) == 0:
        return []
    sharp = mesh.face_adjacency_angles > math.radians(angle_deg)
    if not sharp.any():
        return []
    edges = mesh.face_adjacency_edges[sharp]
    adj = mesh.face_adjacency[sharp]
    fn = mesh.face_normals
    return [
        (
            np.asarray(mesh.vertices[ei[0]], dtype=float),
            np.asarray(mesh.vertices[ei[1]], dtype=float),
            np.asarray(fn[fa[0]], dtype=float),
            np.asarray(fn[fa[1]], dtype=float),
        )
        for ei, fa in zip(edges, adj)
    ]


def _boundary_edges(
    mesh: trimesh.Trimesh,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Edges used by exactly one face: (v0, v1, n_face)."""
    if len(mesh.faces) == 0:
        return []
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_idx, face in enumerate(mesh.faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pair = tuple(sorted((int(face[a]), int(face[b]))))
            edge_to_faces.setdefault(pair, []).append(face_idx)
    out = []
    for pair, faces_using in edge_to_faces.items():
        if len(faces_using) == 1:
            f = faces_using[0]
            out.append((
                np.asarray(mesh.vertices[pair[0]], dtype=float),
                np.asarray(mesh.vertices[pair[1]], dtype=float),
                np.asarray(mesh.face_normals[f], dtype=float),
            ))
    return out


def _closest_on_segment(
    p: np.ndarray, v0: np.ndarray, v1: np.ndarray
) -> tuple[np.ndarray, float]:
    d = v1 - v0
    L2 = float(np.dot(d, d))
    if L2 < 1e-24:
        return v0, float(np.linalg.norm(p - v0))
    t = float(np.dot(p - v0, d)) / L2
    t = max(0.0, min(1.0, t))
    c = v0 + t * d
    return c, float(np.linalg.norm(p - c))


def _fold_onto_adjacent_face(
    seed: np.ndarray,
    edge_v0: np.ndarray,
    edge_v1: np.ndarray,
    n_host: np.ndarray,
    n_adj: np.ndarray,
) -> np.ndarray | None:
    """Rotate `seed` around the edge line so it lands on the plane of `n_adj`.

    The rotated point lies on the adjacent face's plane, at the same distance
    from the edge as the original seed was from the edge.
    """
    t = edge_v1 - edge_v0
    t_len = float(np.linalg.norm(t))
    if t_len < 1e-12:
        return None
    t = t / t_len

    # Closest point on the (infinite) edge line.
    c = edge_v0 + float(np.dot(seed - edge_v0, t)) * t
    rel = seed - c
    rel_len = float(np.linalg.norm(rel))
    if rel_len < 1e-12:
        return None  # seed is on the edge — twin would coincide

    # The adjacent face plane contains the edge and has normal n_adj. A vector
    # in that plane, perpendicular to the edge axis, is t × n_adj (up to sign).
    perp = np.cross(t, n_adj)
    pl = float(np.linalg.norm(perp))
    if pl < 1e-12:
        return None  # edge direction parallel to n_adj (degenerate)
    perp = perp / pl

    cand_a = c + rel_len * perp
    cand_b = c - rel_len * perp
    # Pick the candidate that points *away* from n_host (i.e. into the adjacent
    # face's interior side of the edge, not back into the host's side).
    if float(np.dot(cand_a - c, n_host)) <= float(np.dot(cand_b - c, n_host)):
        return cand_a
    return cand_b


def compute_sharp_edge_twins(
    mesh: trimesh.Trimesh,
    seed_points: np.ndarray,
    seed_normals: np.ndarray,
    max_twin_dist: float,
    sharp_angle_deg: float = 25.0,
) -> TwinSeeds:
    """Generate twin real-seeds across nearby sharp dihedral edges."""
    sharp = _sharp_edges(mesh, sharp_angle_deg)
    if not sharp:
        progress.log("no sharp dihedral edges; no twin seeds")
        return TwinSeeds(np.zeros((0, 3)), np.zeros((0, 3)))

    seen: set[tuple[float, float, float]] = set()
    for s in seed_points:
        seen.add(_round_key(s))

    twin_pts: list[np.ndarray] = []
    twin_normals: list[np.ndarray] = []

    for s_pos, s_norm in zip(seed_points, seed_normals):
        for v0, v1, n_a, n_b in sharp:
            dot_a = float(np.dot(s_norm, n_a))
            dot_b = float(np.dot(s_norm, n_b))
            best = max(dot_a, dot_b)
            if best < _NORMAL_MATCH_COS:
                continue
            if dot_a >= dot_b:
                n_host, n_adj = n_a, n_b
            else:
                n_host, n_adj = n_b, n_a

            _, dist = _closest_on_segment(s_pos, v0, v1)
            if dist >= max_twin_dist:
                continue

            twin = _fold_onto_adjacent_face(s_pos, v0, v1, n_host, n_adj)
            if twin is None:
                continue
            key = _round_key(twin)
            if key in seen:
                continue
            seen.add(key)
            twin_pts.append(twin)
            twin_normals.append(n_adj)

    progress.log(
        f"sharp-edge twins: {len(twin_pts)} (across {len(sharp)} sharp edges)"
    )
    if not twin_pts:
        return TwinSeeds(np.zeros((0, 3)), np.zeros((0, 3)))
    return TwinSeeds(
        points=np.asarray(twin_pts, dtype=float),
        normals=np.asarray(twin_normals, dtype=float),
    )


def compute_boundary_mirrors(
    mesh: trimesh.Trimesh,
    seed_points: np.ndarray,
    seed_normals: np.ndarray,
    max_mirror_dist: float,
) -> np.ndarray:
    """For each seed near an open-boundary edge of its host face, return a
    *virtual* Voronoi neighbour reflected across the edge line."""
    boundary = _boundary_edges(mesh)
    if not boundary:
        return np.zeros((0, 3))

    seen: set[tuple[float, float, float]] = set()
    for s in seed_points:
        seen.add(_round_key(s))

    mirrors: list[np.ndarray] = []
    for s_pos, s_norm in zip(seed_points, seed_normals):
        for v0, v1, n_a in boundary:
            if float(np.dot(s_norm, n_a)) < _NORMAL_MATCH_COS:
                continue
            closest, dist = _closest_on_segment(s_pos, v0, v1)
            if dist >= max_mirror_dist:
                continue
            m = 2.0 * closest - s_pos
            key = _round_key(m)
            if key in seen:
                continue
            seen.add(key)
            mirrors.append(m)

    progress.log(
        f"boundary mirrors: {len(mirrors)} (across {len(boundary)} boundary edges)"
    )
    if not mirrors:
        return np.zeros((0, 3))
    return np.asarray(mirrors, dtype=float)
