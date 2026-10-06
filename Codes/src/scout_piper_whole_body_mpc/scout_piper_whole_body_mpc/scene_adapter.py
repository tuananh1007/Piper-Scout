"""Geometry sources for the MPC: semantic-scene query or analytic fields."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def semantic_distance_fn(query, now: Optional[float] = None, **kw):
    """Wrap ``SemanticDistanceQuery`` (scout_piper_scene_repr) as a DistanceFn.

    The query already clamps unknown/stale space to ≤ 0 and reports ``valid``;
    both are passed through, so the MPC and the safety filter treat unknown
    geometry conservatively."""
    def fn(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = query.query(points, now=now, gradient=False, **kw)   # MPPI needs no gradient
        return r.hard_distance, r.valid
    return fn


def semantic_leaf_fn(query, d_soft: float = 0.02):
    def fn(points: np.ndarray) -> np.ndarray:
        cost, _ = query.leaf_cost(points, d_soft=d_soft)
        return cost
    return fn


def spheres_distance_fn(centers: Sequence[Sequence[float]], radii: Sequence[float]):
    """Analytic field: union of spherical obstacles, everything observed/valid.
    Used for synthetic scenes (WE2/WE3) before the semantic map is connected."""
    C = np.asarray(centers, float).reshape(-1, 3)
    R = np.asarray(radii, float)

    def fn(points: np.ndarray):
        p = np.asarray(points, float).reshape(-1, 3)
        d = np.full(len(p), np.inf)
        if len(C):
            d = (np.linalg.norm(p[:, None] - C[None], axis=-1) - R).min(1)
        return d, np.ones(len(p), bool)
    return fn
