"""Planner-facing semantic distance query (research/semantic_scene C4).

    q = SemanticDistanceQuery(voxel_map, policies)
    r = q.query(points, now=t)            # batched, in the map's world frame
    r.hard_distance    min signed distance over hard classes (m), padding applied
    r.hard_class       which hard class is nearest
    r.hard_gradient    ∇ of the hard field (unit-ish, points away from obstacles)
    r.valid            False where the point is unknown, out of bounds or stale
    r.class_distance   per-class signed distance

Conventions
  * Signed distance: > 0 outside, < 0 inside occupied voxels. Interpolation is
    trilinear over voxel centres, so errors are O(voxel size).
  * Unknown voxels never report free space: ``valid`` is False and the hard
    distance is clamped to ≤ 0 there, so a planner that ignores ``valid``
    still sees a conservative value.
  * Grasp mode excludes hard-class voxels within ``exclusion_radius_m`` of the
    selected target point (the target peduncle itself); neighbours stay hard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from .policy import DEFAULT_POLICIES, ClassPolicy
from .voxel_map import SemanticVoxelMap


@dataclass
class QueryResult:
    hard_distance: np.ndarray            # (N,)
    hard_class: np.ndarray               # (N,) object array of class names
    hard_gradient: np.ndarray            # (N, 3)
    valid: np.ndarray                    # (N,) bool
    class_distance: Dict[str, np.ndarray]
    map_age_s: float


def signed_distance_field(occ: np.ndarray, voxel_size: float) -> np.ndarray:
    """Signed EDT on voxel centres: + outside, − inside (metres).

    With no occupied voxel at all the field is +inf (nothing to hit)."""
    if not occ.any():
        return np.full(occ.shape, np.inf, np.float32)
    out = distance_transform_edt(~occ)
    inside = distance_transform_edt(occ)
    # half-voxel offset: the surface lies between an occupied and a free centre
    sdf = np.where(occ, -(inside - 0.5), out - 0.5) * voxel_size
    return sdf.astype(np.float32)


class SemanticDistanceQuery:
    def __init__(self, vmap: SemanticVoxelMap,
                 policies: Optional[Dict[str, ClassPolicy]] = None,
                 min_hits: int = 2, max_age_s: float = 1.0):
        self.map = vmap
        self.policies = policies or DEFAULT_POLICIES
        self.min_hits = min_hits
        self.max_age_s = max_age_s
        self._cache: Dict[tuple, np.ndarray] = {}
        self._cache_version = -1

    # ------------------------------------------------------------ fields
    def class_field(self, name: str) -> np.ndarray:
        """Signed distance grid of one class on voxel centres (cached; +inf if empty)."""
        return self._field(name)

    def _field(self, name: str, exclude: Optional[tuple] = None) -> np.ndarray:
        if self._cache_version != self.map.version:
            self._cache.clear()
            self._cache_version = self.map.version
        key = (name, exclude)
        if key not in self._cache:
            occ = self.map.occupied(name, self.min_hits)
            if exclude is not None:
                c, r = np.asarray(exclude[:3]), exclude[3]
                idx = np.argwhere(occ)
                if len(idx):
                    far = np.linalg.norm(self.map.spec.index_to_world(idx) - c, axis=1) > r
                    occ = np.zeros_like(occ)
                    occ[tuple(idx[far].T)] = True
            self._cache[key] = signed_distance_field(occ, self.map.spec.voxel_size)
        return self._cache[key]

    def _sample(self, field: np.ndarray, pts: np.ndarray) -> np.ndarray:
        spec = self.map.spec
        g = (pts - spec.origin) / spec.voxel_size - 0.5          # voxel-centre coords
        if np.isinf(field).all():
            return np.full(len(pts), np.inf)
        f = np.where(np.isinf(field), 1e3, field)
        return map_coordinates(f, g.T, order=1, mode="nearest")

    # ------------------------------------------------------------- query
    def query(self, points: np.ndarray, now: Optional[float] = None,
              mode: str = "approach", target_point: Optional[np.ndarray] = None,
              exclusion_radius_m: float = 0.02, grad_eps: Optional[float] = None,
              gradient: bool = True) -> QueryResult:
        pts = np.asarray(points, float).reshape(-1, 3)
        spec = self.map.spec
        now = self.map.stamp if now is None else now
        exclude = None
        if mode == "grasp":
            if target_point is None:
                raise ValueError("grasp mode needs target_point")
            exclude = tuple(np.round(np.asarray(target_point, float), 4)) + (float(exclusion_radius_m),)

        idx = spec.world_to_index(pts)
        inb = spec.in_bounds(idx)
        idc = np.clip(idx, 0, np.array(spec.shape) - 1)
        known = self.map.observed[tuple(idc.T)] & inb
        fresh = (now - self.map.last_seen[tuple(idc.T)]) <= self.max_age_s
        valid = known & fresh

        hard = [n for n, p in self.policies.items() if p.behavior == "hard" and n in self.map.hits]
        class_d = {}
        for n in self.map.hits:
            f = self._field(n, exclude if n in hard else None)
            class_d[n] = self._sample(f, pts)

        if hard:
            stack = np.stack([class_d[n] - self.policies[n].padding_m for n in hard])
            k = np.argmin(stack, axis=0)
            hd = stack[k, np.arange(len(pts))]
            hc = np.array(hard, dtype=object)[k]
            eps = grad_eps or spec.voxel_size
            grad = np.zeros((len(pts), 3))
            for a in range(3 if gradient else 0):    # 6 extra field samples per point
                d = np.zeros(3); d[a] = eps
                plus = np.min(np.stack([self._sample(self._field(n, exclude), pts + d) - self.policies[n].padding_m for n in hard]), 0)
                minus = np.min(np.stack([self._sample(self._field(n, exclude), pts - d) - self.policies[n].padding_m for n in hard]), 0)
                grad[:, a] = (plus - minus) / (2 * eps)
        else:
            hd = np.full(len(pts), np.inf)
            hc = np.array([None] * len(pts), dtype=object)
            grad = np.zeros((len(pts), 3))

        hd = np.where(valid, hd, np.minimum(hd, 0.0))             # unknown ≠ free
        return QueryResult(hd, hc, grad, valid, class_d, float(now - self.map.stamp))

    # ------------------------------------------------------------- costs
    def sphere_clearance(self, centers: np.ndarray, radii: np.ndarray, **kw) -> QueryResult:
        """d_j = φ_hard(p_j) − r_j for robot collision spheres."""
        r = self.query(centers, **kw)
        r.hard_distance = r.hard_distance - np.asarray(radii, float)
        return r

    def leaf_cost(self, points: np.ndarray, d_soft: float = 0.02):
        """ψ(d) = w·(d − d_soft)² for d < d_soft; penetration deeper than the
        policy's ``max_penetration_m`` is reported as a hard violation.
        Returns (cost (N,), hard_violation (N,) bool)."""
        pol = self.policies["leaf"]
        d = self._sample(self._field("leaf"), np.asarray(points, float).reshape(-1, 3))
        cost = np.where(d < d_soft, pol.cost_weight * (d - d_soft) ** 2, 0.0)
        return cost, d < -pol.max_penetration_m

    def target_attraction(self, points: np.ndarray) -> np.ndarray:
        """Negative cost within the attractor radius of the target field."""
        pol = self.policies["target"]
        d = self._sample(self._field("target"), np.asarray(points, float).reshape(-1, 3))
        inside = d < pol.attract_radius_m
        return np.where(inside, pol.attract_weight * (1 - np.clip(d, 0, None) / pol.attract_radius_m), 0.0)
