"""Geometry sources for the MPC: semantic-scene query or analytic fields."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def semantic_distance_fn(query, now: Optional[float] = None, outside_free: bool = True, **kw):
    """Wrap ``SemanticDistanceQuery`` (scout_piper_scene_repr) as a DistanceFn.

    Inside the map's grid the query clamps unknown/stale space to ≤ 0 and
    reports ``valid``; both are passed through, so the MPC and the safety
    filter treat unknown geometry conservatively. The grid covers the plant,
    not the robot's whole workspace: with ``outside_free`` points outside it
    (the base, the arm behind the camera) are valid with +inf distance, as in
    the MoveIt plugin. Without it every out-of-grid sphere counted as unknown
    and the safety filter would never let the base move."""
    spec = query.map.spec

    def fn(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = np.asarray(points, float).reshape(-1, 3)
        r = query.query(p, now=now, gradient=False, **kw)   # MPPI needs no gradient
        d, valid = r.hard_distance, r.valid
        if outside_free:
            out = ~spec.in_bounds(spec.world_to_index(p))
            d = np.where(out, np.inf, d)
            valid = valid | out
        return d, valid
    return fn


class GraspFieldCache:
    """Semantic distance field for a controller that moves onto its target
    (the stem_grasp MPPI servo): the target object excluded (``exclude_target``,
    grasp mode: the obstacle voxels connected to the grasp point within
    ``radius_m``), sampled in the robot base frame.

    ``distance_fn(snap, T_field_base, target_base, now)`` returns a
    DistanceFn over base-frame points: they are mapped into the snapshot's
    frame with ``T_field_base`` (4×4, base → field) and sampled with the
    snapshot conventions (unknown/stale space invalid and ≤ 0, outside the grid
    free). The exclusion (a local distance transform, a few ms) is redone only
    for a new snapshot or when the target moves more than ``retarget_m``.
    ``radius_m`` bounds how much of the target stem is released: the gripper's
    sphere (0.03 m) plus d_safe (0.02) must clear what remains wherever the
    gripper passes, which for an approach inclined to the stem takes more than
    that sum (0.10 m holds at 35°); other objects in the ball stay hard unless
    they touch the target inside it."""

    def __init__(self, radius_m: float = 0.10, max_voxel_age_s: float = 30.0, retarget_m: float = 0.01):
        self.radius, self.max_age, self.retarget = float(radius_m), float(max_voxel_age_s), float(retarget_m)
        self._snap = None
        self._target = None
        self._sampler = None
        self.excluded = None

    def distance_fn(self, snap, T_field_base: np.ndarray, target_base: np.ndarray, now: float):
        from scout_piper_scene_repr_py.field import (FieldSampler, exclude_target,  # noqa: PLC0415
                                                     snapshot_distance_fn)
        T = np.asarray(T_field_base, float)
        target = T[:3, :3] @ np.asarray(target_base, float) + T[:3, 3]
        if (snap is not self._snap or self._target is None
                or np.linalg.norm(target - self._target) > self.retarget):
            self.excluded = exclude_target(snap, target, self.radius)
            self._sampler = FieldSampler(self.excluded)
            self._snap, self._target = snap, target
        fn = snapshot_distance_fn(self.excluded, now, self.max_age, sampler=self._sampler)
        R, t = T[:3, :3], T[:3, 3]

        def in_base(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return fn(np.asarray(points, float).reshape(-1, 3) @ R.T + t)
        return in_base


def semantic_leaf_fn(query, d_soft: float = 0.02):
    def fn(points: np.ndarray) -> np.ndarray:
        cost, _ = query.leaf_cost(points, d_soft=d_soft)
        return cost
    return fn


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points (N, 3) to segments a→b (M, 3) each -> (N, M).

    One pass per segment over contiguous (N, 3) arrays: several times faster
    than one (N, M, 3) broadcast for the few dozen capsules of a plant scene."""
    p = np.asarray(points, float).reshape(-1, 3)
    a, b = np.asarray(a, float).reshape(-1, 3), np.asarray(b, float).reshape(-1, 3)
    out = np.empty((len(p), len(a)))
    for j in range(len(a)):
        ab = b[j] - a[j]
        v = p - a[j]
        t = np.clip(v @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        v -= t[:, None] * ab
        out[:, j] = np.sqrt(np.einsum("ij,ij->i", v, v))
    return out


def capsules_distance_fn(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]],
                         radii: Sequence[float], spheres: Optional[Tuple] = None):
    """Analytic field: union of capsules (segment a→b with radius; stems,
    branches, stakes) and optional spheres ``(centers, radii)`` (pots).
    Everything is observed/valid. Synthetic plant scenes (P3.3)."""
    A, B, R = np.asarray(a, float).reshape(-1, 3), np.asarray(b, float).reshape(-1, 3), np.asarray(radii, float)
    C, Rs = (np.asarray(spheres[0], float).reshape(-1, 3), np.asarray(spheres[1], float)) if spheres else (None, None)

    def fn(points: np.ndarray):
        p = np.asarray(points, float).reshape(-1, 3)
        d = np.full(len(p), np.inf)
        if len(A):
            d = (segment_distance(p, A, B) - R).min(1)
        if C is not None and len(C):
            d = np.minimum(d, (np.linalg.norm(p[:, None] - C[None], axis=-1) - Rs).min(1))
        return d, np.ones(len(p), bool)
    return fn


def disc_distance(points: np.ndarray, centers: np.ndarray, normals: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Distance from points (N, 3) to flat discs (leaves) -> (N, M)."""
    p = np.asarray(points, float).reshape(-1, 3)
    c, r = np.asarray(centers, float).reshape(-1, 3), np.asarray(radii, float)
    n = np.asarray(normals, float).reshape(-1, 3)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    out = np.empty((len(p), len(c)))
    for j in range(len(c)):
        d = p - c[j]
        h = d @ n[j]                                             # height above the disc plane
        d -= h[:, None] * n[j]
        radial = np.sqrt(np.einsum("ij,ij->i", d, d))
        out[:, j] = np.hypot(h, np.clip(radial - r[j], 0, None))
    return out


def discs_leaf_fn(centers, normals, radii, weight: float = 50.0, d_soft: float = 0.02):
    """Soft leaf cost w·(d_soft − d)² below d_soft, d = distance to the nearest
    leaf disc: the same shape as ``SemanticDistanceQuery.leaf_cost``."""
    C, N, R = np.asarray(centers, float), np.asarray(normals, float), np.asarray(radii, float)

    def fn(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, float).reshape(-1, 3)
        if not len(C):
            return np.zeros(len(p))
        d = disc_distance(p, C, N, R).min(1)
        return np.where(d < d_soft, weight * (d - d_soft) ** 2, 0.0)
    return fn


class GridFn:
    """A point function sampled on a voxel grid; calls interpolate trilinearly.

    ``values=False`` wraps a DistanceFn (calls return (d, valid)); ``True`` a
    cost function such as a LeafFn (calls return one array). Points outside
    [lo, top] fall back to the original function. ``grid``, ``lo`` and
    ``voxel_size`` are public so the torch backend can put the same grid on
    the GPU (``torch_backend.TorchGridField.from_grid_fn``)."""

    def __init__(self, fn, lo: Sequence[float], hi: Sequence[float], voxel_size: float = 0.01,
                 values: bool = False):
        self.fn, self.values, self.voxel_size = fn, values, float(voxel_size)
        self.lo = np.asarray(lo, float)
        n = np.ceil((np.asarray(hi, float) - self.lo) / voxel_size).astype(int) + 1
        axes = [self.lo[i] + voxel_size * np.arange(n[i]) for i in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        vals = []
        for k in range(0, len(G), 200_000):
            r = fn(G[k:k + 200_000])
            vals.append(r if values else r[0])
        self.grid = np.concatenate(vals).reshape(n).astype(np.float32)
        self.top = self.lo + voxel_size * (n - 1)

    def sample(self, points: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates  # noqa: PLC0415

        p = np.asarray(points, float).reshape(-1, 3)
        inside = np.all((p >= self.lo) & (p <= self.top), axis=1)
        out = np.empty(len(p))
        if inside.any():
            out[inside] = map_coordinates(self.grid, ((p[inside] - self.lo) / self.voxel_size).T,
                                          order=1, mode="nearest")
        if (~inside).any():
            r = self.fn(p[~inside])
            out[~inside] = r if self.values else r[0]
        return out

    def __call__(self, points: np.ndarray):
        d = self.sample(points)
        return d if self.values else (d, np.ones(len(d), bool))


def gridded(fn, lo: Sequence[float], hi: Sequence[float], voxel_size: float = 0.01,
            values: bool = False) -> GridFn:
    """Sample ``fn`` on a voxel grid once; later calls interpolate (what the
    semantic scene's voxel field does too). The synthetic benchmarks use it so
    that MPPI's tens of thousands of queries per step cost milliseconds
    instead of exact capsule/disc distances."""
    return GridFn(fn, lo, hi, voxel_size, values)


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
