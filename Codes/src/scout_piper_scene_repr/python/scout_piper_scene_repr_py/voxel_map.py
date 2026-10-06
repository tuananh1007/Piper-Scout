"""CPU semantic voxel map: per-class occupancy evidence + shared free space.

v0 backend for the planner query (research/semantic_scene C4). It fuses the
mask-gated depth images that ``class_demux_node`` publishes, so it needs no
nvblox build and serves as the G1–G3 comparator. Per-class voxel sizes and
GPU TSDF integration stay nvblox's job; this map uses one resolution.

Per depth frame (pixels subsampled by ``stride``):
  * the end point of each ray is a *hit* for the pixel's class;
  * points along the ray before the end point are *free*: they mark the voxel
    observed and decrement hit counts (so moved leaves get cleared);
  * voxels never touched stay **unknown**, which queries report as invalid —
    unknown is not free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class VoxelGridSpec:
    origin: np.ndarray            # (3,) world coords of voxel (0,0,0) corner
    shape: Tuple[int, int, int]
    voxel_size: float

    def world_to_index(self, p: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(p) - self.origin) / self.voxel_size).astype(np.int64)

    def in_bounds(self, idx: np.ndarray) -> np.ndarray:
        return np.all((idx >= 0) & (idx < np.array(self.shape)), axis=-1)

    def index_to_world(self, idx: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(idx) + 0.5) * self.voxel_size


@dataclass
class SemanticVoxelMap:
    spec: VoxelGridSpec
    # 'other' collects valid depth outside every class mask (pots, walls,
    # supports, the robot itself if not filtered): without it the planner
    # would be blind to everything that is not plant.
    classes: Sequence[str] = ("stem", "branch", "leaf", "target", "other")
    max_hits: int = 20
    min_range_m: float = 0.05
    max_range_m: float = 1.0

    def __post_init__(self) -> None:
        S = self.spec.shape
        self.hits: Dict[str, np.ndarray] = {c: np.zeros(S, np.uint8) for c in self.classes}
        self.observed = np.zeros(S, bool)
        self.last_seen = np.full(S, -np.inf, np.float32)
        self.stamp = -np.inf
        self.version = 0
        self._scratch = np.zeros(S, bool)      # dedup buffer, avoids sorting

    # ------------------------------------------------------------- update
    def integrate(
        self,
        depth: np.ndarray,
        class_masks: Dict[str, np.ndarray],
        K: np.ndarray,
        T_world_cam: np.ndarray,
        stamp: float,
        stride: int = 4,
    ) -> None:
        """depth in metres (H, W), NaN/0 = invalid; masks bool (H, W)."""
        H, W = depth.shape
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        z = depth[vv, uu].astype(np.float64)
        ok = np.isfinite(z) & (z > self.min_range_m) & (z < self.max_range_m)
        vv, uu, z = vv[ok], uu[ok], z[ok]
        if not len(z):
            return
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        rays = np.column_stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(z)])
        R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
        vs = self.spec.voxel_size

        # free space: sample each ray every voxel up to one voxel short of the hit
        n_steps = int(np.ceil(z.max() / vs))
        s = (np.arange(n_steps) + 0.5) * vs                         # (S,)
        free = s[None, :] < (z[:, None] - vs)                        # (N, S)
        free &= s[None, :] > self.min_range_m
        ri, si = np.nonzero(free)
        pts_c = rays[ri] * s[si, None]

        # hits per class are collected first: within one frame a hit wins over
        # free space, otherwise rays grazing a thin stem would erase it
        end_w = (rays * z[:, None]) @ R.T + t
        class_masks = dict(class_masks)
        if "other" in self.hits and "other" not in class_masks:
            union = np.zeros(depth.shape, bool)
            for m in class_masks.values():
                union |= np.asarray(m).astype(bool)
            class_masks["other"] = ~union
        hit_sets = {}
        for name, mask in class_masks.items():
            if name not in self.hits:
                continue
            sel = np.asarray(mask)[vv, uu].astype(bool)
            if sel.any():
                hit_sets[name] = self._flat(end_w[sel])
        all_hits = np.unique(np.concatenate(list(hit_sets.values()))) if hit_sets else np.zeros(0, np.int64)
        self._mark_free(pts_c @ R.T + t, stamp, protect=all_hits)
        for name, f in hit_sets.items():
            self._mark_hit_flat(name, f, stamp)
        self.stamp = stamp
        self.version += 1

    def _flat(self, pts: np.ndarray, protect: Optional[np.ndarray] = None) -> np.ndarray:
        """Unique flat voxel indices of in-bounds points (minus ``protect``)."""
        idx = self.spec.world_to_index(pts)
        idx = idx[self.spec.in_bounds(idx)]
        if not len(idx):
            return np.zeros(0, np.int64)
        f = np.ravel_multi_index(idx.T, self.spec.shape)
        sc = self._scratch.reshape(-1)
        sc[f] = True
        if protect is not None and len(protect):
            sc[protect] = False
        out = np.flatnonzero(sc) if len(f) > sc.size // 64 else np.unique(f[sc[f]])
        sc[f] = False
        return out

    def _mark_free(self, pts: np.ndarray, stamp: float, protect: Optional[np.ndarray] = None) -> None:
        f = self._flat(pts, protect)
        if not len(f):
            return
        self.observed.flat[f] = True
        self.last_seen.flat[f] = stamp
        for h in self.hits.values():
            hv = h.flat[f]
            h.flat[f] = np.where(hv > 0, hv - 1, 0)

    def _mark_hit_flat(self, name: str, f: np.ndarray, stamp: float) -> None:
        if not len(f):
            return
        self.observed.flat[f] = True
        self.last_seen.flat[f] = stamp
        h = self.hits[name]
        h.flat[f] = np.minimum(h.flat[f].astype(np.int32) + 2, self.max_hits)

    # ------------------------------------------------------------ queries
    def occupied(self, name: str, min_hits: int = 2) -> np.ndarray:
        return self.hits[name] >= min_hits

    def occupied_points(self, name: str, min_hits: int = 2) -> np.ndarray:
        return self.spec.index_to_world(np.argwhere(self.occupied(name, min_hits)))


def grid_around(center: Sequence[float], half_extent_m: float, voxel_size: float) -> VoxelGridSpec:
    n = int(np.ceil(2 * half_extent_m / voxel_size))
    origin = np.asarray(center, float) - half_extent_m
    return VoxelGridSpec(origin=origin, shape=(n, n, n), voxel_size=voxel_size)

