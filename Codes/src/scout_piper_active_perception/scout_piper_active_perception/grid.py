"""Occupancy view of the semantic scene for view planning.

``OccupancyGrid`` holds what the planner needs per voxel: occupied (any class,
leaves included: a leaf hides what is behind it) and unknown (never
observed). It is built from the CPU ``SemanticVoxelMap`` (simulation, offline)
or from a ``DistanceFieldSnapshot`` (the ``/scene_repr/distance_field`` topic
on the robot: occupied where the hard or soft distance is below half a voxel,
unknown where ``age_ds`` says never observed).

Rays are marched one voxel per step. ``transmittance`` is the probability that
a ray reaches its end: an occupied voxel blocks it, an unknown voxel lets it
through with ``1 - p_unknown``. ``p_unknown`` is the canopy's fill fraction
(default 0.02); 0.15 or 0.5 made every line of sight through the 20-30 unknown
voxels left after one view look blocked, and the predicted gains ran opposite
to the rendered ones. (``occupancy_prior``, the occupied share of the observed
voxels, overestimates it: one view observes mostly surfaces.)

``voxel_gain`` scores a set of voxels (the approach corridor's unknown ones)
for a camera: a voxel is cleared only when a ray reaches it (transmittance
before it) *and* returns depth from behind it within range (a depth camera
looking past the plant into empty space returns nothing and clears nothing). ``entropy_gain`` is the expected number of unknown voxels a camera
would observe (each carrying H(p_unknown) bits, a constant left out),
weighted per voxel by a task weight, each voxel counted once with its best
ray.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class OccupancyGrid:
    origin: np.ndarray            # (3,) corner of voxel (0, 0, 0)
    voxel_size: float
    occupied: np.ndarray          # (nx, ny, nz) bool
    unknown: np.ndarray           # (nx, ny, nz) bool

    @property
    def shape(self):
        return self.occupied.shape

    @classmethod
    def from_voxel_map(cls, vmap, min_hits: int = 2) -> "OccupancyGrid":
        occ = np.zeros(vmap.spec.shape, bool)
        for name in vmap.hits:
            occ |= vmap.occupied(name, min_hits)
        return cls(origin=np.asarray(vmap.spec.origin, float), voxel_size=float(vmap.spec.voxel_size),
                   occupied=occ, unknown=~vmap.observed)

    @classmethod
    def from_snapshot(cls, snap, block_m: Optional[float] = None) -> "OccupancyGrid":
        from scout_piper_scene_repr_py.field import AGE_UNKNOWN  # noqa: PLC0415
        block = 0.5 * snap.voxel_size if block_m is None else block_m
        occ = np.asarray(snap.hard_distance) < block
        if snap.soft_distance is not None:
            occ |= np.asarray(snap.soft_distance) < block
        return cls(origin=np.asarray(snap.origin, float), voxel_size=float(snap.voxel_size),
                   occupied=occ, unknown=np.asarray(snap.age_ds) == AGE_UNKNOWN)

    def occupancy_prior(self, lo: float = 0.01, hi: float = 0.2) -> float:
        """Occupied share of the observed voxels: the prior occupancy of an
        unknown one."""
        obs = ~self.unknown
        n = int(obs.sum())
        return float(np.clip((self.occupied & obs).sum() / n, lo, hi)) if n else hi

    def centers(self) -> np.ndarray:
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij"), -1)
        return self.origin + (idx + 0.5) * self.voxel_size

    def index(self, pts: np.ndarray):
        """Voxel indices (..., 3) and an in-bounds mask."""
        idx = np.floor((np.asarray(pts) - self.origin) / self.voxel_size).astype(np.int64)
        inb = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=-1)
        return idx, inb

    def _samples(self, origins: np.ndarray, ends: np.ndarray, stop_short_m: float = 0.0):
        """Points every voxel along origin -> end (stopping ``stop_short_m``
        before the end): (..., S, 3) and a valid mask (..., S)."""
        d = ends - origins
        L = np.linalg.norm(d, axis=-1)
        n = int(np.ceil(max(float(L.max(initial=0.0)), 1e-9) / self.voxel_size)) + 1
        s = (np.arange(n) + 0.5) * self.voxel_size
        u = d / np.maximum(L, 1e-12)[..., None]
        pts = origins[..., None, :] + s[:, None] * u[..., None, :]
        return pts, s < (L - stop_short_m)[..., None]

    def voxel_gain(self, T_world_cam: np.ndarray, K: np.ndarray, image_hw, voxels: np.ndarray,
                   p_unknown: float = 0.02, near_m: float = 0.07, far_m: float = 1.0,
                   p_return_outside: float = 0.2, ignore: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> float:
        """Expected number of ``voxels`` (bool grid) a camera clears: Σ in-view ·
        P(reach) · P(return behind, within ``far_m``); leaving the grid before
        ``far_m`` returns with ``p_return_outside``."""
        idx = np.argwhere(voxels)
        if not len(idx):
            return 0.0
        P = self.origin + (idx + 0.5) * self.voxel_size
        c = T_world_cam[:3, 3]
        pc = (P - c) @ T_world_cam[:3, :3]
        z = pc[:, 2]
        H, W = image_hw
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        ok = (z > near_m) & (z < far_m) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            return 0.0
        P = P[ok]
        d = P - c
        L = np.linalg.norm(d, axis=1)
        dirs = d / L[:, None]
        rng = far_m * L / z[ok]                                    # ray length to depth far_m
        pts, valid = self._samples(np.broadcast_to(c, P.shape), c + rng[:, None] * dirs)
        s_along = (np.arange(pts.shape[1]) + 0.5) * self.voxel_size
        before = valid & (s_along[None] < L[:, None] - 0.5 * self.voxel_size)
        after = valid & (s_along[None] > L[:, None] + 0.5 * self.voxel_size)
        gi, inb = self.index(pts)
        gc = np.clip(gi, 0, np.array(self.shape) - 1)
        occ = self.occupied[gc[..., 0], gc[..., 1], gc[..., 2]] & inb
        unk = self.unknown[gc[..., 0], gc[..., 1], gc[..., 2]] & inb & ~occ
        if ignore is not None:
            ig = ignore(pts.reshape(-1, 3)).reshape(pts.shape[:-1])
            occ_b, unk_b = occ & ~ig, unk & ~ig
        else:
            occ_b, unk_b = occ, unk
        reach = np.where((occ_b & before).any(1), 0.0, (1.0 - p_unknown) ** (unk_b & before).sum(1))
        p_hit = np.where(occ, 1.0, np.where(unk, p_unknown, 0.0)) * after
        miss = np.prod(1.0 - p_hit, axis=1)
        left = (after & ~inb).any(1)                               # the ray leaves the grid within range
        ret = 1.0 - miss * np.where(left, 1.0 - p_return_outside, 1.0)
        return float((reach * ret).sum())

    def transmittance(self, origins: np.ndarray, ends: np.ndarray, p_unknown: Optional[float] = None,
                      stop_short_m: float = 0.0, ignore: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """Probability that each ray origin -> end is unobstructed (outside
        the grid counts as free). ``ignore(points) -> bool`` exempts samples
        (the target's own voxels)."""
        p_unknown = 0.02 if p_unknown is None else p_unknown
        origins, ends = np.broadcast_arrays(np.asarray(origins, float), np.asarray(ends, float))
        pts, valid = self._samples(origins, ends, stop_short_m)
        idx, inb = self.index(pts)
        idc = np.clip(idx, 0, np.array(self.shape) - 1)
        occ = self.occupied[idc[..., 0], idc[..., 1], idc[..., 2]] & inb & valid
        unk = self.unknown[idc[..., 0], idc[..., 1], idc[..., 2]] & inb & valid & ~occ
        if ignore is not None:
            ig = ignore(pts.reshape(-1, 3)).reshape(pts.shape[:-1])
            occ &= ~ig
            unk &= ~ig
        blocked = occ.any(-1)
        return np.where(blocked, 0.0, (1.0 - p_unknown) ** unk.sum(-1))

    def entropy_gain(self, T_world_cam: np.ndarray, K: np.ndarray, image_hw, weight: Optional[np.ndarray] = None,
                     rays_hw=(16, 28), max_range_m: float = 0.6, p_unknown: Optional[float] = None) -> float:
        """Expected task-weighted bits of unknown space a camera at
        ``T_world_cam`` observes (``weight`` (nx, ny, nz), default 1)."""
        p_unknown = 0.02 if p_unknown is None else p_unknown
        H, W = image_hw
        v = (np.arange(rays_hw[0]) + 0.5) * H / rays_hw[0]
        u = (np.arange(rays_hw[1]) + 0.5) * W / rays_hw[1]
        uu, vv = np.meshgrid(u, v)
        rc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).reshape(-1, 3)
        rc /= np.linalg.norm(rc, axis=1, keepdims=True)
        dirs = rc @ T_world_cam[:3, :3].T
        o = T_world_cam[:3, 3]
        pts, valid = self._samples(np.broadcast_to(o, dirs.shape), o + max_range_m * dirs)
        idx, inb = self.index(pts)
        inb &= valid
        idc = np.clip(idx, 0, np.array(self.shape) - 1)
        occ = self.occupied[idc[..., 0], idc[..., 1], idc[..., 2]] & inb
        unk = self.unknown[idc[..., 0], idc[..., 1], idc[..., 2]] & inb & ~occ
        # transmittance up to (not including) each sample
        blocked_before = np.cumsum(occ, axis=1) - occ > 0
        n_unk_before = np.cumsum(unk, axis=1) - unk
        T = np.where(blocked_before, 0.0, (1.0 - p_unknown) ** n_unk_before)
        sel = unk & (T > 1e-3)
        if not sel.any():
            return 0.0
        flat = np.ravel_multi_index(tuple(idc[sel].T), self.shape)
        best = np.zeros(int(np.prod(self.shape)))
        np.maximum.at(best, flat, T[sel])
        w = 1.0 if weight is None else weight.reshape(-1)
        return float((best * w).sum())

    def task_weight(self, target: np.ndarray, sigma_m: float = 0.06, corridor=None,
                    corridor_radius_m: float = 0.04) -> np.ndarray:
        """Gaussian around the target plus 1 inside the approach corridor
        (segment ``corridor`` = (start, end))."""
        C = self.centers()
        w = np.exp(-((C - target) ** 2).sum(-1) / (2 * sigma_m ** 2))
        if corridor is not None:
            w = np.maximum(w, (segment_distance(C, *corridor) < corridor_radius_m).astype(float))
        return w

    def skin(self) -> np.ndarray:
        """Voxels in the 26-neighbourhood of an occupied voxel."""
        if not self.occupied.any():
            return np.zeros(self.shape, bool)
        from scipy.ndimage import binary_dilation  # noqa: PLC0415
        return binary_dilation(self.occupied, structure=np.ones((3, 3, 3), bool))

    def unknown_fraction(self, mask: np.ndarray, skip_skin: bool = True) -> float:
        """Unknown share of ``mask``. With ``skip_skin`` the unknown voxels
        next to an occupied one (26-neighbourhood) do not count: depth rays
        stop one voxel short of a surface, so a layer in front of every surface
        stays unknown from frontal views, and planners keep their clearance
        from surfaces anyway."""
        n = int(mask.sum())
        if not n:
            return 0.0
        unk = self.unknown & mask
        if skip_skin:
            unk &= ~self.skin()
        return float(unk.sum() / n)


def segment_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[..., None] * ab), axis=-1)


def corridor_mask(grid: OccupancyGrid, start: np.ndarray, end: np.ndarray, radius_m: float = 0.04,
                  exclude: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """Voxels within ``radius_m`` of the approach segment (minus ``exclude``)."""
    C = grid.centers()
    m = segment_distance(C, start, end) < radius_m
    if exclude is not None:
        m &= ~exclude(C.reshape(-1, 3)).reshape(m.shape)
    return m
