"""Distance-field snapshot for consumers outside this process (P1.7.7).

``export_field`` turns a live ``SemanticDistanceQuery`` into the grids carried
by ``msg/SemanticDistanceField.msg`` (published by ``scene_query_node``), and
``FieldSampler`` is the numpy reference of the C++ sampler in
``include/scout_piper_scene_repr/semantic_distance_field.hpp`` that the MoveIt
semantic collision plugin uses. The tests hold the two to each other and to
``SemanticDistanceQuery``.

Differences from ``SemanticDistanceQuery.query`` (by design):
  * hard classes are merged into one grid, min over classes of (d − padding),
    before interpolation. Interpolating the minimum is never larger than the
    minimum of the interpolations, so the snapshot is at least as conservative;
  * voxel age is quantised to 0.1 s and judged against the consumer's own
    ``max_voxel_age_s`` (a planner looks at a larger, older region than the
    reactive MPC does);
  * outside the grid the snapshot says nothing (``in_bounds`` False): the
    grid only covers the plant, the rest of the robot's workspace is left to
    the planning scene;
  * no grasp-mode exclusion (planning to a pre-grasp does not need it).
"""

from __future__ import annotations

import array
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .distance_query import SemanticDistanceQuery

AGE_UNKNOWN = 65535          # age_ds value of a never-observed voxel
AGE_PER_SECOND = 10.0        # age_ds units per second (0.1 s resolution)
INF_SUBSTITUTE = 1e3         # same substitution SemanticDistanceQuery uses for interpolation


@dataclass
class DistanceFieldSnapshot:
    origin: np.ndarray                 # (3,) world coords of voxel (0,0,0) corner
    voxel_size: float
    shape: tuple                       # (nx, ny, nz)
    stamp: float                       # map stamp (s) the ages refer to
    hard_classes: List[str]
    hard_distance: np.ndarray          # (nx, ny, nz) float32, min_c (d_c − padding_c)
    hard_class: np.ndarray             # (nx, ny, nz) uint8 index into hard_classes
    soft_class: str = ""
    soft_max_penetration: float = 0.0
    soft_distance: Optional[np.ndarray] = None   # (nx, ny, nz) float32
    age_ds: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), np.uint16))


def export_field(q: SemanticDistanceQuery) -> Optional[DistanceFieldSnapshot]:
    """Snapshot of ``q`` at the map's last integration; None before the first frame."""
    vmap, spec = q.map, q.map.spec
    if vmap.version == 0:
        return None
    hard = [n for n, p in q.policies.items() if p.behavior == "hard" and n in vmap.hits]
    if hard:
        stack = np.stack([q.class_field(n) - np.float32(q.policies[n].padding_m) for n in hard])
        k = np.argmin(stack, axis=0)
        hard_d = np.take_along_axis(stack, k[None], axis=0)[0].astype(np.float32)
        hard_k = k.astype(np.uint8)
    else:
        hard_d = np.full(spec.shape, np.inf, np.float32)
        hard_k = np.zeros(spec.shape, np.uint8)

    soft = [n for n, p in q.policies.items() if p.behavior == "soft" and n in vmap.hits]
    soft_d, soft_name, soft_pen = None, "", 0.0
    if soft:
        # one soft class in the shipped policy (leaf); several are merged with the strictest cap
        soft_d = np.min(np.stack([q.class_field(n) for n in soft]), axis=0).astype(np.float32)
        soft_name = "+".join(soft)
        soft_pen = float(min(q.policies[n].max_penetration_m for n in soft))

    age = np.full(spec.shape, AGE_UNKNOWN, np.uint16)
    seen = vmap.observed
    a = np.rint((vmap.stamp - vmap.last_seen[seen]) * AGE_PER_SECOND)
    age[seen] = np.clip(a, 0, AGE_UNKNOWN - 1).astype(np.uint16)

    return DistanceFieldSnapshot(
        origin=np.asarray(spec.origin, float).copy(), voxel_size=float(spec.voxel_size),
        shape=tuple(int(s) for s in spec.shape), stamp=float(vmap.stamp),
        hard_classes=hard, hard_distance=hard_d, hard_class=hard_k,
        soft_class=soft_name, soft_max_penetration=soft_pen, soft_distance=soft_d, age_ds=age)


def fill_msg(msg, snap: DistanceFieldSnapshot, frame_id: str):
    """Copy ``snap`` into a ``SemanticDistanceField`` message (duck-typed, so the
    layout is testable without ROS). Arrays are C-order over (i, j, k); numeric
    sequences go in as ``array.array``, which rclpy accepts without a copy loop."""
    msg.header.frame_id = frame_id
    sec = int(np.floor(snap.stamp))
    msg.header.stamp.sec, msg.header.stamp.nanosec = sec, int(round((snap.stamp - sec) * 1e9)) % 1_000_000_000
    msg.origin.x, msg.origin.y, msg.origin.z = (float(v) for v in snap.origin)
    msg.voxel_size = float(snap.voxel_size)
    msg.size = [int(s) for s in snap.shape]
    msg.hard_classes = list(snap.hard_classes)
    msg.hard_distance = array.array("f", snap.hard_distance.astype(np.float32).tobytes())
    msg.hard_class = snap.hard_class.astype(np.uint8).tobytes()
    msg.soft_class = snap.soft_class
    msg.soft_max_penetration = float(snap.soft_max_penetration)
    soft = snap.soft_distance if snap.soft_distance is not None else np.zeros(0, np.float32)
    msg.soft_distance = array.array("f", soft.astype(np.float32).tobytes())
    msg.age_ds = array.array("H", snap.age_ds.astype(np.uint16).tobytes())
    return msg


def snapshot_from_msg(msg) -> DistanceFieldSnapshot:
    """Inverse of ``fill_msg``: a ``SemanticDistanceField`` message (or any
    object with the same fields) back into a snapshot. Consumers outside the
    publishing process (the whole-body MPC) sample it with ``FieldSampler``."""
    shape = tuple(int(v) for v in msg.size)
    n = int(np.prod(shape))

    def arr(seq, dtype):
        a = np.frombuffer(bytes(seq), dtype) if isinstance(seq, (bytes, bytearray)) else np.asarray(seq, dtype)
        return a.reshape(shape) if a.size == n else None

    stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
    soft = arr(msg.soft_distance, np.float32) if len(msg.soft_distance) else None
    return DistanceFieldSnapshot(
        origin=np.array([msg.origin.x, msg.origin.y, msg.origin.z], float), voxel_size=float(msg.voxel_size),
        shape=shape, stamp=float(stamp), hard_classes=list(msg.hard_classes),
        hard_distance=arr(msg.hard_distance, np.float32), hard_class=arr(msg.hard_class, np.uint8),
        soft_class=str(msg.soft_class), soft_max_penetration=float(msg.soft_max_penetration),
        soft_distance=soft, age_ds=arr(msg.age_ds, np.uint16))


def snapshot_distance_fn(snap: DistanceFieldSnapshot, now: float, max_voxel_age_s: float,
                         outside_free: bool = True):
    """DistanceFn (points -> (hard distance, valid)) over a snapshot for the
    whole-body MPC and its safety filter. Inside the grid, unknown or stale
    voxels are invalid with distance clamped to ≤ 0 (unknown is not free).
    Outside the grid (the snapshot covers the plant, not the robot's whole
    workspace) points are valid with +inf when ``outside_free``."""
    sampler = FieldSampler(snap)

    def fn(points):
        q = sampler.query(points, now, max_voxel_age_s)
        d = np.asarray(q["hard"], float)
        valid = q["fresh"] | (~q["in_bounds"] if outside_free else False)
        d = np.where(q["in_bounds"], np.where(valid, d, np.minimum(d, 0.0)),
                     np.inf if outside_free else 0.0)
        return d, valid
    return fn


def snapshot_leaf_fn(snap: DistanceFieldSnapshot, weight: float = 50.0, d_soft: float = 0.02):
    """Soft leaf cost w·(d_soft − d)² from the snapshot's soft class (0 without one)."""
    sampler = FieldSampler(snap)

    def fn(points):
        q = sampler.query(points, snap.stamp, np.inf)
        d = np.where(q["in_bounds"], q["soft"], np.inf)
        return np.where(d < d_soft, weight * (d - d_soft) ** 2, 0.0)
    return fn


# ----------------------------------------------------------------- sampler
SPHERE_STATUS = ("outside", "free", "hard", "soft", "unknown", "stale")


class FieldSampler:
    """Numpy reference of ``SemanticDistanceField`` (C++). Keep the two in step."""

    def __init__(self, snap: DistanceFieldSnapshot):
        self.s = snap
        self.n = np.array(snap.shape)
        self._hard = self._prep(snap.hard_distance)
        self._soft = self._prep(snap.soft_distance) if snap.soft_distance is not None else None

    @staticmethod
    def _prep(grid: np.ndarray):
        g = np.asarray(grid, np.float64)
        if np.isinf(g).all():
            return None                                    # nothing of this kind: +inf everywhere
        return np.where(np.isinf(g), INF_SUBSTITUTE, g)

    def _interp(self, grid, pts: np.ndarray) -> np.ndarray:
        if grid is None:
            return np.full(len(pts), np.inf)
        g = (pts - self.s.origin) / self.s.voxel_size - 0.5            # voxel-centre coordinates
        gc = np.clip(g, 0.0, self.n - 1)
        i0 = np.minimum(np.floor(gc).astype(np.int64), np.maximum(self.n - 2, 0))
        t = gc - i0
        i1 = np.minimum(i0 + 1, self.n - 1)
        out = np.zeros(len(pts))
        for dx in (0, 1):
            wx = t[:, 0] if dx else 1 - t[:, 0]
            ix = i1[:, 0] if dx else i0[:, 0]
            for dy in (0, 1):
                wy = t[:, 1] if dy else 1 - t[:, 1]
                iy = i1[:, 1] if dy else i0[:, 1]
                for dz in (0, 1):
                    wz = t[:, 2] if dz else 1 - t[:, 2]
                    iz = i1[:, 2] if dz else i0[:, 2]
                    out += wx * wy * wz * grid[ix, iy, iz]
        return out

    def query(self, points, now: float, max_voxel_age_s: float) -> Dict[str, np.ndarray]:
        pts = np.asarray(points, float).reshape(-1, 3)
        idx = np.floor((pts - self.s.origin) / self.s.voxel_size).astype(np.int64)
        inb = np.all((idx >= 0) & (idx < self.n), axis=1)
        idc = np.clip(idx, 0, self.n - 1)
        age = self.s.age_ds[tuple(idc.T)]
        known = inb & (age != AGE_UNKNOWN)
        fresh = known & (age / AGE_PER_SECOND + (now - self.s.stamp) <= max_voxel_age_s)
        hk = self.s.hard_class[tuple(idc.T)].astype(np.int64)
        return {"in_bounds": inb, "known": known, "fresh": fresh,
                "hard": self._interp(self._hard, pts),
                "hard_class": np.where(self._hard is not None, hk, -1),
                "soft": self._interp(self._soft, pts)}

    def check_spheres(self, centers, radii, now: float, max_voxel_age_s: float = 30.0,
                      unknown_is_occupied: bool = True, check_soft: bool = True):
        """Per sphere: (status name, clearance m, collision bool). Mirrors
        ``SemanticDistanceField::checkSphere``."""
        q = self.query(centers, now, max_voxel_age_s)
        r = np.asarray(radii, float).reshape(-1)
        out = []
        for j in range(len(r)):
            if not q["in_bounds"][j]:
                out.append(("outside", np.inf, False))
                continue
            hard, soft = q["hard"][j], q["soft"][j]
            invalid = not q["fresh"][j]
            if invalid and unknown_is_occupied:
                status = "unknown" if not q["known"][j] else "stale"
                out.append((status, min(hard, 0.0) - r[j], True))
                continue
            clear_hard = hard - r[j]
            clear_soft = soft - r[j] + self.s.soft_max_penetration if check_soft else np.inf
            if clear_hard < 0:
                out.append(("hard", min(clear_hard, clear_soft), True))
            elif clear_soft < 0:
                out.append(("soft", clear_soft, True))
            else:
                out.append(("free", min(clear_hard, clear_soft), False))
        return out
