"""Per-class nvblox ESDFs → one ``DistanceFieldSnapshot`` (P1.3.1, P3.2.2).

``nvblox_semantic.launch.py`` runs one nvblox_node per class. Each answers
``~/get_esdf_and_gradient`` (nvblox_msgs/srv/EsdfAndGradients, needs
``esdf_mode: 3d``) with a dense signed-distance grid over an AABB:
``esdf_and_gradients`` (std_msgs/Float32MultiArray, dims x, y, z, z fastest),
``origin_m`` (corner of the minimal voxel) and ``voxel_size_m``; unobserved
voxels hold ``esdf_and_gradients_unobserved_value`` (nvblox default −1000).

``merge_class_grids`` resamples every class onto one output grid (trilinear at
the output voxel centres; a sample touching an unobserved voxel is unknown),
applies the class policies exactly like ``field.export_field`` (hard classes:
min over (d − padding); the soft class: leaf distance) and marks a voxel
known when any class mapper observed it. The result is the same message the
CPU map publishes, so the MoveIt plugin and the MPC consume either.

Resampling a thin stem (3 mm voxels) onto a coarser output grid can
overestimate the clearance between output samples by up to half the output
voxel diagonal; ``conservative`` subtracts that bound from hard distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .field import AGE_UNKNOWN, DistanceFieldSnapshot
from .policy import ClassPolicy

UNOBSERVED_THRESHOLD = -999.0       # values at or below: nvblox's unobserved sentinel (−1000)


@dataclass
class ClassGrid:
    distance: np.ndarray            # (nx, ny, nz) metres; NaN = unobserved
    origin: np.ndarray              # (3,) corner of voxel (0, 0, 0)
    voxel_size: float


def grid_from_response(data: Sequence[float], dims: Sequence[int], origin: Sequence[float],
                       voxel_size: float, unobserved_threshold: float = UNOBSERVED_THRESHOLD) -> ClassGrid:
    """EsdfAndGradients response fields -> ClassGrid (unobserved -> NaN)."""
    a = np.asarray(data, np.float32)
    nx, ny, nz = (int(v) for v in dims)
    if a.size != nx * ny * nz:
        raise ValueError(f"ESDF has {a.size} values for dims {nx}x{ny}x{nz}")
    g = a.reshape(nx, ny, nz).astype(np.float32)
    g[g <= unobserved_threshold] = np.nan
    return ClassGrid(g, np.asarray(origin, float), float(voxel_size))


def sample_grid(cg: ClassGrid, points: np.ndarray) -> np.ndarray:
    """Trilinear sample at points (N, 3); NaN when any of the 8 neighbours is
    unobserved or the point lies outside the grid's voxel centres."""
    p = np.asarray(points, float).reshape(-1, 3)
    n = np.array(cg.distance.shape)
    g = (p - cg.origin) / cg.voxel_size - 0.5
    inside = np.all((g >= 0) & (g <= n - 1), axis=1)
    out = np.full(len(p), np.nan)
    if not inside.any() or (n < 1).any():
        return out
    gi = g[inside]
    i0 = np.minimum(np.floor(gi).astype(np.int64), np.maximum(n - 2, 0))
    t = gi - i0
    i1 = np.minimum(i0 + 1, n - 1)
    acc = np.zeros(len(gi))
    for dx in (0, 1):
        wx = t[:, 0] if dx else 1 - t[:, 0]
        ix = i1[:, 0] if dx else i0[:, 0]
        for dy in (0, 1):
            wy = t[:, 1] if dy else 1 - t[:, 1]
            iy = i1[:, 1] if dy else i0[:, 1]
            for dz in (0, 1):
                wz = t[:, 2] if dz else 1 - t[:, 2]
                iz = i1[:, 2] if dz else i0[:, 2]
                acc = acc + wx * wy * wz * cg.distance[ix, iy, iz]      # NaN propagates
    out[inside] = acc
    return out


def merge_class_grids(grids: Dict[str, Optional[ClassGrid]], policies: Dict[str, ClassPolicy],
                      origin: Sequence[float], shape: Sequence[int], voxel_size: float, stamp: float,
                      conservative: bool = True) -> DistanceFieldSnapshot:
    """One snapshot on the output grid from per-class ESDFs (None = no response)."""
    shape = tuple(int(s) for s in shape)
    origin = np.asarray(origin, float)
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    centres = origin + (idx + 0.5) * voxel_size
    per_class = {}
    known = np.zeros(len(centres), bool)
    for name, cg in grids.items():
        if cg is None:
            continue
        d = sample_grid(cg, centres)
        known |= np.isfinite(d)
        per_class[name] = d
    margin = 0.5 * np.sqrt(3) * voxel_size if conservative else 0.0

    hard = [n for n, p in policies.items() if p.behavior == "hard" and n in per_class]
    if hard:
        stack = np.stack([np.where(np.isfinite(per_class[n]), per_class[n], np.inf)
                          - policies[n].padding_m - margin for n in hard])
        k = np.argmin(stack, axis=0)
        hard_d = np.take_along_axis(stack, k[None], axis=0)[0].astype(np.float32).reshape(shape)
        hard_k = k.astype(np.uint8).reshape(shape)
    else:
        hard_d = np.full(shape, np.inf, np.float32)
        hard_k = np.zeros(shape, np.uint8)

    soft = [n for n, p in policies.items() if p.behavior == "soft" and n in per_class]
    soft_d, soft_name, soft_pen = None, "", 0.0
    if soft:
        soft_d = np.min(np.stack([np.where(np.isfinite(per_class[n]), per_class[n], np.inf) for n in soft]),
                        axis=0).astype(np.float32).reshape(shape)
        soft_name = "+".join(soft)
        soft_pen = float(min(policies[n].max_penetration_m for n in soft))

    age = np.where(known, 0, AGE_UNKNOWN).astype(np.uint16).reshape(shape)
    return DistanceFieldSnapshot(origin=origin.copy(), voxel_size=float(voxel_size), shape=shape,
                                 stamp=float(stamp), hard_classes=hard, hard_distance=hard_d,
                                 hard_class=hard_k, soft_class=soft_name, soft_max_penetration=soft_pen,
                                 soft_distance=soft_d, age_ds=age)


def target_points(cg: Optional[ClassGrid], surface_m: Optional[float] = None) -> np.ndarray:
    """Voxel centres on or inside the target surface (d ≤ half a voxel)."""
    if cg is None:
        return np.zeros((0, 3))
    thr = 0.5 * cg.voxel_size if surface_m is None else surface_m
    idx = np.argwhere(np.nan_to_num(cg.distance, nan=np.inf) <= thr)
    return cg.origin + (idx + 0.5) * cg.voxel_size
