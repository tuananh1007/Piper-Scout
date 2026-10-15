"""Target class as an attractor for goal generation (P1.3.3).

The ``target`` class (behaviour ``attractor`` in semantic_classes.yaml) marks
the flower / peduncle to grasp. ``target_goal`` turns its occupied voxels into
a pre-grasp pose for the whole-body MPC: the largest connected cluster's
centroid is the grasp point, the approach axis is horizontal from the robot
base toward it (stem_grasp's convention), and the pre-grasp point sits
``offset_m`` in front of it. Published by scene_query_node and the nvblox
bridge as /scene_repr/target_goal (geometry_msgs/PoseStamped, z axis = approach
axis), which /whole_body_mpc/goal_pose accepts as is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class TargetGoal:
    grasp_point: np.ndarray        # (3,) cluster centroid
    pre_grasp: np.ndarray          # (3,) grasp_point − offset · axis
    axis: np.ndarray               # (3,) unit approach direction (TCP z)
    n_voxels: int


def largest_cluster(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Points of the largest 26-connected voxel cluster."""
    from scipy.ndimage import label  # noqa: PLC0415

    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) == 0:
        return p
    idx = np.floor(p / voxel_size).astype(np.int64)
    lo = idx.min(0)
    grid = np.zeros(tuple(idx.max(0) - lo + 1), bool)
    grid[tuple((idx - lo).T)] = True
    lab, n = label(grid, structure=np.ones((3, 3, 3), bool))
    if n <= 1:
        return p
    pl = lab[tuple((idx - lo).T)]
    best = np.bincount(pl, minlength=n + 1)[1:].argmax() + 1
    return p[pl == best]


def target_goal(points: np.ndarray, voxel_size: float, robot_xy: Sequence[float],
                offset_m: float = 0.12, min_voxels: int = 3) -> Optional[TargetGoal]:
    c = largest_cluster(points, voxel_size)
    if len(c) < min_voxels:
        return None
    g = c.mean(0)
    d = np.r_[g[:2] - np.asarray(robot_xy, float)[:2], 0.0]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    axis = d / n
    return TargetGoal(grasp_point=g, pre_grasp=g - offset_m * axis, axis=axis, n_voxels=len(c))


def axis_quaternion(axis: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) of a frame whose z axis is ``axis`` (x kept horizontal)."""
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    w = np.sqrt(max(1.0 + np.trace(R), 1e-12)) / 2
    if w > 1e-3:
        return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w), w])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12)) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q
