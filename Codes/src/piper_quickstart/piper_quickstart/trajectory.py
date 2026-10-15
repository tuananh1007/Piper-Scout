"""Pure helpers for sampling joint trajectories and mapping Piper joint names.

Kept free of ROS imports so they can be unit-tested without a running graph.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

ARM_JOINTS: List[str] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# MoveIt/URDF gripper joints. joint7 and joint8 are the two prismatic fingers
# (joint8 = -joint7); the driver reports a single total opening named "gripper".
GRIPPER_JOINT = "joint7"
MIRROR_GRIPPER_JOINT = "joint8"
DRIVER_GRIPPER_NAME = "gripper"


def sample_trajectory(times: np.ndarray, positions: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate a trajectory at time ``t`` (seconds from start).

    ``times`` is (N,) strictly non-decreasing, ``positions`` is (N, J).
    Before the first point the first point is held, after the last the last.
    """
    if positions.ndim != 2 or positions.shape[0] != times.shape[0]:
        raise ValueError("positions must be (N, J) with N == len(times)")
    if t <= times[0]:
        return positions[0].copy()
    if t >= times[-1]:
        return positions[-1].copy()
    i = int(np.searchsorted(times, t, side="right"))
    t0, t1 = times[i - 1], times[i]
    if t1 <= t0:
        return positions[i].copy()
    a = (t - t0) / (t1 - t0)
    return (1.0 - a) * positions[i - 1] + a * positions[i]


def driver_to_urdf(
    names: Sequence[str], positions: Sequence[float], gripper_scale: float
) -> Dict[str, float]:
    """Map a piper_ros driver JointState to URDF joint positions.

    The driver publishes ``joint1..joint6`` plus ``gripper`` = total finger
    opening in metres. Each URDF finger travels half of that, so
    ``joint7 = gripper / gripper_scale`` and ``joint8 = -joint7``.
    """
    out: Dict[str, float] = {}
    for name, pos in zip(names, positions):
        if name == DRIVER_GRIPPER_NAME:
            j7 = float(pos) / gripper_scale
            out[GRIPPER_JOINT] = j7
            out[MIRROR_GRIPPER_JOINT] = -j7
        elif name in ARM_JOINTS:
            out[name] = float(pos)
    return out


def max_abs_error(target: Dict[str, float], actual: Dict[str, float], joints: Sequence[str]) -> float:
    """Largest |target - actual| over ``joints``; inf if any joint is missing."""
    worst = 0.0
    for j in joints:
        if j not in target or j not in actual:
            return float("inf")
        worst = max(worst, abs(target[j] - actual[j]))
    return worst
