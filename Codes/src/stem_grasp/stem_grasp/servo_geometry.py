"""Image-space geometry for the stem servo (P0.4.12).

``core.FullAdaptiveServoController`` is image-based visual servoing on one
point: its Jacobian ``[[-fx/Z, 0, u/Z], [0, -fy/Z, v/Z]]`` maps a camera
translation (camera optical frame) to the motion of that point in the image,
so it needs, per control step,

- the measured image position of the grasp target (``raw_uv``),
- the image position the target should have (``desired_uv``),
- the target's depth along the optical axis (``depth_z``),

and its output velocity is in the camera optical frame.

The desired position is where the target appears when it lies on the
gripper's approach axis (the TCP z axis): the camera is offset from the
gripper, so that is not the image centre. The ROS 1 code used the image
centre with a TODO to project the target instead; this module defines the
projection from the camera-to-TCP transform.

The measured position is taken from the live stem mask at the image row where
the 3-D target projects (the stem's horizontal position at the grasp height);
the whole-mask centroid is the fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def project(K: np.ndarray, p_cam: np.ndarray) -> Optional[np.ndarray]:
    """Pinhole projection of a point in the camera optical frame (z forward)."""
    if p_cam[2] <= 1e-6:
        return None
    return np.array([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
                     K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])


def axis_point_at_depth(origin_cam: np.ndarray, axis_cam: np.ndarray, depth: float,
                        min_axis_z: float = 0.1) -> Optional[np.ndarray]:
    """Point on the line origin + s * axis whose optical depth is ``depth``.

    None when the axis is nearly perpendicular to the optical axis (the line
    never reaches that depth in a well-defined place)."""
    axis = np.asarray(axis_cam, float) / np.linalg.norm(axis_cam)
    if abs(axis[2]) < min_axis_z:
        return None
    s = (depth - origin_cam[2]) / axis[2]
    return np.asarray(origin_cam, float) + s * axis


def desired_uv(K: np.ndarray, tcp_cam: np.ndarray, approach_cam: np.ndarray,
               depth: float) -> Optional[np.ndarray]:
    """Image position of the target when it lies on the gripper approach axis."""
    p = axis_point_at_depth(tcp_cam, approach_cam, depth)
    return None if p is None else project(K, p)


def stem_feature_uv(mask: np.ndarray, v_row: float, band_px: int = 12,
                    min_pixels: int = 10) -> Optional[np.ndarray]:
    """Stem position in the mask at image row ``v_row``.

    Mean column of the mask pixels within ``band_px`` rows of ``v_row``; the
    whole-mask centroid when the band is (nearly) empty; None for an empty
    mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    near = np.abs(ys - v_row) <= band_px
    if near.sum() >= min_pixels:
        return np.array([float(xs[near].mean()), float(v_row)])
    return np.array([float(xs.mean()), float(ys.mean())])
