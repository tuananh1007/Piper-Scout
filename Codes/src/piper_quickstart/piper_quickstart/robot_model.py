"""Robot model additions: the D405 camera body as a URDF collision link.

The camera body is rigidly attached to the arm, so it belongs in the URDF:
MoveIt then checks it against the scene and the arm's own links like any
other link, and it shows in RViz with the robot.

Its pose comes from the same hand-eye calibration file camera_mount_publisher
uses (link6 -> camera_link, or link6 -> an optical frame), so the collision
box and the camera TF cannot disagree. The body is a separate link,
``camera_body``, at the camera_link pose: camera_link itself stays owned by
camera_mount_publisher + the RealSense driver.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np

from piper_quickstart.transforms import CameraMount, make_transform

# Intel RealSense D405, from realsense2_description/urdf/_d405.urdf.xacro
# (datasheet: 42 x 42 x 23 mm). camera_link axes: x out of the lens, y left, z up.
D405_SIZE_XYZ = (0.023, 0.042, 0.042)           # depth (along x), width, height
D405_CENTER_IN_CAMERA_LINK = (0.0037 - 0.023 / 2, -0.009, 0.0)  # body centre in camera_link
# camera_link -> optical frames: zero translation (depth and color share the
# D405's depth module), rotation rpy(-pi/2, 0, -pi/2).
_OPTICAL_Q = (-0.5, 0.5, -0.5, 0.5)
D405_CAMERA_LINK_TO = {
    "camera_link": make_transform([0, 0, 0], [0, 0, 0, 1]),
    "camera_color_optical_frame": make_transform([0, 0, 0], _OPTICAL_Q),
    "camera_depth_optical_frame": make_transform([0, 0, 0], _OPTICAL_Q),
    "camera_color_frame": make_transform([0, 0, 0], [0, 0, 0, 1]),
    "camera_depth_frame": make_transform([0, 0, 0], [0, 0, 0, 1]),
}

# Links rigidly attached to the camera's parent (link6). Contact between the
# camera body and these is the mount itself, never a real collision.
RIGID_WITH_FLANGE = ("link6", "gripper_base")


def matrix_to_rpy(r: np.ndarray) -> Tuple[float, float, float]:
    """Rotation matrix -> URDF rpy (fixed-axis X, then Y, then Z: R = Rz(y) Ry(p) Rx(r))."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 1.0 - 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: yaw absorbs roll
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return roll, pitch, yaw


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def camera_link_in_parent(mount: CameraMount) -> np.ndarray:
    """4x4 pose of camera_link in mount.parent_frame, whatever frame the calibration targeted."""
    child = mount.child_frame
    if child not in D405_CAMERA_LINK_TO:
        raise ValueError(f"calibration child_frame {child!r} is not a known D405 frame "
                         f"({sorted(D405_CAMERA_LINK_TO)})")
    return mount.matrix() @ np.linalg.inv(D405_CAMERA_LINK_TO[child])


def camera_body_xacro_args(mount: CameraMount, margin_m: float) -> Dict[str, str]:
    """xacro mappings for urdf/piper_d405.urdf.xacro."""
    t = camera_link_in_parent(mount)
    rpy = matrix_to_rpy(t[:3, :3])
    size = [s + 2 * margin_m for s in D405_SIZE_XYZ]
    fmt = lambda v: " ".join(f"{x:.6f}" for x in v)  # noqa: E731
    return {
        "camera_parent": mount.parent_frame,
        "camera_xyz": fmt(t[:3, 3]),
        "camera_rpy": fmt(rpy),
        "camera_body_size": fmt(size),
        "camera_body_center": fmt(D405_CENTER_IN_CAMERA_LINK),
    }


def add_disabled_collisions(srdf: str, pairs: Iterable[Tuple[str, str, str]]) -> str:
    """Insert <disable_collisions link1 link2 reason/> entries before </robot>."""
    lines = "".join(f'    <disable_collisions link1="{a}" link2="{b}" reason="{why}"/>\n' for a, b, why in pairs)
    idx = srdf.rfind("</robot>")
    if idx < 0:
        raise ValueError("SRDF has no </robot>")
    return srdf[:idx] + lines + srdf[idx:]
