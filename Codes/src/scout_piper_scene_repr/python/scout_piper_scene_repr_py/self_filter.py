"""Robot self-filter for depth images (P1.7.8).

The gripper fingers sit in front of the eye-in-hand camera. Without a filter
their depth pixels become ``other`` (hard) obstacles right at the gripper, so
the planner's own fingers block it. ``robot_mask`` projects boxes attached to
robot links (default: the two Piper fingers, link7 / link8) into the depth
image and marks pixels inside a projected box whose measured depth is no
farther than the box's far side plus ``margin_m``; those pixels are removed
(set invalid) before integration. Boxes are approximate: check them in RViz
(debug image) and adjust ``self_filter_boxes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class LinkBox:
    frame: str
    center: np.ndarray        # (3,) box centre in the link frame
    size: np.ndarray          # (3,) full extents


def parse_boxes(entries: Sequence[str]) -> list:
    """["piper_link7:0,-0.035,0.01,0.02,0.07,0.02", ...] -> [LinkBox]."""
    out = []
    for e in entries or []:
        frame, _, nums = str(e).partition(":")
        v = [float(x) for x in nums.split(",")]
        if len(v) != 6:
            raise ValueError(f"self-filter box {e!r}: frame:cx,cy,cz,sx,sy,sz")
        out.append(LinkBox(frame, np.array(v[:3]), np.array(v[3:])))
    return out


def _corners(box: LinkBox) -> np.ndarray:
    s = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
    return box.center + s * box.size


def _hull_fill(pts: np.ndarray, H: int, W: int) -> np.ndarray:
    """Pixels inside the convex hull of 2-D points."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    m = np.zeros((H, W), bool)
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return m
    u0, v0 = np.floor(pts.min(0)).astype(int)
    u1, v1 = np.ceil(pts.max(0)).astype(int)
    u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W - 1), min(v1, H - 1)
    if u1 < u0 or v1 < v0:
        return m
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    q = np.column_stack([uu.ravel() + 0.5, vv.ravel() + 0.5])
    inside = np.all(q @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9, axis=1)
    m[v0:v1 + 1, u0:u1 + 1] = inside.reshape(vv.shape)
    return m


def robot_mask(depth: np.ndarray, K: np.ndarray, boxes: Sequence[LinkBox],
               T_cam_link: Sequence[Optional[np.ndarray]], margin_m: float = 0.02) -> np.ndarray:
    """Bool (H, W): pixels showing one of the boxes. ``T_cam_link[i]`` maps
    box i's link frame into the camera optical frame (None: skip the box).
    Depth in metres, NaN / 0 = invalid."""
    H, W = depth.shape
    out = np.zeros((H, W), bool)
    for box, T in zip(boxes, T_cam_link):
        if T is None:
            continue
        c = _corners(box) @ T[:3, :3].T + T[:3, 3]
        if (c[:, 2] <= 1e-3).all():
            continue                                    # behind the camera
        c = c[c[:, 2] > 1e-3]
        uv = np.column_stack([K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2], K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]])
        hull = _hull_fill(uv, H, W)
        d = np.nan_to_num(depth, nan=np.inf)
        out |= hull & (d > 0) & (d <= c[:, 2].max() + margin_m)
    return out


DEFAULT_PIPER_FINGER_BOXES = [
    "piper_link7:0.0,-0.035,0.01,0.025,0.075,0.025",
    "piper_link8:0.0,-0.035,0.01,0.025,0.075,0.025",
]
