"""Leaf outline + holes from a segmentation mask, expressed in the leaf plane.

Pipeline (all numpy/OpenCV, no ROS):
  1. ``fit_plane``            — PCA plane through the leaf's 3D points; returns
                                 a rotation vector, origin and the 2D→3D frame.
  2. ``mask_contours``        — outer contour + hole contours from a binary
                                 mask (``cv2.RETR_CCOMP`` hierarchy), simplified
                                 with ``approxPolyDP``.
  3. ``contours_to_plane``    — back-project contour pixels with the depth
                                 image (nearest valid depth within a small
                                 window) and project them into the plane.
  ``outline_from_mask`` chains the three and returns what ``LeafModel`` needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:  # OpenCV is only needed for contour extraction
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None



@dataclass
class LeafPlane:
    origin: np.ndarray        # (3,) world
    rotvec: np.ndarray        # (3,) world_R_leaf as axis-angle
    axes: np.ndarray          # (3, 3) columns = leaf x, y, normal in world

    def to_plane(self, pts: np.ndarray) -> np.ndarray:
        """World (N, 3) → leaf-plane (N, 2), dropping the normal component."""
        return (np.asarray(pts, dtype=float) - self.origin) @ self.axes[:, :2]


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:  # angle ≈ π: axis from the symmetric part
        w, v = np.linalg.eigh(R)
        axis = v[:, np.argmax(w)]
        return axis / np.linalg.norm(axis) * angle
    return axis / n * angle


def fit_plane(points: np.ndarray, toward: Optional[np.ndarray] = None) -> LeafPlane:
    """PCA plane. ``toward`` (e.g. the camera position) orients the normal so
    the leaf's +z faces it; this keeps the normal sign stable across frames."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    axes = vt.T.copy()                       # columns: major, minor, normal
    if toward is not None and np.dot(axes[:, 2], np.asarray(toward) - c) < 0:
        axes[:, 2] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1
    return LeafPlane(origin=c, rotvec=matrix_to_rotvec(axes), axes=axes)


def mask_contours(mask: np.ndarray, epsilon_px: float = 1.5,
                  min_hole_px: int = 20) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Largest outer contour (N, 2) in (u, v) pixels + its hole contours."""
    if cv2 is None:
        raise ImportError("plant_twin.outline needs OpenCV (cv2)")
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("empty mask")
    hier = hier[0]
    outer_ids = [i for i, h in enumerate(hier) if h[3] < 0]
    best = max(outer_ids, key=lambda i: cv2.contourArea(contours[i]))
    outer = cv2.approxPolyDP(contours[best], epsilon_px, True).reshape(-1, 2)
    holes = []
    for i, h in enumerate(hier):
        if h[3] == best and cv2.contourArea(contours[i]) >= min_hole_px:
            holes.append(cv2.approxPolyDP(contours[i], epsilon_px, True).reshape(-1, 2))
    return outer.astype(float), [h.astype(float) for h in holes]


def backproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray,
                window: int = 3) -> np.ndarray:
    """Pixels (N, 2) → camera-frame 3D using the nearest valid depth in a
    ``window``-pixel neighbourhood (contour pixels often sit on a depth edge)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth.shape
    out = np.zeros((len(uv), 3))
    for k, (u, v) in enumerate(np.round(uv).astype(int)):
        u0, u1 = max(u - window, 0), min(u + window + 1, W)
        v0, v1 = max(v - window, 0), min(v + window + 1, H)
        patch = depth[v0:v1, u0:u1]
        valid = np.isfinite(patch) & (patch > 0)
        if not valid.any():
            out[k] = np.nan
            continue
        vv, uu = np.nonzero(valid)
        d2 = (vv + v0 - v) ** 2 + (uu + u0 - u) ** 2
        j = np.argmin(d2)
        z = float(patch[vv[j], uu[j]])
        out[k] = [(u - cx) * z / fx, (v - cy) * z / fy, z]
    return out


def contours_to_plane(contour_uv: np.ndarray, depth: np.ndarray, K: np.ndarray,
                      cam_R: np.ndarray, cam_t: np.ndarray, plane: LeafPlane) -> np.ndarray:
    """Contour pixels → world → leaf plane 2D. NaN-depth pixels are dropped."""
    cam = backproject(contour_uv, depth, K)
    ok = np.isfinite(cam).all(1)
    world = cam[ok] @ cam_R.T + cam_t
    return plane.to_plane(world)


def outline_from_mask(mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
                      cam_R: np.ndarray, cam_t: np.ndarray,
                      leaf_points_world: np.ndarray,
                      min_outline_pts: int = 8) -> Tuple[np.ndarray, List[np.ndarray], LeafPlane]:
    """Return (outline_2d, holes_2d, plane) ready for ``LeafModel``."""
    plane = fit_plane(leaf_points_world, toward=cam_t)
    outer_uv, holes_uv = mask_contours(mask)
    outline = contours_to_plane(outer_uv, depth, K, cam_R, cam_t, plane)
    if len(outline) < min_outline_pts:
        raise ValueError(f"outline has only {len(outline)} points with valid depth")
    holes = []
    for h in holes_uv:
        h2 = contours_to_plane(h, depth, K, cam_R, cam_t, plane)
        if len(h2) >= 3:
            holes.append(h2)
    return outline, holes, plane


def petiole_index(outline: np.ndarray, stem_base_2d: np.ndarray) -> int:
    """Outline vertex closest to the stem base — where the stem attaches."""
    return int(np.argmin(np.linalg.norm(outline - stem_base_2d, axis=1)))
