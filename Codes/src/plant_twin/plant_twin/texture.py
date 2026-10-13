"""Leaf texture in UV space, silhouette check and mesh export.

``UVTexture`` bakes the camera image into a texture image over the leaf's
rest plane: each texel (u, v) is a point of the rest shape (``LeafModel.uv``
maps vertices to the same square), lifted by the bending field and moved by
the rigid pose of the frame, projected into the camera and coloured from the
image where it is visible (in front of the camera, inside the image, facing
it, and on the observed surface when a depth image is given). Frames are
blended per texel with weight cos²(view angle), so a texel keeps its colour
from the views that saw it best and fills in as the leaf turns. Texels outside
the outline or in a hole stay transparent.

``silhouette`` rasterises the fitted mesh into the camera; its IoU with the
leaf mask tells when the rest shape no longer explains what the camera sees
(the leaf turned or folded, the mask grew or a second leaf merged in):
``ReinitMonitor`` then asks for a re-initialisation from the current mask,
but not while the gripper touches the leaf (the fingers occlude it then).

``export_obj`` writes OBJ + MTL + PNG (vertex uv = ``LeafModel.uv``), for a
render outside RViz.

Camera poses are (R, t) with world = R @ p_cam + t, as in twin_node.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import bump_basis, rotvec_to_matrix

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _project(points: np.ndarray, K: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray):
    """World points -> (pixel (N, 2), camera-frame points (N, 3))."""
    pc = (np.asarray(points, float) - cam_t) @ cam_R
    z = np.where(pc[:, 2] > 1e-6, pc[:, 2], np.nan)
    uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)
    return uv, pc


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Colours (N, C) of an (H, W, C) image at sub-pixel (u, v) inside it
    (numpy: cv2.remap takes at most 32767 points per map dimension)."""
    h, w = img.shape[:2]
    u = np.clip(uv[:, 0], 0, w - 1)
    v = np.clip(uv[:, 1], 0, h - 1)
    u0 = np.minimum(np.floor(u).astype(int), w - 2) if w > 1 else np.zeros(len(u), int)
    v0 = np.minimum(np.floor(v).astype(int), h - 2) if h > 1 else np.zeros(len(v), int)
    a, b = (u - u0)[:, None], (v - v0)[:, None]
    u1, v1 = np.minimum(u0 + 1, w - 1), np.minimum(v0 + 1, h - 1)
    return ((1 - a) * (1 - b) * img[v0, u0] + a * (1 - b) * img[v0, u1]
            + (1 - a) * b * img[v1, u0] + a * b * img[v1, u1])


def silhouette(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, cam_R: np.ndarray,
               cam_t: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Binary image (H, W) of the mesh's triangles in front of the camera."""
    uv, pc = _project(vertices, K, cam_R, cam_t)
    out = np.zeros(hw, np.uint8)
    ok = np.isfinite(uv).all(1)
    tris = [np.round(uv[f]).astype(np.int32) for f in faces if ok[f].all()]
    if tris:
        cv2.fillPoly(out, tris, 1)
    return out.astype(bool)


def silhouette_iou(model_mask: np.ndarray, observed_mask: np.ndarray) -> float:
    a, b = np.asarray(model_mask, bool), np.asarray(observed_mask) > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


@dataclass
class ReinitMonitor:
    """Re-initialise when the silhouette IoU stays below ``min_iou`` for
    ``frames`` frames in a row without contact (the fingers occlude the leaf
    while touching, so those frames neither count nor reset the run)."""

    min_iou: float = 0.6
    frames: int = 5
    low: int = 0
    last_iou: Optional[float] = None

    def update(self, iou: float, touching: bool) -> bool:
        self.last_iou = iou
        if touching:
            return False
        self.low = self.low + 1 if iou < self.min_iou else 0
        if self.low >= self.frames:
            self.low = 0
            return True
        return False


class UVTexture:
    """Texture image (size × size, RGBA) of a ``LeafModel`` baked from camera frames."""

    def __init__(self, model, size: int = 256):
        self.model, self.size = model, int(size)
        lo, hi = model.outline.min(0), model.outline.max(0)
        span = np.maximum(hi - lo, 1e-9)
        # texel (row i, col j) <-> u = (j + 0.5) / size, v = 1 - (i + 0.5) / size (OBJ: v up)
        c = (np.arange(self.size) + 0.5) / self.size
        uu, vv = np.meshgrid(c, 1.0 - c)
        inside = np.zeros((self.size, self.size), np.uint8)
        to_px = lambda xy: np.round(np.stack([(xy[:, 0] - lo[0]) / span[0] * self.size - 0.5,  # noqa: E731
                                              (1.0 - (xy[:, 1] - lo[1]) / span[1]) * self.size - 0.5],
                                             axis=1)).astype(np.int32)
        cv2.fillPoly(inside, [to_px(model.outline)], 1)
        for h in model.holes:
            cv2.fillPoly(inside, [to_px(h)], 0)
        self.inside = inside.astype(bool)
        self.uv = np.stack([uu[self.inside], vv[self.inside]], axis=1)
        self.rest_xy = lo + self.uv * span
        self.basis = bump_basis(self.uv, model.grid)
        dx = 1e-3                                    # bending slope for the texel normals
        self.basis_du = (bump_basis(self.uv + [dx, 0.0], model.grid) - self.basis) / (dx * span[0])
        self.basis_dv = (bump_basis(self.uv + [0.0, dx], model.grid) - self.basis) / (dx * span[1])
        n = len(self.uv)
        self.rgb = np.zeros((n, 3))
        self.weight = np.zeros(n)

    def texel_points(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """World positions and unit normals of the texels for a parameter vector."""
        rv, t, bend = self.model.split(params)
        R = rotvec_to_matrix(rv)
        local = np.column_stack([self.rest_xy, self.basis @ bend])
        n_local = np.column_stack([-(self.basis_du @ bend), -(self.basis_dv @ bend), np.ones(len(local))])
        n_local /= np.linalg.norm(n_local, axis=1, keepdims=True)
        return local @ R.T + t, n_local @ R.T

    def accumulate(self, params: np.ndarray, image_rgb: np.ndarray, K: np.ndarray, cam_R: np.ndarray,
                   cam_t: np.ndarray, depth: Optional[np.ndarray] = None, depth_tol_m: float = 0.01,
                   min_cos: float = 0.2) -> int:
        """Blend one frame in; returns the number of texels it coloured."""
        pts, nrm = self.texel_points(params)
        uv, pc = _project(pts, K, cam_R, cam_t)
        h, w = image_rgb.shape[:2]
        ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        ray = pts - cam_t
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
        cos = np.abs(np.einsum("ij,ij->i", nrm, ray))       # either side of the leaf
        ok &= cos >= min_cos
        if depth is not None:
            ui = np.clip(np.round(uv[:, 0]), 0, w - 1).astype(int)
            vi = np.clip(np.round(uv[:, 1]), 0, h - 1).astype(int)
            d = np.where(ok, depth[vi, ui], np.nan)
            ok &= np.isfinite(d) & (d > 0) & (np.abs(d - pc[:, 2]) < depth_tol_m)
        if not ok.any():
            return 0
        img = np.asarray(image_rgb, np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        col = _bilinear(img, uv[ok])
        wgt = cos[ok] ** 2
        W = self.weight[ok]
        self.rgb[ok] = (self.rgb[ok] * W[:, None] + col * wgt[:, None]) / (W + wgt)[:, None]
        self.weight[ok] = W + wgt
        return int(ok.sum())

    def coverage(self) -> float:
        """Share of the leaf's texels coloured so far."""
        return float((self.weight > 0).mean()) if len(self.weight) else 0.0

    def image(self) -> np.ndarray:
        """RGBA uint8 (size, size, 4); alpha 0 outside the leaf and where never seen."""
        out = np.zeros((self.size, self.size, 4), np.uint8)
        seen = self.weight > 0
        rgba = np.zeros((len(self.weight), 4), np.uint8)
        rgba[:, :3] = np.clip(np.round(self.rgb * 255.0), 0, 255).astype(np.uint8)
        rgba[:, 3] = np.where(seen, 255, 0)
        out[self.inside] = rgba
        return out


def export_obj(prefix: str, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray,
               texture_rgba: Optional[np.ndarray] = None) -> str:
    """Write ``prefix``.obj (+ .mtl and _texture.png with a texture); returns the OBJ path."""
    base = os.path.basename(prefix)
    obj, mtl, png = prefix + ".obj", prefix + ".mtl", prefix + "_texture.png"
    with open(obj, "w") as f:
        if texture_rgba is not None:
            f.write(f"mtllib {base}.mtl\nusemtl leaf\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in uv:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for tri in np.asarray(faces) + 1:
            f.write("f " + " ".join(f"{i}/{i}" for i in tri) + "\n")
    if texture_rgba is not None:
        cv2.imwrite(png, cv2.cvtColor(np.asarray(texture_rgba, np.uint8), cv2.COLOR_RGBA2BGRA))
        with open(mtl, "w") as f:
            f.write(f"newmtl leaf\nKa 1 1 1\nKd 1 1 1\nd 1\nillum 1\nmap_Kd {os.path.basename(png)}\n")
    return obj
