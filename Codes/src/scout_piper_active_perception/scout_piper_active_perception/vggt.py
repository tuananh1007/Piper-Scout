"""Multi-view depth from VGGT for thin structures (P4.1).

The D405's stereo depth drops out on stems that project only a few pixels
wide (``scene.DepthSensor`` models this). VGGT (Wang et al., CVPR 2025,
facebookresearch/vggt) predicts per-view depth, confidence and camera poses
from a handful of RGB images in one forward pass, without stereo matching on
the stem itself. This module keeps a sparse set of views, runs the model on
them and turns its output into metric depth that fills the sensor's holes.

  * ``SparseViewBatcher``: keyframes from the wrist camera; a frame is kept
    when its camera moved ``min_baseline_m`` or turned ``min_angle_deg`` from
    every kept one; at most ``max_views`` (VGGT's memory and time grow with
    the number of views; the first keyframe is the anchor and stays).
  * ``VGGTReconstructor``: lazy wrapper around the model (``vggt`` package,
    weights ``facebook/VGGT-1B``, ~1.2 B parameters; run in bf16/fp16 on the
    GPU). ``model_fn`` replaces it (tests, other backends): it maps images
    (S, 3, H, W) in [0, 1] to depth (S, H, W), depth_conf (S, H, W),
    extrinsic (S, 3, 4) camera-from-world, intrinsic (S, 3, 3).
  * ``metric_scale``: VGGT's output is up to scale (its world is the first
    camera, normalised). The scale comes from the D405 where both have depth
    (median ratio over confident pixels) or, without sensor depth, from the
    robot's camera baselines (FK + hand-eye) against VGGT's.
  * ``DepthFiller``: keeps the newest keyframe's reconstruction and fills
    the current frame only while the camera is still at that keyframe (the
    scene integrators pair depth with the newest masks); ``fused_depth_node``
    publishes the result for the semantic map;
  * ``fuse_depth``: sensor depth where valid; scaled VGGT depth where the
    sensor has none and VGGT is confident. Where both have depth, the share
    agreeing within ``max_rel_diff`` is reported (a low value means a wrong
    scale or a failed reconstruction). The fused image can replace the
    aligned depth for the semantic map and the stem estimate.

The model itself is not part of this repository and needs a GPU; the Orin
deployment (TensorRT or torch, batch size from memory) is MODULE_TASKS.md B8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def is_novel_view(T: np.ndarray, poses, min_baseline_m: float = 0.03, min_angle_deg: float = 8.0) -> bool:
    """True unless some pose in ``poses`` is within both the baseline and
    the angle of camera pose ``T``."""
    for P in poses:
        if (np.linalg.norm(T[:3, 3] - P[:3, 3]) < min_baseline_m
                and np.degrees(_rot_angle(T[:3, :3], P[:3, :3])) < min_angle_deg):
            return False
    return True


@dataclass
class View:
    rgb: np.ndarray                    # (H, W, 3) uint8
    depth: Optional[np.ndarray]        # (H, W) metres, NaN/0 invalid (aligned to rgb)
    T_world_cam: np.ndarray            # (4, 4) from FK + hand-eye
    K: np.ndarray                      # (3, 3)
    stamp: float


@dataclass
class SparseViewBatcher:
    max_views: int = 8
    min_baseline_m: float = 0.03
    min_angle_deg: float = 8.0
    views: List[View] = field(default_factory=list)

    def novel(self, T: np.ndarray) -> bool:
        return is_novel_view(T, [v.T_world_cam for v in self.views], self.min_baseline_m, self.min_angle_deg)

    def offer(self, view: View) -> bool:
        """Keep the frame if it adds a viewpoint; drops the oldest non-anchor
        keyframe when full."""
        if not self.novel(view.T_world_cam):
            return False
        self.views.append(view)
        if len(self.views) > self.max_views:
            del self.views[1]
        return True

    def ready(self, min_views: int = 2) -> bool:
        return len(self.views) >= min_views

    def clear(self) -> None:
        self.views.clear()


def preprocess(images: np.ndarray, width: int = 518, patch: int = 14) -> np.ndarray:
    """uint8 (S, H, W, 3) -> float32 (S, 3, H', W') in [0, 1]: width ``width``,
    height scaled with the aspect ratio and rounded to a multiple of the patch
    size (VGGT's input convention)."""
    import cv2  # noqa: PLC0415
    S, H, W, _ = images.shape
    h = max(patch, int(round(H * width / W / patch)) * patch)
    out = np.stack([cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA if W > width else cv2.INTER_LINEAR)
                    for im in images])
    return np.ascontiguousarray(out.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)


class VGGTReconstructor:
    def __init__(self, model_fn: Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]] = None,
                 weights: str = "facebook/VGGT-1B", device: str = "cuda", width: int = 518):
        self.model_fn = model_fn
        self.weights, self.device, self.width = weights, device, width
        self._model = None

    def _load(self):
        import torch  # noqa: PLC0415
        from vggt.models.vggt import VGGT  # noqa: PLC0415
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: PLC0415
        model = VGGT.from_pretrained(self.weights).to(self.device).eval()
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        dtype = (torch.bfloat16 if use_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

        def fn(images: np.ndarray) -> Dict[str, np.ndarray]:
            x = torch.from_numpy(images).to(self.device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=use_cuda):
                pred = model(x)
            extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], x.shape[-2:])
            return {"depth": pred["depth"][0, ..., 0].float().cpu().numpy(),
                    "depth_conf": pred["depth_conf"][0].float().cpu().numpy(),
                    "extrinsic": extr[0].float().cpu().numpy(), "intrinsic": intr[0].float().cpu().numpy()}
        return fn

    def __call__(self, images_uint8: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model_fn is None:
            self.model_fn = self._load()
        x = preprocess(images_uint8, self.width)
        out = dict(self.model_fn(x))
        out["input_hw"] = x.shape[-2:]
        return out


def resize_to(arr: np.ndarray, hw, nearest: bool = False) -> np.ndarray:
    import cv2  # noqa: PLC0415
    return cv2.resize(arr.astype(np.float32), (int(hw[1]), int(hw[0])),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def metric_scale(pred: Dict[str, np.ndarray], views: List[View], min_conf: float = 3.0,
                 min_pixels: int = 200) -> Optional[float]:
    """Scale from VGGT units to metres: median sensor/VGGT depth ratio over
    confident pixels with sensor depth in all views, else the ratio of the
    robot's camera baselines to VGGT's (needs a baseline), else None."""
    ratios = []
    for i, v in enumerate(views):
        if v.depth is None:
            continue
        d = resize_to(pred["depth"][i], v.depth.shape)
        c = resize_to(pred["depth_conf"][i], v.depth.shape)
        ok = np.isfinite(v.depth) & (v.depth > 0) & (d > 1e-6) & (c >= min_conf)
        if ok.any():
            ratios.append(v.depth[ok] / d[ok])
    if ratios and sum(len(r) for r in ratios) >= min_pixels:
        return float(np.median(np.concatenate(ratios)))
    E = pred["extrinsic"]                                     # camera-from-world
    c_v = np.stack([-E[i, :3, :3].T @ E[i, :3, 3] for i in range(len(views))])
    c_r = np.stack([v.T_world_cam[:3, 3] for v in views])
    b_v = np.linalg.norm(c_v - c_v[0], axis=1).sum()
    b_r = np.linalg.norm(c_r - c_r[0], axis=1).sum()
    return float(b_r / b_v) if b_v > 1e-6 and b_r > 0.01 else None


def fuse_depth(sensor: np.ndarray, vggt_metric: np.ndarray, conf: np.ndarray, min_conf: float = 3.0,
               max_rel_diff: float = 0.05):
    """(fused depth, source, agreement): source 0 none, 1 sensor, 2 VGGT;
    agreement the share of pixels with both depths within ``max_rel_diff``
    (NaN without overlap)."""
    s_ok = np.isfinite(sensor) & (sensor > 0)
    v_ok = np.isfinite(vggt_metric) & (vggt_metric > 0) & (conf >= min_conf)
    fused = np.where(s_ok, sensor, np.where(v_ok, vggt_metric, np.nan))
    src = np.where(s_ok, 1, np.where(v_ok, 2, 0)).astype(np.uint8)
    both = s_ok & v_ok
    agree = float(np.mean(np.abs(vggt_metric[both] - sensor[both]) <= max_rel_diff * sensor[both])) if both.any() \
        else float("nan")
    return fused, src, agree


def reconstruct(batcher: SparseViewBatcher, model: VGGTReconstructor, min_conf: float = 3.0):
    """Run the model on the batcher's views; per view (fused metric depth at
    the sensor resolution, source map, agreement), and the scale. None without
    a scale."""
    views = batcher.views
    pred = model(np.stack([v.rgb for v in views]))
    s = metric_scale(pred, views, min_conf)
    if s is None:
        return None
    out = []
    for i, v in enumerate(views):
        hw = v.depth.shape if v.depth is not None else v.rgb.shape[:2]
        d = s * resize_to(pred["depth"][i], hw)
        c = resize_to(pred["depth_conf"][i], hw)
        sensor = v.depth if v.depth is not None else np.full(hw, np.nan)
        out.append(fuse_depth(sensor, d, c, min_conf))
    return {"scale": s, "views": out}


class DepthFiller:
    """Fills the current depth frame's holes from the last reconstruction.

    The scene integrators pair each depth frame with the newest class masks,
    so filled depth must belong to the current camera pose. A reconstruction
    is kept for its newest keyframe, and frames are filled only while the
    camera stays within ``still_m`` / ``still_deg`` of that keyframe (after a
    planned view is reached the camera settles there); otherwise the sensor
    depth passes through unchanged."""

    def __init__(self, model: VGGTReconstructor, batcher: SparseViewBatcher, min_views: int = 2,
                 still_m: float = 0.01, still_deg: float = 2.0, min_conf: float = 3.0):
        self.model, self.batcher = model, batcher
        self.min_views, self.still_m, self.still_deg, self.min_conf = min_views, still_m, still_deg, min_conf
        self.result: Optional[Dict] = None
        self.dirty = False

    def add_frame(self, view: View) -> bool:
        added = self.batcher.offer(view)
        self.dirty |= added
        return added

    def ready(self) -> bool:
        return self.dirty and self.batcher.ready(self.min_views)

    def reconstruct(self) -> Optional[Dict]:
        """Run the model on the keyframes; keep the newest keyframe's metric
        depth and confidence. None without a scale."""
        import time  # noqa: PLC0415
        views = list(self.batcher.views)
        self.dirty = False
        t0 = time.perf_counter()
        pred = self.model(np.stack([v.rgb for v in views]))
        s = metric_scale(pred, views, self.min_conf)
        if s is None:
            return None
        k = len(views) - 1
        hw = views[k].depth.shape if views[k].depth is not None else views[k].rgb.shape[:2]
        self.result = {"T": views[k].T_world_cam.copy(), "depth": s * resize_to(pred["depth"][k], hw),
                       "conf": resize_to(pred["depth_conf"][k], hw), "scale": s, "views": len(views),
                       "seconds": time.perf_counter() - t0, "stamp": views[k].stamp}
        return self.result

    def fill(self, depth: np.ndarray, T_world_cam: np.ndarray):
        """(depth out, info dict): fused when the camera is at the
        reconstructed keyframe, else the input unchanged."""
        r = self.result
        if r is None:
            return depth, {"filled": 0, "reason": "no reconstruction"}
        moved = float(np.linalg.norm(T_world_cam[:3, 3] - r["T"][:3, 3]))
        turned = float(np.degrees(_rot_angle(T_world_cam[:3, :3], r["T"][:3, :3])))
        if moved > self.still_m or turned > self.still_deg or r["depth"].shape != depth.shape:
            return depth, {"filled": 0, "reason": f"camera moved {100 * moved:.1f} cm / {turned:.1f} deg"}
        fused, src, agree = fuse_depth(depth, r["depth"], r["conf"], self.min_conf)
        return fused, {"filled": int((src == 2).sum()), "agreement": agree, "reason": "fused"}


def inpaint_stand_in(batcher: SparseViewBatcher, scale: float = 2.5) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    """A stand-in for VGGT (tests, hardware-free checks): each keyframe's
    sensor depth with its holes inpainted, in VGGT-like units (÷ ``scale``),
    confident everywhere. Not a reconstruction."""
    import cv2  # noqa: PLC0415

    def fn(x: np.ndarray) -> Dict[str, np.ndarray]:
        S, _, h, w = x.shape
        out = []
        for v in batcher.views[:S]:
            d = np.nan_to_num(v.depth if v.depth is not None else np.zeros(v.rgb.shape[:2]), nan=0.0).astype(np.float32)
            hole = (d <= 0).astype(np.uint8)
            dmax = max(float(d.max()), 1e-3)
            filled = cv2.inpaint((d / dmax * 255).astype(np.uint8), hole, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255 * dmax
            filled = np.where(hole > 0, filled, d)
            out.append(cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST) / scale)
        E = np.stack([np.c_[np.eye(3), np.zeros(3)] for _ in range(S)])
        return {"depth": np.stack(out), "depth_conf": np.full((S, h, w), 5.0, np.float32), "extrinsic": E,
                "intrinsic": np.repeat(np.eye(3)[None], S, 0)}
    return fn
