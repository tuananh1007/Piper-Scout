"""Per-instance description of the segmentation for the operator layer (Phase 5).

``describe_instances`` turns instance masks into what language grounding needs
and an operator interface can show: class (id and name), confidence,
bounding box, mask centroid and area in pixels, mean colour (HSV and a colour
name) and, with aligned depth and intrinsics, the centroid's 3-D point in the
camera optical frame (median depth of the mask's core), and the outline of
its largest region as a polygon (a few dozen points) from which the
operator layer rasterises the target mask for the tracker. The segmentation
node publishes the list as JSON on ``/stem_grasp/detections``
(``publish_detections``), each instance with a stable index within the frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# hue ranges (OpenCV H in [0, 180)) for colour names; saturation / value gates first
_HUES = [(0, 8, "red"), (8, 22, "orange"), (22, 35, "yellow"), (35, 85, "green"), (85, 128, "blue"),
         (128, 160, "purple"), (160, 172, "pink"), (172, 180, "red")]


def colour_name(h: float, s: float, v: float) -> str:
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 180 else "grey"
    for lo, hi, name in _HUES:
        if lo <= h < hi:
            return name
    return "red"


def _outline(mask: np.ndarray, eps_px: float = 1.5) -> List[List[int]]:
    import cv2  # noqa: PLC0415
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps_px, True)
    return [[int(x), int(y)] for x, y in c[:, 0]]


def describe_instances(masks: Sequence[np.ndarray], classes: Sequence[int], confs: Sequence[float],
                       names: Dict[int, str], bgr: np.ndarray, depth_m: Optional[np.ndarray] = None,
                       K: Optional[np.ndarray] = None, min_pixels: int = 30) -> List[Dict]:
    import cv2  # noqa: PLC0415
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, W = bgr.shape[:2]
    out = []
    for i, m in enumerate(masks):
        mm = np.asarray(m) > 0
        if mm.shape != (H, W):
            mm = cv2.resize(mm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        n = int(mm.sum())
        if n < min_pixels:
            continue
        v, u = np.nonzero(mm)
        hh, ss, vv = (hsv[..., k][mm].astype(float) for k in range(3))
        # circular mean of hue (red wraps around 0/180)
        ang = hh / 180.0 * 2 * np.pi
        h_mean = float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / (2 * np.pi) * 180.0)
        cid = int(classes[i]) if i < len(classes) else -1
        det = {"index": len(out), "class_id": cid, "class": names.get(cid, str(cid)),
               "conf": round(float(confs[i]) if i < len(confs) else 0.0, 3),
               "bbox": [int(u.min()), int(v.min()), int(u.max()), int(v.max())],
               "centroid": [round(float(u.mean()), 1), round(float(v.mean()), 1)], "area_px": n,
               "hsv": [round(h_mean, 1), round(float(ss.mean()), 1), round(float(vv.mean()), 1)],
               "colour": colour_name(h_mean, float(ss.mean()), float(vv.mean())), "p_camera": None,
               "polygon": _outline(mm)}
        if depth_m is not None and K is not None:
            core = cv2.erode(mm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            sel = core if core.sum() >= 10 else mm
            z = depth_m[sel]
            z = z[np.isfinite(z) & (z > 0)]
            if len(z) >= 5:
                zc = float(np.median(z))
                cu, cv = det["centroid"]
                det["p_camera"] = [round((cu - K[0, 2]) * zc / K[0, 0], 4), round((cv - K[1, 2]) * zc / K[1, 1], 4),
                                   round(zc, 4)]
        out.append(det)
    return out


def collect_caption_instances(base: str, stem_instances, captions, instances_for):
    """(masks, class ids, confs, names) from open-vocabulary queries: the stem
    caption's instances as class 0, then each further caption (target and
    detection captions; empty and repeated ones skipped) as the next class,
    its instances from ``instances_for(caption) -> [(mask, conf)]``."""
    names = {0: base}
    masks, classes, confs = [], [], []
    for m, c in stem_instances:
        masks.append(m)
        classes.append(0)
        confs.append(float(c))
    for cap in captions:
        cap = str(cap).strip()
        if not cap or cap.lower() in (n.lower() for n in names.values()):
            continue
        cid = len(names)
        names[cid] = cap
        for m, c in instances_for(cap):
            masks.append(m)
            classes.append(cid)
            confs.append(float(c))
    return masks, classes, confs, names
