"""Target / distractor masks for exported episodes (P2A.5 E1, P3B.8 E3).

    # masks from any labelling tool (CVAT, SAM 2 video, ...): one PNG per frame
    python3 -m scout_piper_jepa.annotate import EP.npz MASK_DIR [--key target_masks]
    python3 -m scout_piper_jepa.annotate import EP.npz DISTRACTOR_DIR --key distractors

    # draw polygons on keyframes; frames between two keyframes interpolate
    python3 -m scout_piper_jepa.annotate polygons EP.npz --every 5 [--key target_masks]

    # what is annotated so far
    python3 -m scout_piper_jepa.annotate info EP.npz

PNG names give the frame index (``000123.png`` or ``frame_000123.png``);
non-zero pixels are the mask. Frames without a file keep their mask.
``--key distractors`` adds one distractor instance per import (call it once
per neighbouring flower).

Polygon tool (OpenCV window): left click adds a vertex, right click removes
the last one, Enter accepts the keyframe, ``o`` marks the target hidden
(occluded: empty mask), ``n`` / ``p`` next / previous keyframe, ``s`` saves,
``q`` saves and quits. Between two accepted keyframes with the same number of
vertices the polygon is interpolated vertex by vertex; otherwise the nearer
keyframe's polygon is copied. A hidden keyframe hides the frames up to the
next keyframe.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np

Polygon = Optional[np.ndarray]          # (V, 2) pixel vertices (u, v); None = hidden


def interpolate_polygons(keyframes: Dict[int, Polygon], n_frames: int) -> List[Polygon]:
    """Per-frame polygons from keyframe polygons (see module doc)."""
    keys = sorted(k for k in keyframes if 0 <= k < n_frames)
    out: List[Polygon] = [None] * n_frames
    if not keys:
        return out
    for i in range(n_frames):
        prev = max((k for k in keys if k <= i), default=None)
        nxt = min((k for k in keys if k >= i), default=None)
        if prev is None:
            out[i] = keyframes[nxt]
            continue
        if nxt is None or nxt == prev:
            out[i] = keyframes[prev]
            continue
        a, b = keyframes[prev], keyframes[nxt]
        if a is None:
            out[i] = None
        elif b is not None and len(a) == len(b):
            f = (i - prev) / (nxt - prev)
            out[i] = (1 - f) * np.asarray(a, float) + f * np.asarray(b, float)
        else:
            out[i] = a if (i - prev) <= (nxt - i) or b is None else b
    return out


def rasterize(poly: Polygon, hw) -> np.ndarray:
    """Polygon -> bool mask (even-odd rule on pixel centres; no OpenCV needed)."""
    H, W = hw
    m = np.zeros((H, W), bool)
    if poly is None or len(poly) < 3:
        return m
    P = np.asarray(poly, float)
    u0, v0 = np.floor(P.min(0)).astype(int)
    u1, v1 = np.ceil(P.max(0)).astype(int)
    u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W - 1), min(v1, H - 1)
    if u1 < u0 or v1 < v0:
        return m
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    x, y = uu + 0.5, vv + 0.5
    inside = np.zeros(x.shape, bool)
    for (xa, ya), (xb, yb) in zip(P, np.roll(P, -1, 0)):
        cross = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = xa + (y - ya) * (xb - xa) / (yb - ya)
        inside ^= cross & (x < xi)
    m[v0:v1 + 1, u0:u1 + 1] = inside
    return m


def _frame_index(name: str) -> Optional[int]:
    m = re.search(r"(\d+)\.png$", name)
    return int(m.group(1)) if m else None


def _read_png(path: str) -> np.ndarray:
    try:
        import cv2  # noqa: PLC0415
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is not None:
            return m
    except ImportError:
        pass
    from PIL import Image  # noqa: PLC0415
    return np.asarray(Image.open(path))


def import_masks(ep_path: str, mask_dir: str, key: str = "target_masks") -> int:
    ep = dict(np.load(ep_path))
    T, H, W = ep["frames"].shape[:3]
    new = np.zeros((T, H, W), bool)
    idxs = []
    for name in sorted(os.listdir(mask_dir)):
        k = _frame_index(name)
        if k is None or not 0 <= k < T:
            continue
        m = _read_png(os.path.join(mask_dir, name))
        m = m.max(-1) if m.ndim == 3 else m
        if m.shape != (H, W):
            raise ValueError(f"{name}: mask {m.shape} vs frame {(H, W)}")
        new[k] = m > 0
        idxs.append(k)
    if key == "distractors":
        old = ep.get("distractors", np.zeros((T, 0, H, W), bool))
        ep["distractors"] = np.concatenate([old, new[:, None]], 1)
    else:
        cur = ep.get(key, np.zeros((T, H, W), bool)).copy()
        cur[idxs] = new[idxs]
        ep[key] = cur
    np.savez_compressed(ep_path, **ep)
    return len(idxs)


def polygon_tool(ep_path: str, every: int = 5, key: str = "target_masks") -> None:  # pragma: no cover — GUI
    import cv2  # noqa: PLC0415

    ep = dict(np.load(ep_path))
    frames = ep["frames"]
    T, H, W = frames.shape[:3]
    keys = list(range(0, T, max(every, 1)))
    if keys[-1] != T - 1:
        keys.append(T - 1)
    polys: Dict[int, Polygon] = {}
    i, pts = 0, []
    win = "annotate: click vertices, Enter accept, o hidden, n/p next/prev, s save, q quit"

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    def save():
        per = interpolate_polygons(polys, T)
        ep[key] = np.stack([rasterize(p, (H, W)) for p in per])
        np.savez_compressed(ep_path, **ep)
        print(f"saved {len(polys)} keyframes -> {key}")

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        k = keys[i]
        img = cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR).copy()
        if k in polys and polys[k] is not None and not pts:
            cv2.polylines(img, [np.int32(polys[k])], True, (0, 200, 0), 1)
        if pts:
            cv2.polylines(img, [np.int32(pts)], False, (0, 0, 255), 1)
        state = "hidden" if k in polys and polys[k] is None else ("done" if k in polys else "")
        cv2.putText(img, f"frame {k} ({i + 1}/{len(keys)}) {state}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.imshow(win, img)
        c = cv2.waitKey(30) & 0xFF
        if c in (13, 10) and len(pts) >= 3:
            polys[k] = np.array(pts, float)
            pts.clear()
            i = min(i + 1, len(keys) - 1)
        elif c == ord("o"):
            polys[k] = None
            pts.clear()
            i = min(i + 1, len(keys) - 1)
        elif c == ord("n"):
            pts.clear()
            i = min(i + 1, len(keys) - 1)
        elif c == ord("p"):
            pts.clear()
            i = max(i - 1, 0)
        elif c == ord("s"):
            save()
        elif c == ord("q"):
            save()
            break
    cv2.destroyAllWindows()


def info(ep_path: str) -> dict:
    ep = np.load(ep_path)
    tm = ep["target_masks"]
    out = {"frames": int(len(ep["frames"])), "fields": sorted(ep.files),
           "target_annotated_frames": int(tm.reshape(len(tm), -1).any(1).sum())}
    if "distractors" in ep:
        out["distractor_instances"] = int(ep["distractors"].shape[1])
    if "states" in ep:
        out["frames_with_state"] = int(np.isfinite(ep["states"]).all(1).sum())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("import")
    i.add_argument("episode"); i.add_argument("mask_dir")
    i.add_argument("--key", default="target_masks", choices=["target_masks", "distractors", "plant_masks"])
    g = sub.add_parser("polygons")
    g.add_argument("episode"); g.add_argument("--every", type=int, default=5)
    g.add_argument("--key", default="target_masks", choices=["target_masks", "plant_masks"])
    n = sub.add_parser("info")
    n.add_argument("episode")
    a = ap.parse_args(argv)
    if a.cmd == "import":
        print(f"imported {import_masks(a.episode, a.mask_dir, a.key)} masks into {a.key}")
    elif a.cmd == "polygons":
        polygon_tool(a.episode, a.every, a.key)
    else:
        print(info(a.episode))


if __name__ == "__main__":
    main()
