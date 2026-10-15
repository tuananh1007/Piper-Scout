"""E1 tracking comparison T0–T4 on annotated episodes (P2A.6, go/no-go gate H1).

    python3 -m scout_piper_jepa.e1 run EP1.npz EP2.npz ... [--config config/e1_methods.yaml]
        [--methods T0,T1,T2,T3,T4] [--device cuda] [--out e1_results.json]

Methods (PIPER_JEPA_EXPERIMENTS.md §7; configured in ``config/e1_methods.yaml``):
  T0  framewise segmentation: the recorded ``seg_masks`` (e.g. /stem_grasp/target_mask),
      largest component per frame, no memory
  T1  conventional tracker: Lucas–Kanade optical flow on corners inside the
      initial mask (forward–backward checked); lost below 3 points, no re-detection
  T2  dense self-supervised features (DINOv2) + the Stage A target memory
  T3  V-JEPA 2 + the Stage A target memory
  T4  V-JEPA 2.1 + the Stage A target memory
  ref the numpy ColorPatchEncoder + memory (pipeline check, not a result)

Every method starts from the first annotated target mask and is scored with
``metrics.score_sequence`` against the annotation (and the distractor masks).
Scores are averaged over episodes weighted by visible frames. The H1 decision
compares T4 with the best non-JEPA method (T0–T2): support needs ID retention
at least ``--min-gain`` (default 0.05) higher, or occlusion recovery at least
one frame faster with no loss of retention.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from .metrics import score_sequence

def default_config() -> str:
    """config/e1_methods.yaml from the installed share directory, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415
        path = os.path.join(get_package_share_directory("scout_piper_jepa"), "config", "e1_methods.yaml")
        if os.path.exists(path):
            return path
    except Exception:  # noqa: BLE001 — not installed with ament
        pass
    return os.path.join(os.path.dirname(__file__), "..", "config", "e1_methods.yaml")


def _largest_component_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    from scipy.ndimage import label  # noqa: PLC0415
    if not mask.any():
        return None
    lab, n = label(mask)
    if n > 1:
        k = np.bincount(lab.ravel())[1:].argmax() + 1
        mask = lab == k
    vv, uu = np.nonzero(mask)
    return np.array([uu.mean() + 0.5, vv.mean() + 0.5])


def track_segmentation(ep: dict, first: int) -> List[Optional[np.ndarray]]:
    if "seg_masks" not in ep:
        raise KeyError("episode has no seg_masks (export with --seg)")
    return [_largest_component_centroid(m) for m in ep["seg_masks"][first:]]


def track_optical_flow(ep: dict, first: int, max_corners: int = 60, fb_tol_px: float = 1.5
                       ) -> List[Optional[np.ndarray]]:
    import cv2  # noqa: PLC0415
    frames = ep["frames"][first:]
    init = ep["target_masks"][first].astype(np.uint8) * 255
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    pts = cv2.goodFeaturesToTrack(prev, max_corners, 0.01, 3, mask=init)
    if pts is None:
        vv, uu = np.nonzero(init)
        pts = np.column_stack([uu, vv])[:: max(len(uu) // max_corners, 1)].astype(np.float32)[:, None]
    out = [np.median(pts[:, 0], 0) if len(pts) else None]
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        if pts is None or len(pts) < 3:
            out.append(None)
            prev = g
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, nxt, None)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((back - pts)[:, 0], axis=1) < fb_tol_px)
        pts = nxt[ok]
        out.append(np.median(pts[:, 0], 0) if len(pts) >= 3 else None)
        prev = g
    return out


def track_memory(ep: dict, first: int, encoder, cfg) -> List[Optional[np.ndarray]]:
    from .episode import run_tracker  # noqa: PLC0415
    return run_tracker(ep["frames"][first:], ep["target_masks"][first], encoder, cfg)


def run_method(spec: dict, episodes: List[str], device: str) -> Dict[str, object]:
    from .encoder import make_encoder  # noqa: PLC0415
    from .target_memory import TargetMemoryConfig  # noqa: PLC0415

    kind = spec["kind"]
    encoder = None
    if kind == "memory":
        kw = {k: v for k, v in spec.items() if k not in ("kind", "encoder", "note")}
        if spec["encoder"] in ("vjepa", "dinov2"):
            kw.setdefault("device", device)
            if spec["encoder"] == "vjepa" and not kw.get("hub_entry"):
                return {"skipped": "hub_entry not set in the config (V-JEPA release entry point)"}
        encoder = make_encoder(spec["encoder"], **kw)
    per_ep, weights = [], []
    for path in episodes:
        ep = dict(np.load(path))
        tm = ep["target_masks"]
        vis = tm.reshape(len(tm), -1).any(1)
        if not vis.any():
            continue
        first = int(np.argmax(vis))
        try:
            if kind == "segmentation":
                preds = track_segmentation(ep, first)
            elif kind == "optical_flow":
                preds = track_optical_flow(ep, first)
            else:
                preds = track_memory(ep, first, encoder, TargetMemoryConfig())
        except KeyError as exc:
            return {"skipped": str(exc)}
        gts = [m if m.any() else None for m in tm[first:]]
        dis = [list(d) for d in ep["distractors"][first:]] if "distractors" in ep else ()
        s = score_sequence(preds, gts, dis)
        per_ep.append(s)
        weights.append(max(s.n_visible, 1))
    if not per_ep:
        return {"skipped": "no annotated episode"}
    w = np.array(weights, float)
    agg = {}
    for k in ("id_retention", "false_switch_rate", "center_error_px", "jitter_px", "occlusion_recovery_frames"):
        v = np.array([getattr(s, k) for s in per_ep], float)
        ok = np.isfinite(v)
        agg[k] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
    agg["episodes"] = len(per_ep)
    agg["visible_frames"] = int(sum(s.n_visible for s in per_ep))
    return agg


def h1_decision(res: Dict[str, dict], min_gain: float = 0.05) -> dict:
    base = {k: v for k, v in res.items() if k in ("T0", "T1", "T2") and "skipped" not in v}
    if "T4" not in res or "skipped" in res["T4"] or not base:
        return {"decision": "incomplete", "reason": "T4 and at least one of T0–T2 are needed"}
    best = max(base, key=lambda k: base[k]["id_retention"])
    t4, b = res["T4"], base[best]
    gain = t4["id_retention"] - b["id_retention"]
    faster = b["occlusion_recovery_frames"] - t4["occlusion_recovery_frames"]
    supported = gain >= min_gain or (faster >= 1.0 and gain >= 0.0)
    return {"decision": "H1 supported (go)" if supported else "H1 not supported (no-go)",
            "best_baseline": best, "id_retention_gain": round(gain, 3), "recovery_frames_faster": round(faster, 2)}


def main(argv=None) -> None:
    import yaml  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("episodes", nargs="+")
    r.add_argument("--config", default="")
    r.add_argument("--methods", default="T0,T1,T2,T3,T4")
    r.add_argument("--device", default="cuda")
    r.add_argument("--min-gain", type=float, default=0.05)
    r.add_argument("--out", default="")
    a = ap.parse_args(argv)
    config = a.config or default_config()
    with open(config, encoding="utf-8") as f:
        methods = yaml.safe_load(f)["methods"]
    res = {}
    for name in a.methods.split(","):
        if name not in methods:
            raise SystemExit(f"{name} not in {config}")
        res[name] = run_method(methods[name], a.episodes, a.device)
        print(json.dumps({name: res[name]}), flush=True)
    res["h1"] = h1_decision(res, a.min_gain)
    print(json.dumps({"h1": res["h1"]}))
    print("\n| Method | ID retention | False switches | Centre error px | Jitter px | Recovery frames |\n"
          "|---|---|---|---|---|---|")
    for name, v in res.items():
        if name == "h1":
            continue
        if "skipped" in v:
            print(f"| {name} | skipped: {v['skipped']} | | | | |")
        else:
            print(f"| {name} | {v['id_retention']:.3f} | {v['false_switch_rate']:.3f} | {v['center_error_px']:.1f} | "
                  f"{v['jitter_px']:.1f} | {v['occlusion_recovery_frames']:.1f} |")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
