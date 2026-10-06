"""E1/E2 tracking metrics (research/piper_jepa/PIPER_JEPA_EXPERIMENTS.md §8).

Inputs per frame:
  pred_u      (2,) predicted target centre in pixels, or None if the method
              reports the target as not visible
  gt_mask     target mask (H, W) or None when the target is truly occluded
  distractors list of masks of other instances (same class), may be empty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class TrackingScores:
    id_retention: float          # hits / frames where GT target visible
    false_switch_rate: float     # frames on a distractor / visible frames
    center_error_px: float       # mean over visible frames with a prediction
    jitter_px: float             # mean |second difference| of predictions
    occlusion_recovery_frames: float   # mean frames to first hit after reappearance
    n_visible: int
    n_frames: int


def _inside(mask: Optional[np.ndarray], u: np.ndarray) -> bool:
    if mask is None:
        return False
    H, W = mask.shape
    x, y = int(u[0]), int(u[1])
    return 0 <= x < W and 0 <= y < H and bool(mask[y, x])


def _centroid(mask: np.ndarray) -> np.ndarray:
    vv, uu = np.nonzero(mask)
    return np.array([uu.mean() + 0.5, vv.mean() + 0.5])


def score_sequence(
    preds: Sequence[Optional[np.ndarray]],
    gt_masks: Sequence[Optional[np.ndarray]],
    distractors: Sequence[Sequence[np.ndarray]] = (),
) -> TrackingScores:
    n = len(preds)
    distractors = list(distractors) or [[] for _ in range(n)]
    hits = switches = visible = 0
    errs: List[float] = []
    recov: List[int] = []
    waiting: Optional[int] = None
    prev_vis = True
    for k in range(n):
        m, u = gt_masks[k], preds[k]
        vis = m is not None and np.asarray(m).any()
        if vis:
            visible += 1
            if not prev_vis:
                waiting = 0                     # target just reappeared
            hit = u is not None and _inside(np.asarray(m) > 0, u)
            if hit:
                hits += 1
                if waiting is not None:
                    recov.append(waiting)
                    waiting = None
            else:
                if waiting is not None:
                    waiting += 1
                if u is not None and any(_inside(np.asarray(d) > 0, u) for d in distractors[k]):
                    switches += 1
            if u is not None:
                errs.append(float(np.linalg.norm(u - _centroid(np.asarray(m) > 0))))
        prev_vis = vis
    if waiting is not None:
        recov.append(waiting)
    tr = np.array([p for p in preds if p is not None])
    jitter = float(np.linalg.norm(np.diff(tr, 2, axis=0), axis=1).mean()) if len(tr) >= 3 else 0.0
    return TrackingScores(
        id_retention=hits / max(visible, 1),
        false_switch_rate=switches / max(visible, 1),
        center_error_px=float(np.mean(errs)) if errs else float("nan"),
        jitter_px=jitter,
        occlusion_recovery_frames=float(np.mean(recov)) if recov else 0.0,
        n_visible=visible, n_frames=n,
    )
