"""E3 metrics: action-conditioned prediction (PIPER_JEPA_EXPERIMENTS.md §10).

Per horizon step k (physical time = k × predictor step):
  target_error_px   ‖û_{t+k} − u_{t+k}‖ over samples where the target is truly visible
  visibility_f1     predicted visible (peak ≥ threshold) vs truly visible
  visibility_auroc  threshold-free: peak similarity ranks visible above hidden
                    (deterministic predictors blur the target, which shifts the
                    usable threshold; AUROC separates ranking from calibration)
  identity_acc      among truly visible samples, û is nearer the target than any
                    distractor and within one grid cell of it
  switch_rate       among truly visible samples, û is nearer a distractor (and
                    within one cell of it)
  target_latent     mean L1 error of predicted features on true target cells
  global_latent     mean L1 error over all cells
E_target(H) of the plan is ``cumulative_target_error`` (mean over k ≤ H).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .readout import ReadoutConfig, readout_sequence
from .synthetic import DISTRACTOR, TARGET
from .target_memory import cell_centers_px


@dataclass
class PredictionScores:
    target_error_px: np.ndarray
    visibility_f1: np.ndarray
    visibility_auroc: np.ndarray
    identity_acc: np.ndarray
    switch_rate: np.ndarray
    target_latent: np.ndarray
    global_latent: np.ndarray

    def cumulative_target_error(self, H: int) -> float:
        return float(np.nanmean(self.target_error_px[:H]))

    def summary(self, horizons: Sequence[int] = (1, 2, 4, 8)) -> Dict[str, float]:
        out = {}
        for H in horizons:
            if H <= len(self.target_error_px):
                out[f"E_target@{H}"] = round(self.cumulative_target_error(H), 2)
                out[f"vis_F1@{H}"] = round(float(self.visibility_f1[H - 1]), 3)
                out[f"vis_AUROC@{H}"] = round(float(self.visibility_auroc[H - 1]), 3)
                out[f"id_acc@{H}"] = round(float(self.identity_acc[H - 1]), 3)
                out[f"target_L1@{H}"] = round(float(self.target_latent[H - 1]), 3)
                out[f"global_L1@{H}"] = round(float(self.global_latent[H - 1]), 3)
        return out


def _auroc(score: np.ndarray, true: np.ndarray) -> float:
    """Mann–Whitney AUROC; NaN when one class is missing."""
    pos, neg = score[true], score[~true]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                       # average ranks over ties
        tie = allv == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _f1(pred: np.ndarray, true: np.ndarray) -> float:
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def score_predictions(Z_pred: np.ndarray, Z_true: np.ndarray, label_true: np.ndarray,
                      u_true: np.ndarray, vis_true: np.ndarray, r: np.ndarray, u0: np.ndarray,
                      image_hw: Tuple[int, int], cfg: ReadoutConfig = ReadoutConfig()) -> PredictionScores:
    """Z_pred, Z_true (N, H, Hf, Wf, C); label_true (N, H, Hf, Wf); u_true (N, H, 2);
    vis_true (N, H); u0 (N, 2) target location at prediction time."""
    ro = readout_sequence(Z_pred, r, u0, image_hw, cfg)
    N, H, hf, wf = label_true.shape
    err = np.linalg.norm(ro.u - u_true, axis=-1)                         # N, H
    centers = cell_centers_px((hf, wf), image_hw).reshape(-1, 2)
    cell = float(np.hypot(image_hw[0] / hf, image_hw[1] / wf))
    d_cell = np.linalg.norm(ro.u[..., None, :] - centers, axis=-1)      # N, H, M
    lab = label_true.reshape(N, H, -1)
    d_t = np.where(lab == TARGET, d_cell, np.inf).min(-1)
    d_d = np.where(lab == DISTRACTOR, d_cell, np.inf).min(-1)
    on_target = (d_t < d_d) & (d_t <= cell)
    on_twin = (d_d < d_t) & (d_d <= cell)
    l1 = np.abs(np.asarray(Z_pred, float) - np.asarray(Z_true, float)).sum(-1)    # N, H, Hf, Wf
    tmask = label_true == TARGET
    te, f1, au, ida, sw, tl, gl = [], [], [], [], [], [], []
    for k in range(H):
        v = vis_true[:, k]
        te.append(float(err[v, k].mean()) if v.any() else np.nan)
        f1.append(_f1(ro.visible[:, k], v))
        au.append(_auroc(ro.peak[:, k], v))
        ida.append(float(on_target[v, k].mean()) if v.any() else np.nan)
        sw.append(float(on_twin[v, k].mean()) if v.any() else np.nan)
        tl.append(float(l1[:, k][tmask[:, k]].mean()) if tmask[:, k].any() else np.nan)
        gl.append(float(l1[:, k].mean()))
    return PredictionScores(*(np.array(a) for a in (te, f1, au, ida, sw, tl, gl)))
