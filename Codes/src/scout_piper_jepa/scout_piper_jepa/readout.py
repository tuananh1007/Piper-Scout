"""Target read-outs from dense (predicted) features (research plan §12).

For features Z (..., Hf, Wf, C) and the target descriptor r (C,):

    C(p) = cos(Z(p), r)                     similarity map
    P(p) ∝ exp(C(p)/τ) · prior(p)           target distribution
    û    = Σ P(p) p                         location (px)
    Ĥ    = −Σ P log P / log(Hf·Wf)          normalised entropy
    r̂    = normalise(Σ P(p) Z(p))           target-region descriptor
    c_id = cos(r̂, r)                        identity consistency
    visible = max_{gate} C(p) ≥ visible_similarity

An identical twin has the same appearance, so without a spatial prior P is
bimodal and û lands between the two. ``readout_sequence`` therefore gates
each predicted step around the previous predicted location, the same rule the
Stage A memory uses frame to frame.

Everything is batched over leading dimensions so the predictive MPC cost can
read out (samples × horizon) predictions at once, in float32 (the predicted
features of one control step can be hundreds of MB).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .target_memory import cell_centers_px


@dataclass
class ReadoutConfig:
    temperature: float = 0.05
    visible_similarity: float = 0.6
    prior_sigma_px: Optional[float] = 24.0      # None: no spatial prior (ablation)
    gate_px: Optional[float] = 40.0             # hard gate radius around the prior (None = soft only)


@dataclass
class Readout:
    u: np.ndarray            # (..., 2) px
    entropy: np.ndarray      # (...,) normalised to [0, 1]
    peak: np.ndarray         # (...,) max similarity inside the gate
    visible: np.ndarray      # (...,) bool
    identity: np.ndarray     # (...,) cos(r̂, r)
    prob: np.ndarray         # (..., Hf, Wf)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def readout(Z: np.ndarray, r: np.ndarray, image_hw: Tuple[int, int],
            cfg: ReadoutConfig = ReadoutConfig(),
            prior_u: Optional[np.ndarray] = None) -> Readout:
    """Z (..., Hf, Wf, C); prior_u (..., 2) or None."""
    Z = _unit(np.asarray(Z, np.float32))
    r = _unit(np.asarray(r, np.float32))
    lead, (hf, wf) = Z.shape[:-3], Z.shape[-3:-1]
    sim = Z @ r                                                      # (..., Hf, Wf)
    logits = sim / cfg.temperature
    gate = np.ones(sim.shape, bool)
    if prior_u is not None and cfg.prior_sigma_px is not None:
        C = cell_centers_px((hf, wf), image_hw)                       # (Hf, Wf, 2)
        d2 = ((C - np.asarray(prior_u)[..., None, None, :]) ** 2).sum(-1)
        logits = logits - 0.5 * d2 / cfg.prior_sigma_px ** 2
        if cfg.gate_px is not None:
            gate = d2 <= cfg.gate_px ** 2
            empty = ~gate.reshape(lead + (-1,)).any(-1)
            if empty.any():                                            # prior left the image
                nearest = d2.reshape(lead + (-1,)).min(-1)
                gate = gate | (empty[..., None, None] & (d2 <= nearest[..., None, None]))
            logits = np.where(gate, logits, -np.inf)
    flat = logits.reshape(lead + (-1,))
    flat = flat - flat.max(-1, keepdims=True)
    P = np.exp(flat)
    P /= P.sum(-1, keepdims=True)
    P = P.reshape(sim.shape)
    C = cell_centers_px((hf, wf), image_hw)
    u = (P[..., None] * C).sum((-3, -2))
    Pf = P.reshape(lead + (-1,))
    H = -(Pf * np.log(Pf + 1e-12)).sum(-1) / np.log(hf * wf)
    peak = np.where(gate, sim, -np.inf).reshape(lead + (-1,)).max(-1)
    r_hat = _unit(np.einsum("...hw,...hwc->...c", P, Z))
    ident = r_hat @ r
    return Readout(u=u, entropy=H, peak=peak, visible=peak >= cfg.visible_similarity,
                   identity=ident, prob=P)


def readout_sequence(Zs: np.ndarray, r: np.ndarray, u0: np.ndarray, image_hw: Tuple[int, int],
                     cfg: ReadoutConfig = ReadoutConfig()) -> Readout:
    """Predicted features Zs (..., H, Hf, Wf, C) read out step by step, each
    step gated around the previous step's location (u0 (..., 2) = current
    target location). While a step reads as not visible its location is not
    trusted: the prior stays where the target was last seen."""
    Zs = np.asarray(Zs)
    H = Zs.shape[-4]
    prior = np.broadcast_to(np.asarray(u0, float), Zs.shape[:-4] + (2,)).copy()
    outs = []
    for k in range(H):
        ro = readout(Zs[..., k, :, :, :], r, image_hw, cfg, prior_u=prior)
        prior = np.where(ro.visible[..., None], ro.u, prior)
        outs.append(ro)
    stack = lambda name: np.stack([getattr(o, name) for o in outs], axis=-1 if name != "u" else -2)  # noqa: E731
    return Readout(u=stack("u"), entropy=stack("entropy"), peak=stack("peak"),
                   visible=stack("visible"), identity=stack("identity"),
                   prob=np.stack([o.prob for o in outs], axis=-3))
