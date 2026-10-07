"""Stage C: JEPA-aware terms for the whole-body MPC (research plan §16).

    J = J_geo + w_v J_vis + w_i J_id

    J_vis = Σ_k [ ‖(û_{t+k} − u_des)/d‖² + λ_H Ĥ_{t+k} + B_FoV(û_{t+k}) + w_lost (1 − v̂_{t+k}) ]
    J_id  = Σ_k [ 1 − cos(r̂_{t+k}, r_t) ]

û, Ĥ, r̂ come from ``readout_sequence`` on the features a ``StatePredictor``
forecasts for each candidate rollout; d is the image diagonal, B_FoV a
quadratic barrier inside ``fov_margin_px`` of the border, and v̂ a soft
visibility, sigmoid((peak − threshold) / visibility_softness), so finite
differences (the MPPI refinement) see a slope rather than a step.

Geometry-anchored read-out (deviation from §12, measured on the synthetic
twin scene): when the Stage A memory has a metric target position p_world and
the cost knows the camera pose of each planned state, each predicted step is
read out only around the projection of p_world (gate ``anchor_gate_px``), and
a projection behind the camera or outside the image counts as out of view.
Appearance cannot separate the selected flower from an identical twin, so the
plain sequential read-out happily scored views of the twin as "visible, same
identity"; anchoring on explicit geometry (the research plan's own division of
labour) removes that failure. Without p_world (thin peduncles without depth)
the cost falls back to the sequential read-out.

``JepaVisibilityCost`` is an ``ExtraTerm`` for
``scout_piper_whole_body_mpc.costs.terms.WholeBodyCost.extra``: call
``set_context`` every control cycle with the latest feature history, the
target descriptor and the target's current image location (from the Stage A
memory); without a context it adds nothing, so the controller degrades to the
geometry-only MPC (C2).

Geometry, collision and the safety filter are untouched: this only re-ranks
geometrically valid motions by what they do to the view of the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Tuple

import numpy as np

from .predictor import StatePredictor
from .readout import ReadoutConfig, readout, readout_sequence


@dataclass
class VisibilityCostWeights:
    w_vis: float = 1.0
    w_id: float = 1.0
    lambda_entropy: float = 0.5
    w_lost: float = 2.0
    w_fov: float = 4.0
    fov_margin_px: float = 10.0
    visibility_softness: float = 0.05
    u_des: Optional[Tuple[float, float]] = None      # default: image centre
    anchor_gate_px: float = 16.0                     # read-out window around the projected target


@dataclass
class JepaVisibilityCost:
    predictor: StatePredictor
    image_hw: Tuple[int, int]
    readout_cfg: ReadoutConfig = field(default_factory=ReadoutConfig)
    w: VisibilityCostWeights = field(default_factory=VisibilityCostWeights)
    # optional geometry anchor: states (..., 9) -> T_world_cam (..., 4, 4), intrinsics
    camera_pose: Optional[Callable[[np.ndarray], np.ndarray]] = None
    K: Optional[np.ndarray] = None
    stride: int = 1                       # must match the predictor's step

    def __post_init__(self) -> None:
        self.Z_hist: Optional[np.ndarray] = None
        self.r: Optional[np.ndarray] = None
        self.u_now: Optional[np.ndarray] = None
        self.p_world: Optional[np.ndarray] = None
        self.last_terms: dict = {}

    def set_context(self, Z_hist: np.ndarray, r: np.ndarray, u_now: np.ndarray,
                    p_world: Optional[np.ndarray] = None) -> None:
        """Z_hist (K, Hf, Wf, C) newest last; r (C,) target descriptor; u_now (2,)
        px; p_world (3,) metric target position when depth was valid."""
        self.Z_hist = np.asarray(Z_hist)
        self.r = np.asarray(r)
        self.u_now = np.asarray(u_now, float)
        self.p_world = None if p_world is None else np.asarray(p_world, float)

    def clear(self) -> None:
        self.Z_hist = self.r = self.u_now = self.p_world = None

    def _anchor(self, X: np.ndarray):
        """Projection of p_world into the camera at each predicted state:
        u (B, H', 2) and in_view (B, H')."""
        Xs = X[:, ::self.stride][:, 1:]
        T = self.camera_pose(Xs)
        p = np.einsum("...ij,j->...i", np.linalg.inv(T), np.r_[self.p_world, 1.0])[..., :3]
        z = p[..., 2]
        zs = np.where(z > 1e-3, z, 1.0)
        u = np.stack([self.K[0, 0] * p[..., 0] / zs + self.K[0, 2],
                      self.K[1, 1] * p[..., 1] / zs + self.K[1, 2]], -1)
        H, W = self.image_hw
        in_view = (z > 1e-3) & (u[..., 0] >= 0) & (u[..., 0] < W) & (u[..., 1] >= 0) & (u[..., 1] < H)
        return u, in_view

    def terms(self, X: np.ndarray) -> dict:
        """Per-sample J_vis and J_id (B,) for state rollouts X (B, H+1, 9)."""
        w = self.w
        H_img, W_img = self.image_hw
        Zp = self.predictor.predict(self.Z_hist, X)                       # (B, H', Hf, Wf, C)
        anchored = self.p_world is not None and self.camera_pose is not None and self.K is not None
        if anchored:
            u_geo, in_view = self._anchor(np.asarray(X, float))
            if u_geo.shape[1] != Zp.shape[1]:
                raise ValueError(f"cost stride {self.stride} does not match the predictor "
                                 f"({u_geo.shape[1]} vs {Zp.shape[1]} predicted steps)")
            cfg = replace(self.readout_cfg, gate_px=w.anchor_gate_px)
            ro = readout(Zp, self.r, self.image_hw, cfg, prior_u=np.where(in_view[..., None], u_geo, -1e3))
            ro.peak = np.where(in_view, ro.peak, -1.0)                       # out of view: not visible
            ro.u = np.where(in_view[..., None], u_geo, ro.u)
        else:
            ro = readout_sequence(Zp, self.r, self.u_now, self.image_hw, self.readout_cfg)
        u_des = np.array(w.u_des if w.u_des is not None else (W_img / 2, H_img / 2))
        diag = float(np.hypot(H_img, W_img))
        e = ((ro.u - u_des) / diag) ** 2
        center = e.sum(-1)
        m = w.fov_margin_px
        edge = np.stack([ro.u[..., 0], W_img - ro.u[..., 0], ro.u[..., 1], H_img - ro.u[..., 1]], -1)
        fov = (np.clip(m - edge, 0, None) / m) ** 2
        soft_vis = 1.0 / (1.0 + np.exp(-(ro.peak - self.readout_cfg.visible_similarity) / w.visibility_softness))
        j_vis = (center + w.lambda_entropy * ro.entropy + w.w_fov * fov.sum(-1)
                 + w.w_lost * (1.0 - soft_vis)).sum(-1)
        j_id = (1.0 - ro.identity).sum(-1)
        return {"J_vis": j_vis, "J_id": j_id, "visible": soft_vis, "u": ro.u}

    def __call__(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        if self.Z_hist is None:
            return np.zeros(len(X))
        t = self.terms(X)
        self.last_terms = {k: v for k, v in t.items() if k in ("J_vis", "J_id")}
        return self.w.w_vis * t["J_vis"] + self.w.w_id * t["J_id"]
