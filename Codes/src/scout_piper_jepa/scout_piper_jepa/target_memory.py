"""Target-conditioned dense memory (Piper-JEPA Stage A).

Given a dense feature grid F_t (Hf, Wf, C) and an initial target mask, keep a
target descriptor r and, every frame, compute

    C(p)  = cos(F(p), r)                       dense similarity
    P(p)  ∝ softmax(C(p)/τ) · prior(p)          target distribution
    û     = Σ P(p) p,   Σ_u = Σ P(p)(p-û)(p-û)ᵀ  image mean / covariance
    H     = -Σ P(p) log P(p)                    entropy (uncertainty)

``prior`` is a Gaussian around the constant-velocity prediction of the
previous position, combined with a hard **gate**: only cells within
``gate_px`` of the prediction can be the target. Appearance alone cannot
separate the selected flower from an identical neighbour (both have cosine
≈ 1), so without the gate a hidden target is silently replaced by its twin.
While the target is occluded its position coasts at the last velocity and the
gate widens. Once ``lost``, the memory does **not** re-acquire on its own: the
caller must re-ground (``reinitialize``), as the safety rules require. Set
``prior_sigma_px=None`` to ablate both prior and gate.

The descriptor is updated by an exponential moving average only while the
target is confidently visible, and is always blended with the initial
descriptor so it cannot drift onto a neighbour.

Coordinates: everything public is in **image pixels** (u right, v down); the
grid is mapped to pixels by its shape, so encoders that resize the frame work
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- helpers
def mask_to_grid(mask: np.ndarray, grid_hw: Tuple[int, int]) -> np.ndarray:
    """Fraction of each grid cell covered by the (H, W) mask -> (Hf, Wf)."""
    m = (np.asarray(mask) > 0).astype(np.float32)
    H, W = m.shape
    hf, wf = grid_hw
    ys = np.minimum((np.arange(H) * hf) // H, hf - 1)
    xs = np.minimum((np.arange(W) * wf) // W, wf - 1)
    acc = np.zeros((hf, wf), np.float32)
    cnt = np.zeros((hf, wf), np.float32)
    np.add.at(acc, (ys[:, None], xs[None, :]), m)
    np.add.at(cnt, (ys[:, None], xs[None, :]), 1.0)
    return acc / np.maximum(cnt, 1.0)


def cell_centers_px(grid_hw: Tuple[int, int], image_hw: Tuple[int, int]) -> np.ndarray:
    """(Hf, Wf, 2) pixel coordinates (u, v) of each grid-cell centre."""
    hf, wf = grid_hw
    H, W = image_hw
    u = (np.arange(wf) + 0.5) * W / wf
    v = (np.arange(hf) + 0.5) * H / hf
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv], -1)


def _normalize(x: np.ndarray, axis=-1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


# ------------------------------------------------------------------ types
@dataclass
class TargetState:
    stamp: float
    status: str                    # 'tracking' | 'occluded' | 'lost' | 'uninitialized'
    u_mean: np.ndarray             # (2,) px
    u_cov: np.ndarray              # (2, 2) px²
    entropy: float                 # nats, normalised to [0, 1] in entropy_norm
    entropy_norm: float
    confidence: float              # [0, 1]
    peak_similarity: float
    visible: bool
    p_world: Optional[np.ndarray] = None   # (3,) metres in the world, with valid depth and camera pose
    p_cov: Optional[np.ndarray] = None     # (3, 3)
    prob: Optional[np.ndarray] = None      # (Hf, Wf) for debugging


@dataclass
class TargetMemoryConfig:
    temperature: float = 0.05          # τ for the similarity softmax
    prior_sigma_px: Optional[float] = 30.0
    gate_px: float = 48.0              # hard spatial gate radius (≈3 patches)
    max_speed_px: float = 8.0          # gate grows by this per unseen frame
    max_gate_px: float = 120.0         # never search wider than this
    velocity_smoothing: float = 0.5
    visible_similarity: float = 0.6    # peak cos above this ⇒ target visible
    lost_after_frames: int = 30        # consecutive invisible frames ⇒ lost
    ema_alpha: float = 0.1             # descriptor update rate while confident
    anchor_weight: float = 0.5         # blend with the initial descriptor
    update_min_confidence: float = 0.7
    depth_top_mass: float = 0.8        # use cells holding this much P for 3-D


# ------------------------------------------------------------------ memory
@dataclass
class TargetMemory:
    cfg: TargetMemoryConfig = field(default_factory=TargetMemoryConfig)

    def __post_init__(self) -> None:
        self.r0: Optional[np.ndarray] = None
        self.r: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.vel = np.zeros(2)
        self.invisible_run = 0
        self.status = "uninitialized"

    # -------------------------------------------------------------- init
    def initialize(self, features: np.ndarray, mask: np.ndarray) -> None:
        """Bind the memory to the target selected by ``mask`` (image-sized)."""
        hf, wf, _ = features.shape
        w = mask_to_grid(mask, (hf, wf))
        if w.sum() <= 0:
            raise ValueError("target mask covers no feature cells")
        r = (features * w[..., None]).sum((0, 1)) / w.sum()
        self.r0 = self.r = _normalize(r)
        H, W = np.asarray(mask).shape
        vv, uu = np.nonzero(np.asarray(mask) > 0)
        self.u = np.array([uu.mean() + 0.5, vv.mean() + 0.5])
        self.vel = np.zeros(2)
        self.image_hw = (H, W)
        self.invisible_run = 0
        self.status = "tracking"

    def reinitialize(self, features: np.ndarray, mask: np.ndarray) -> None:
        """Re-ground (e.g. after 'lost'): new anchor descriptor and position."""
        self.initialize(features, mask)

    # ------------------------------------------------------------ update
    def update(
        self,
        features: np.ndarray,
        stamp: float = 0.0,
        depth: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
        T_world_cam: Optional[np.ndarray] = None,
        image_hw: Optional[Tuple[int, int]] = None,
        keep_prob: bool = False,
    ) -> TargetState:
        if self.r is None:
            raise RuntimeError("TargetMemory.update before initialize")
        cfg = self.cfg
        hf, wf, _ = features.shape
        image_hw = image_hw or self.image_hw
        centers = cell_centers_px((hf, wf), image_hw)               # hf,wf,2

        sim = features @ self.r                                      # hf,wf
        logits = sim / cfg.temperature
        pred = self.u + self.vel                                     # constant velocity
        gate = np.ones(sim.shape, bool)
        if cfg.prior_sigma_px is not None:
            radius = min(cfg.gate_px + cfg.max_speed_px * self.invisible_run, cfg.max_gate_px)
            grow = radius / cfg.gate_px
            d2 = ((centers - pred) ** 2).sum(-1)
            gate = d2 <= radius ** 2
            if not gate.any():                                       # prediction left the image
                gate = d2 <= d2.min()
            logits = logits - 0.5 * d2 / (cfg.prior_sigma_px * grow) ** 2
            logits = np.where(gate, logits, -np.inf)
        logits -= logits.max()
        P = np.exp(logits)
        P /= P.sum()

        u = (P[..., None] * centers).sum((0, 1))
        d = centers - u
        cov = np.einsum("ij,ijk,ijl->kl", P, d, d)
        Hn = float(-(P * np.log(P + 1e-12)).sum())
        Hnorm = Hn / np.log(P.size)
        peak = float(sim[gate].max())
        visible = peak >= cfg.visible_similarity and self.status != "lost"

        # confidence: similarity margin above the visibility threshold,
        # discounted by how spread the distribution is
        conf = float(np.clip((peak - cfg.visible_similarity) / (1 - cfg.visible_similarity + 1e-6), 0, 1)
                     * (1.0 - Hnorm))
        conf = float(np.clip(conf * 2.0, 0.0, 1.0)) if visible else 0.0

        if visible:
            self.status = "tracking"
            if self.invisible_run == 0:
                a = cfg.velocity_smoothing
                step = u - self.u
                n = np.linalg.norm(step)
                if n > cfg.max_speed_px:          # physically implausible: clamp
                    step *= cfg.max_speed_px / n
                self.vel = a * self.vel + (1 - a) * step
            self.u = u
            if conf >= cfg.update_min_confidence:
                r_new = _normalize((P[..., None] * features).sum((0, 1)))
                r_ema = _normalize((1 - cfg.ema_alpha) * self.r + cfg.ema_alpha * r_new)
                self.r = _normalize(cfg.anchor_weight * self.r0 + (1 - cfg.anchor_weight) * r_ema)
            self.invisible_run = 0
        else:
            # coast at the last velocity; the gate widens via invisible_run
            self.invisible_run += 1
            if self.status != "lost":
                self.u = pred
            self.status = "lost" if self.invisible_run >= cfg.lost_after_frames else "occluded"

        state = TargetState(
            stamp=stamp, status=self.status,
            u_mean=(u if visible else self.u).copy(), u_cov=cov,
            entropy=Hn, entropy_norm=Hnorm, confidence=conf,
            peak_similarity=peak, visible=visible,
            prob=P if keep_prob else None,
        )
        # metric position only with the camera pose: a camera-frame point must not
        # pass for a world point (a TF gap is routine)
        if visible and depth is not None and K is not None and T_world_cam is not None:
            p, pc = self._target_3d(P, centers, depth, K, T_world_cam)
            state.p_world, state.p_cov = p, pc
        return state

    # ---------------------------------------------------------------- 3-D
    def _target_3d(self, P, centers, depth, K, T_world_cam):
        """Back-project the highest-probability cells that have valid depth.

        Thin peduncles often have no depth; then None is returned rather than
        a hallucinated metric position."""
        order = np.argsort(P.ravel())[::-1]
        mass = np.cumsum(P.ravel()[order])
        keep = order[: int(np.searchsorted(mass, self.cfg.depth_top_mass)) + 1]
        H, W = depth.shape
        uv = centers.reshape(-1, 2)[keep]
        w = P.ravel()[keep]
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
        vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
        z = depth[vi, ui].astype(float)
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() < 1:
            return None, None
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts = np.column_stack([(uv[ok, 0] - cx) * z[ok] / fx,
                               (uv[ok, 1] - cy) * z[ok] / fy, z[ok]])
        pts = pts @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
        ww = w[ok] / w[ok].sum()
        mean = (ww[:, None] * pts).sum(0)
        dd = pts - mean
        cov = (ww[:, None, None] * dd[:, :, None] * dd[:, None, :]).sum(0)
        return mean, cov
