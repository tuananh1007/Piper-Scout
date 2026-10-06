"""Scout 2.0 base model: unicycle with identified skid-steer corrections.

    x_{t+1} = x_t + Δt · k_v v_t cos θ_t
    y_{t+1} = y_t + Δt · k_v v_t sin θ_t
    θ_{t+1} = θ_t + Δt · k_ω ω_t

``k_v`` and ``k_ω`` (``ScoutParams``) are the slip / effective-track-width
corrections identified on the real floor (experiment WE1, ``fit_slip``).
Defaults are 1.0 (ideal unicycle) until identified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoutParams:
    k_v: float = 1.0
    k_omega: float = 1.0
    v_max: float = 0.3           # m/s near plants (hardware max is 1.5)
    omega_max: float = 0.6       # rad/s
    a_max: float = 0.5           # m/s²
    alpha_max: float = 1.0       # rad/s²


def rollout_base(b0: np.ndarray, vw: np.ndarray, dt: float, p: ScoutParams = ScoutParams()) -> np.ndarray:
    """b0 (B, 3) or (3,), vw (B, H, 2) -> poses (B, H+1, 3)."""
    vw = np.asarray(vw, float)
    if vw.ndim == 2:
        vw = vw[None]
    B, H, _ = vw.shape
    b = np.broadcast_to(np.asarray(b0, float), (B, 3)).copy()
    out = np.empty((B, H + 1, 3))
    out[:, 0] = b
    v = p.k_v * vw[..., 0]
    w = p.k_omega * vw[..., 1]
    for k in range(H):
        th = out[:, k, 2]
        out[:, k + 1, 0] = out[:, k, 0] + dt * v[:, k] * np.cos(th)
        out[:, k + 1, 1] = out[:, k, 1] + dt * v[:, k] * np.sin(th)
        out[:, k + 1, 2] = th + dt * w[:, k]
    return out


def fit_slip(vw: np.ndarray, poses: np.ndarray, dt: float) -> ScoutParams:
    """Least-squares k_v, k_ω from commanded (v, ω) (N, 2) and measured poses
    (N+1, 3) from an external reference (WE1). Wheel odometry alone hides slip."""
    d = np.diff(poses, axis=0)
    th = poses[:-1, 2]
    fwd = d[:, 0] * np.cos(th) + d[:, 1] * np.sin(th)            # displacement along heading
    dth = np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2]))
    v, w = vw[:, 0] * dt, vw[:, 1] * dt
    k_v = float(v @ fwd / max(v @ v, 1e-12))
    k_w = float(w @ dth / max(w @ w, 1e-12))
    return ScoutParams(k_v=k_v, k_omega=k_w)


def planar_T(b: np.ndarray) -> np.ndarray:
    """Base poses (..., 3) -> homogeneous (..., 4, 4) world_T_base."""
    b = np.asarray(b, float)
    T = np.zeros(b.shape[:-1] + (4, 4))
    c, s = np.cos(b[..., 2]), np.sin(b[..., 2])
    T[..., 0, 0], T[..., 0, 1], T[..., 1, 0], T[..., 1, 1] = c, -s, s, c
    T[..., 2, 2] = T[..., 3, 3] = 1.0
    T[..., 0, 3], T[..., 1, 3] = b[..., 0], b[..., 1]
    return T
