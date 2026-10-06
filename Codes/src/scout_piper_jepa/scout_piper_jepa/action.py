"""Embodiment-normalised action Γ(x, u) for the Stage B predictor.

Optimizer action u = [v, ω, q̇₁..q̇₆]; predictor action
a = [Δs_b, Δθ_b, Δp_ee(3), Δr_ee(3), Δg] (research plan §9).

Δp_ee / Δr_ee are expressed in the **base frame at time t**, so the predictor
sees the same numbers for the same relative motion anywhere on the floor, and
base yaw is not double counted in the EE terms.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# fk(q) -> 4x4 pose of the end-effector (or camera) in the Scout base frame
FK = Callable[[np.ndarray], np.ndarray]


def _rot_log(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * th / (2 * np.sin(th)) if th < np.pi - 1e-6 else w / max(np.linalg.norm(w), 1e-9) * th


def _planar(x: float, y: float, th: float) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(th), np.sin(th)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def action_embedding(q: np.ndarray, u: np.ndarray, dt: float, fk: FK,
                     dg: float = 0.0) -> np.ndarray:
    """Γ(x_t, u_t) for one control step. Returns (9,)."""
    v, w, qd = float(u[0]), float(u[1]), np.asarray(u[2:8], float)
    ds, dth = v * dt, w * dt
    T0 = fk(q)
    T1_local = fk(q + dt * qd)
    # base moves by (ds, dth) along its own heading: midpoint integration
    Tb = _planar(ds * np.cos(dth / 2), ds * np.sin(dth / 2), dth)
    T1 = Tb @ T1_local                       # EE at t+1 in base frame at t
    dp = T1[:3, 3] - T0[:3, 3]
    dr = _rot_log(T1[:3, :3] @ T0[:3, :3].T)
    return np.concatenate([[ds, dth], dp, dr, [dg]])
