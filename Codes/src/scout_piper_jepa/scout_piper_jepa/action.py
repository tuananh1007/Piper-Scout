"""Embodiment-normalised action Γ(x, u) for the Stage B predictor.

Optimizer action u = [v, ω, q̇₁..q̇₆]; predictor action
a = [Δs_b, Δθ_b, Δp_ee(3), Δr_ee(3), Δg] (research plan §9).

Δp_ee / Δr_ee are expressed in the **base frame at time t**, so the predictor
sees the same numbers for the same relative motion anywhere on the floor, and
base yaw is not double counted in the EE terms.

``action_embedding`` maps a commanded control (one step); ``action_from_states``
maps two whole-body states (batched). Training data use the latter on
recorded states (realised motion, slip included) and the predictive MPC cost
uses it on rolled-out states, so both see the same numbers.
"""

from __future__ import annotations

from typing import Callable, Optional

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


def _rot_log_batch(R: np.ndarray) -> np.ndarray:
    """Batched SO(3) log: R (..., 3, 3) -> rotation vectors (..., 3)."""
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    small = th < 1e-6
    near_pi = th > np.pi - 1e-4
    scale = np.where(small, 0.5, th / (2 * np.where(small | near_pi, 1.0, s)))
    out = w * scale[..., None]
    if near_pi.any():                     # w ≈ 0 there: axis from the symmetric part
        B = (R[near_pi] + np.eye(3)) / 2
        axis = B[np.arange(len(B)), :, np.argmax(np.diagonal(B, axis1=-2, axis2=-1), -1)]
        out[near_pi] = axis / np.linalg.norm(axis, axis=-1, keepdims=True) * th[near_pi][..., None]
    return out


def _planar_batch(b: np.ndarray) -> np.ndarray:
    T = np.zeros(b.shape[:-1] + (4, 4))
    c, s = np.cos(b[..., 2]), np.sin(b[..., 2])
    T[..., 0, 0], T[..., 0, 1], T[..., 1, 0], T[..., 1, 1] = c, -s, s, c
    T[..., 2, 2] = T[..., 3, 3] = 1.0
    T[..., 0, 3], T[..., 1, 3] = b[..., 0], b[..., 1]
    return T


def action_from_states(x0: np.ndarray, x1: np.ndarray, fk: FK,
                       dg: Optional[np.ndarray] = None) -> np.ndarray:
    """Γ between whole-body states x = [x_b, y_b, θ_b, q₁..q₆] (..., 9).

    fk must accept batched joint vectors (..., 6) -> (..., 4, 4) in the base
    frame. Returns (..., 9) = [Δs_b, Δθ_b, Δp_ee, Δr_ee, Δg] with the base
    displacement taken along the heading at x0 and the EE motion in the base
    frame at x0.
    """
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    d = x1[..., :2] - x0[..., :2]
    th0 = x0[..., 2]
    ds = d[..., 0] * np.cos(th0) + d[..., 1] * np.sin(th0)
    dth = (x1[..., 2] - th0 + np.pi) % (2 * np.pi) - np.pi
    T_b0_b1 = np.linalg.inv(_planar_batch(x0[..., :3])) @ _planar_batch(x1[..., :3])
    T0 = fk(x0[..., 3:])
    T1 = T_b0_b1 @ fk(x1[..., 3:])
    dp = T1[..., :3, 3] - T0[..., :3, 3]
    dr = _rot_log_batch(T1[..., :3, :3] @ np.swapaxes(T0[..., :3, :3], -1, -2))
    g = np.zeros(ds.shape) if dg is None else np.broadcast_to(dg, ds.shape)
    return np.concatenate([ds[..., None], dth[..., None], dp, dr, g[..., None]], -1)
