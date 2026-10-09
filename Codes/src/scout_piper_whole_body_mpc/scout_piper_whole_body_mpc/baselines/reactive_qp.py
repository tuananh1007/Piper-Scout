"""W2 baseline: holistic reactive QP (research/whole_body_mpc §15, W2).

One quadratic programme per control cycle over the whole-body command
u = [v, ω, q̇₁..q̇₆], in the spirit of holistic mobile-manipulator controllers
(NEO-style differential IK with the base as extra, non-holonomic joints):

    min_u,s   w_task ‖J_p u Δt − Δp*‖² + uᵀ W u + w_slack s²
    s.t.      u_low ≤ u ≤ u_high,  joint limits one step ahead,
              d_i + ∇d_i · (J_i u Δt) ≥ d_safe − s      (spheres within influence_m)
              e_w + ∇e_w · q̇ Δt ≤ reach_max_m + s      (wrist reach margin, optional)
              s ≥ 0

* Δp* = the TCP error, scaled so the desired TCP speed is at most
  ``v_tcp_max``;
* J_p, J_i are the one-step TCP / collision-sphere Jacobians of the same
  rollout model the MPC uses (finite differences over the eight control
  axes), so the base enters only through (v, ω): the non-holonomic
  constraint holds by construction;
* W makes the base "lazy" (``w_base`` ≫ ``w_joint``): the arm moves first and
  the base only where the arm alone cannot reduce the error;
* the distance gradient ∇d comes from central differences of the same
  ``DistanceFn`` the MPC and the safety filter use.

It plans one step: no horizon, so it can stall in front of an obstacle (the
local minimum the MPC's horizon is meant to avoid). Every command still goes
through ``SafetyFilter``. Solved with SciPy's SLSQP (9 variables).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ..costs.terms import WholeBodyCost
from ..safety.projection import SafetyFilter
from ..sim import SimLog


@dataclass
class ReactiveQPConfig:
    v_tcp_max: float = 0.15          # m/s cap of the desired TCP velocity
    gain: float = 1.5                # 1/s: desired TCP velocity = gain · error (capped)
    w_task: float = 1e4
    w_joint: float = 0.05            # per (rad/s)²
    w_base_v: float = 1.0            # per (m/s)²: lazy base
    w_base_omega: float = 0.5        # per (rad/s)²
    w_slack: float = 1e6
    influence_m: float = 0.10        # spheres closer than this get a clearance constraint
    d_safe: float = 0.03             # same planning margin as the MPC (CostWeights.d_safe)
    joint_margin: float = 0.05       # rad inside the joint limits
    reach_max_m: Optional[float] = None   # wrist reach margin (MPC: reach_max_m); None = off
    grad_eps: float = 1e-3           # m, distance gradient step
    jac_eps: float = 1e-3            # control step for the one-step Jacobians


class ReactiveQP:
    def __init__(self, cost: WholeBodyCost, cfg: ReactiveQPConfig = ReactiveQPConfig()):
        self.cost, self.m, self.cfg = cost, cost.model, cfg

    # ------------------------------------------------------- linearisation
    def _one_step(self, x: np.ndarray):
        """TCP position, collision spheres and their Jacobians per unit control."""
        m, eps = self.m, self.cfg.jac_eps
        U = np.zeros((9, 1, 8))
        U[1:, 0] = eps * np.eye(8)
        Xn = m.rollout(x, U)[:, 1]                             # (9, 9): zero control + one per axis
        p = m.tcp_world(Xn)[:, :3, 3]
        C, r = m.collision_spheres(Xn)                         # (9, S, 3)
        Jp = (p[1:] - p[0]).T / eps                            # (3, 8): Δp per unit u over one step
        Jc = np.moveaxis((C[1:] - C[0]) / eps, 0, -1)          # (S, 3, 8)
        C0, _ = m.collision_spheres(x[None])
        return m.tcp_world(x)[:3, 3], Jp, C0[0], Jc, r

    def _distance_and_gradient(self, C: np.ndarray):
        fn, e = self.cost.distance_fn, self.cfg.grad_eps
        d, valid = fn(C)
        pts = np.concatenate([C + s * e * np.eye(3)[a] for a in range(3) for s in (1, -1)])
        dd, _ = fn(pts)
        dd = dd.reshape(3, 2, len(C))
        g = (dd[:, 0] - dd[:, 1]).T / (2 * e)                  # (S, 3)
        return d, valid, g

    # ---------------------------------------------------------------- solve
    def command(self, x: np.ndarray) -> np.ndarray:
        cfg, m = self.cfg, self.m
        dt = m.dt
        p, Jp, C0, Jc, r = self._one_step(x)
        e = self.cost.goal.p - p
        v_des = cfg.gain * e
        n = np.linalg.norm(v_des)
        if n > cfg.v_tcp_max:
            v_des *= cfg.v_tcp_max / n
        dp = v_des * dt
        Wd = np.r_[cfg.w_base_v, cfg.w_base_omega, [cfg.w_joint] * 6]

        lo, hi = m.u_low.copy(), m.u_high.copy()
        q = x[3:]
        lo[2:] = np.maximum(lo[2:], (m.kin.lower + cfg.joint_margin - q) / dt)
        hi[2:] = np.minimum(hi[2:], (m.kin.upper - cfg.joint_margin - q) / dt)
        lo = np.minimum(lo, hi)                                # outside the margin: allow standing still

        G, h = [], []                                          # G u + h ≥ −s
        if self.cost.distance_fn is not None:
            d, valid, g = self._distance_and_gradient(C0)
            clear = d - r
            for i in np.flatnonzero((clear < cfg.influence_m) | ~valid):
                G.append(g[i] @ Jc[i])                         # Δclearance per unit u
                h.append(clear[i] - cfg.d_safe)
        if cfg.reach_max_m is not None:
            ext0 = m.kin.wrist_extension(q)
            dq = np.zeros(6)
            dq[2] = 1e-4
            gq = (m.kin.wrist_extension(q + dq) - m.kin.wrist_extension(q - dq)) / 2e-4   # depends on q3 only
            row = np.zeros(8)
            row[4] = -gq * dt
            G.append(row)
            h.append(cfg.reach_max_m - ext0)
        G, h = np.array(G).reshape(-1, 8), np.array(h)

        def f(z):
            u, s = z[:8], z[8]
            res = Jp @ u - dp
            return cfg.w_task * res @ res + (Wd * u * u).sum() + cfg.w_slack * s * s

        def grad(z):
            u, s = z[:8], z[8]
            g_u = 2 * cfg.w_task * Jp.T @ (Jp @ u - dp) + 2 * Wd * u
            return np.r_[g_u, 2 * cfg.w_slack * s]

        cons = []
        if len(G):
            cons.append({"type": "ineq", "fun": lambda z: G @ z[:8] + h + z[8],
                         "jac": lambda z: np.c_[G, np.ones(len(G))]})
        z0 = np.zeros(9)
        res = minimize(f, z0, jac=grad, bounds=list(zip(lo, hi)) + [(0.0, None)],
                       constraints=cons, method="SLSQP", options={"maxiter": 100, "ftol": 1e-10})
        u = np.clip(res.x[:8], lo, hi) if np.all(np.isfinite(res.x)) else np.zeros(8)
        return u


def run_reactive_qp(x0: np.ndarray, cost: WholeBodyCost, safety: SafetyFilter, steps: int = 150,
                    cfg: ReactiveQPConfig = ReactiveQPConfig(), tol_m: float = 0.01) -> SimLog:
    """Closed loop W2 → safety filter → model, same termination as ``run_closed_loop``."""
    qp = ReactiveQP(cost, cfg)
    m = cost.model
    x = np.asarray(x0, float).copy()
    u_prev = np.zeros(8)
    log = SimLog(X=[x.copy()])
    for _ in range(steps):
        u = qp.command(x)
        rep = safety.project(x, u, u_prev)
        x = m.rollout(x, rep.u[None, None])[0, 1]
        u_prev = rep.u
        log.X.append(x.copy()); log.U.append(rep.u); log.reasons.append(rep.reason)
        log.min_clearance.append(rep.min_clearance)
        if np.linalg.norm(m.tcp_world(x)[:3, 3] - cost.goal.p) < tol_m and np.abs(rep.u).max() < 0.05:
            break
    return log
