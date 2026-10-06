"""Safety projection (research/whole_body_mpc §12) and watchdog.

The executed command is the closest command to u_MPC that satisfies

  * box limits on v, ω, q̇;
  * rate limits (|u − u_prev| ≤ a_max·Δt) — box ∩ box, so per-axis clipping
    is the exact Euclidean projection;
  * joint limits one step ahead;
  * hard clearance one step ahead: if the next state would bring any
    collision sphere below d_safe **and** reduce its clearance, the command is
    scaled toward zero (bisection on α ∈ [0, 1]). Scaling is a conservative
    approximation of the projection for this non-convex constraint, not the
    exact argmin; it never increases speed.
  * watchdog: stale state, stale geometry or invalid geometry at the robot ⇒ stop.

This is a safety *filter*, not a formal guarantee (see the research plan's
formal-guarantee caution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..costs.terms import DistanceFn
from ..dynamics.whole_body import WholeBodyModel


@dataclass
class SafetyReport:
    u: np.ndarray
    scale: float
    reason: str                 # 'ok' | 'rate' | 'joint_limit' | 'clearance' | 'watchdog'
    min_clearance: float


@dataclass
class SafetyFilter:
    model: WholeBodyModel
    distance_fn: Optional[DistanceFn] = None
    d_safe: float = 0.02
    max_state_age_s: float = 0.2
    max_geometry_age_s: float = 1.0

    def stop(self, reason: str) -> SafetyReport:
        return SafetyReport(np.zeros(8), 0.0, reason, float("nan"))

    def project(self, x: np.ndarray, u_mpc: np.ndarray, u_prev: np.ndarray,
                state_age_s: float = 0.0, geometry_age_s: float = 0.0) -> SafetyReport:
        m = self.model
        if state_age_s > self.max_state_age_s or geometry_age_s > self.max_geometry_age_s:
            return self.stop("watchdog")
        reason = "ok"
        u = np.clip(u_mpc, m.u_low, m.u_high)
        rate = np.r_[m.scout.a_max, m.scout.alpha_max, [m.arm.qdd_max] * 6] * m.dt
        u_r = np.clip(u, u_prev - rate, u_prev + rate)
        if not np.allclose(u_r, u):
            reason = "rate"
        u = u_r
        # joint limits one step ahead: zero the offending joint velocities
        q_next = x[3:] + m.dt * u[2:]
        bad = (q_next < m.kin.lower) | (q_next > m.kin.upper)
        if bad.any():
            u[2:][bad] = 0.0
            reason = "joint_limit"
        if self.distance_fn is None:
            return SafetyReport(u, 1.0, reason, float("nan"))

        C0, r = m.collision_spheres(x[None])
        d0, v0 = self.distance_fn(C0[0])
        if not v0.all():
            return self.stop("watchdog")             # robot inside unknown/stale geometry
        d0 = d0 - r

        def clearance(alpha: float):
            X = m.rollout(x, (alpha * u)[None, None])
            C, _ = m.collision_spheres(X[0, 1])
            d, valid = self.distance_fn(C)
            return d - r, valid

        def ok(alpha: float) -> bool:
            d1, valid = clearance(alpha)
            danger = d1 < self.d_safe
            return bool(valid.all() and not np.any(danger & (d1 < d0 - 1e-9)))

        if ok(1.0):
            return SafetyReport(u, 1.0, reason, float(clearance(1.0)[0].min()))
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return SafetyReport(lo * u, lo, "clearance", float(clearance(lo)[0].min()))
