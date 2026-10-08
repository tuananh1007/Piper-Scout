"""Cost terms of J_geo (research/whole_body_mpc §5–10).

Every term maps a rollout (X (B, H+1, 9), U (B, H, 8)) to a per-sample cost
(B,). Geometry enters through a ``DistanceFn``:

    distance_fn(points (N, 3)) -> (hard_distance (N,), valid (N,) bool)

which ``scene_adapter.semantic_distance_fn`` builds from the semantic-scene
query; tests and early experiments use analytic fields. Piper-JEPA adds its
visibility/identity terms through ``WholeBodyCost.extra`` without touching
these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..dynamics.whole_body import WholeBodyModel

DistanceFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
LeafFn = Callable[[np.ndarray], np.ndarray]
ExtraTerm = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(c, -1, 1))


@dataclass
class Goal:
    p: np.ndarray                         # (3,) world grasp point
    R: Optional[np.ndarray] = None        # (3, 3) desired TCP orientation, optional
    approach_axis: Optional[np.ndarray] = None   # desired TCP z direction (world), optional


def same_goal(a: Optional[Goal], b: Goal, tol_m: float = 1e-3, tol_axis: float = 1e-3) -> bool:
    """True when b repeats a (position within tol_m, same approach axis)."""
    if a is None or np.linalg.norm(a.p - b.p) > tol_m:
        return False
    if (a.approach_axis is None) != (b.approach_axis is None):
        return False
    return a.approach_axis is None or float(np.linalg.norm(a.approach_axis - b.approach_axis)) < tol_axis


@dataclass
class CostWeights:
    goal: float = 50.0
    goal_terminal: float = 200.0
    goal_terminal_linear: float = 0.0     # optional ‖e‖ term for the last cm; 20–50 helped one
                                          # obstacle case but hurt R3 convergence, so off by default
    orient: float = 2.0
    collision: float = 1e4
    unknown: float = 1e3                  # sphere centre in unknown/stale space
    leaf: float = 1.0
    manip: float = 0.02
    base: float = 100.0                   # tuned: R1 stays mostly arm-only, R3 still drives
    omega: float = 0.5                    # λ_ω inside J_base
    smooth: float = 0.5
    joint_limit: float = 100.0
    # Clearance below which the collision penalty starts. Keep it above the
    # safety filter's d_safe (0.02): with equal margins the planner grazes the
    # boundary the filter refuses to cross and the robot deadlocks there (P3A.6).
    d_safe: float = 0.03


@dataclass
class WholeBodyCost:
    model: WholeBodyModel
    goal: Goal
    distance_fn: Optional[DistanceFn] = None
    leaf_fn: Optional[LeafFn] = None
    w: CostWeights = field(default_factory=CostWeights)
    extra: List[ExtraTerm] = field(default_factory=list)   # e.g. Piper-JEPA J_vis, J_id

    def __call__(self, X: np.ndarray, U: np.ndarray, u_prev: Optional[np.ndarray] = None) -> np.ndarray:
        w, m = self.w, self.model
        B, H1, _ = X.shape
        Xk = X[:, 1:]                                     # costs on predicted states
        T = m.tcp_world(Xk)                               # (B, H, 4, 4)
        e = T[..., :3, 3] - self.goal.p
        dist2 = (e ** 2).sum(-1)
        J = (w.goal * dist2.mean(1) + w.goal_terminal * dist2[:, -1]
             + w.goal_terminal_linear * np.sqrt(dist2[:, -1]))
        if self.goal.R is not None:
            J += w.orient * _rot_angle(T[:, -1, :3, :3], self.goal.R) ** 2
        elif self.goal.approach_axis is not None:
            z = T[:, -1, :3, 2]
            J += w.orient * (1 - z @ self.goal.approach_axis)

        if self.distance_fn is not None:
            C, r = m.collision_spheres(Xk)                 # (B, H, S, 3)
            d, valid = self.distance_fn(C.reshape(-1, 3))
            d = d.reshape(C.shape[:-1]) - r
            valid = valid.reshape(C.shape[:-1])
            viol = np.clip(w.d_safe - d, 0, None)
            # smooth penalty only: a per-violation constant makes every sample
            # that brushes an obstacle look catastrophic, so MPPI stops instead
            # of detouring. Hard clearance is enforced by the safety filter.
            J += w.collision * (viol ** 2).sum((1, 2))
            J += w.unknown * (~valid).sum((1, 2)) / valid.shape[2]
        if self.leaf_fn is not None:
            C, _ = m.collision_spheres(Xk)
            J += w.leaf * self.leaf_fn(C.reshape(-1, 3)).reshape(C.shape[:-1]).sum((1, 2))

        q = Xk[..., 3:]
        J += w.manip * (1.0 / (m.kin.manipulability(q) + 1e-3)).mean(1)
        margin = 0.05
        lim = np.clip(m.kin.lower + margin - q, 0, None) + np.clip(q - (m.kin.upper - margin), 0, None)
        J += w.joint_limit * (lim ** 2).sum((1, 2))

        J += w.base * (U[..., 0] ** 2 + w.omega * U[..., 1] ** 2).mean(1)
        dU = np.diff(U, axis=1)
        J += w.smooth * (dU ** 2).sum((1, 2))
        if u_prev is not None:
            J += w.smooth * ((U[:, 0] - u_prev) ** 2).sum(-1)
        for term in self.extra:
            J += term(X, U)
        return J
