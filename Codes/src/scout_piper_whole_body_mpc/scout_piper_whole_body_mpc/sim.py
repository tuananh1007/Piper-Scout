"""Closed-loop simulation: MPPI → safety filter → model (offline WE2/WE3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .costs.terms import WholeBodyCost
from .safety.projection import SafetyFilter
from .solvers.mppi import MPPI


@dataclass
class SimLog:
    X: List[np.ndarray] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    min_clearance: List[float] = field(default_factory=list)

    @property
    def base_distance(self) -> float:
        P = np.array([x[:2] for x in self.X])
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def run_closed_loop(x0: np.ndarray, mppi: MPPI, cost: WholeBodyCost, safety: SafetyFilter,
                    steps: int = 60, freeze_base: bool = False, tol_m: float = 0.01) -> SimLog:
    """``freeze_base`` additionally zeroes executed base commands; for a fair
    arm-only baseline (W0) also build the solver with ``MPPIConfig(arm_only=True)``
    so it never *plans* with base motion it cannot use."""
    m = mppi.m
    x = np.asarray(x0, float).copy()
    u_prev = np.zeros(8)
    log = SimLog(X=[x.copy()])
    for _ in range(steps):
        if freeze_base:
            mppi.U[:, :2] = 0.0
        u = mppi.solve(x, cost, u_prev)
        if freeze_base:
            u[:2] = 0.0
        rep = safety.project(x, u, u_prev)
        x = m.rollout(x, rep.u[None, None])[0, 1]
        u_prev = rep.u
        mppi.shift()
        log.X.append(x.copy()); log.U.append(rep.u); log.reasons.append(rep.reason)
        log.min_clearance.append(rep.min_clearance)
        if np.linalg.norm(m.tcp_world(x)[:3, 3] - cost.goal.p) < tol_m and np.abs(rep.u).max() < 0.05:
            break
    return log
