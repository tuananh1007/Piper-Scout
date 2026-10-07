"""MPPI over the whole-body control sequence (candidate A in §11), with an
optional gradient refinement of the result (candidate B, P3A.6).

Each iteration samples ``samples`` perturbed control sequences around the
nominal one (noise is sampled at ``noise_knots`` knots and linearly
interpolated over the horizon: white noise would be dominated by the
smoothness cost, so the solver would either freeze or move only by averaging
noise), rolls them out with the explicit non-holonomic model, scores
them with ``WholeBodyCost`` and takes the exponentially weighted average.
The nominal sequence is shifted by one step between control cycles (warm
start). Controls are clipped to the model's box limits; per-step rate limits
are left to the safety projection.

Near the goal the weighted average of noisy samples is often *worse* than
samples it was built from, "stop" included, so the executed command carried
sampling noise and the arm wandered 1–4 cm around the goal instead of
settling (P3A.6: the "stall" behind an obstacle had zero collision cost).
Two fixes, both on by default:

  * elitism (``keep_best``): the averaged sequence is evaluated and replaced
    by the best sample of the iteration when that one is cheaper, so a solve
    never returns something worse than a candidate it already scored;
  * gradient refinement (candidate B, ``refine_iters``): steps of gradient
    descent on the same cost in the noise-knot space (central differences:
    2 × knots × 8 rollouts in one batch, then a batched line search over step
    sizes); a step is kept only if it lowers the cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..costs.terms import WholeBodyCost
from ..dynamics.whole_body import WholeBodyModel


@dataclass
class MPPIConfig:
    horizon: int = 20
    samples: int = 256
    iterations: int = 2
    temperature: float = 0.1            # λ on std-normalised costs; 1.0 averages too much noise
    noise_base: tuple = (0.10, 0.25)    # σ for v, ω
    noise_arm: float = 0.25             # σ for each q̇
    noise_knots: int = 4                # temporally smooth noise (0 = white noise)
    include_zero: bool = True           # always evaluate the "stop" sequence
    arm_only: bool = False              # W0 baseline: base never sampled or planned
    keep_best: bool = True              # never return worse than the best scored sample
    refine_iters: int = 2               # candidate B gradient steps per solve (0 = off)
    refine_eps: float = 1e-3            # central-difference step on knot values (rad/s, m/s)
    refine_max_step: float = 0.3        # largest knot change per step, before line search
    seed: Optional[int] = None


class MPPI:
    def __init__(self, model: WholeBodyModel, cfg: MPPIConfig = MPPIConfig()):
        self.m, self.cfg = model, cfg
        self.U = np.zeros((cfg.horizon, 8))
        self.sigma = np.r_[cfg.noise_base, [cfg.noise_arm] * 6]
        if cfg.arm_only:
            self.sigma[:2] = 0.0
        self.rng = np.random.default_rng(cfg.seed)
        self.last_cost = np.inf
        self.last_X: Optional[np.ndarray] = None
        H, k = cfg.horizon, cfg.noise_knots
        if k >= 2:
            t = np.linspace(0, k - 1, H)
            A = np.zeros((H, k))
            i = np.minimum(np.floor(t).astype(int), k - 2)
            f = t - i
            A[np.arange(H), i] = 1 - f
            A[np.arange(H), i + 1] = f
            self._interp = A                     # (H, knots)
        else:
            self._interp = None

    def _noise(self) -> np.ndarray:
        cfg = self.cfg
        if self._interp is None:
            return self.rng.normal(size=(cfg.samples, cfg.horizon, 8)) * self.sigma
        z = self.rng.normal(size=(cfg.samples, self._interp.shape[1], 8)) * self.sigma
        return np.einsum("hk,skd->shd", self._interp, z)

    def reset(self) -> None:
        self.U[:] = 0.0

    def solve(self, x0: np.ndarray, cost: WholeBodyCost, u_prev: Optional[np.ndarray] = None) -> np.ndarray:
        cfg, m = self.cfg, self.m
        lo, hi = m.u_low.copy(), m.u_high.copy()
        if cfg.arm_only:
            lo[:2] = hi[:2] = 0.0
        for _ in range(cfg.iterations):
            eps = self._noise()
            Us = np.clip(self.U[None] + eps, lo, hi)
            if cfg.include_zero:
                Us[0] = 0.0
                Us[1] = np.clip(self.U, lo, hi)
            X = m.rollout(x0, Us)
            J = cost(X, Us, u_prev)
            Jn = (J - J.min()) / max(J.std(), 1e-9)
            wts = np.exp(-Jn / cfg.temperature)
            wts /= wts.sum()
            self.U = np.clip((wts[:, None, None] * Us).sum(0), lo, hi)
            if cfg.keep_best:
                b = int(np.argmin(J))
                j_avg = cost(m.rollout(x0, self.U[None]), self.U[None], u_prev)[0]
                if J[b] < j_avg:
                    self.U = Us[b].copy()
        if cfg.refine_iters > 0 and self._interp is not None:
            self._refine(x0, cost, u_prev, lo, hi)
        X = m.rollout(x0, self.U[None])
        self.last_cost = float(cost(X, self.U[None], u_prev)[0])
        self.last_X = X[0]
        return self.U[0].copy()

    def _refine(self, x0: np.ndarray, cost: WholeBodyCost, u_prev: Optional[np.ndarray],
                lo: np.ndarray, hi: np.ndarray) -> None:
        """Gradient descent on knot offsets Z (knots, 8): U = clip(U + A Z)."""
        cfg, m, A = self.cfg, self.m, self._interp
        k = A.shape[1]
        dims = [d for d in range(8) if not (cfg.arm_only and d < 2)]
        n = k * len(dims)
        E = np.zeros((n, k, 8))
        for j, (i, d) in enumerate((i, d) for i in range(k) for d in dims):
            E[j, i, d] = cfg.refine_eps
        steps = np.geomspace(1e-3, 1.0, 10) * cfg.refine_max_step

        def J(Z: np.ndarray) -> np.ndarray:
            Us = np.clip(self.U[None] + np.einsum("hk,bkd->bhd", A, Z), lo, hi)
            return cost(m.rollout(x0, Us), Us, u_prev)

        j0 = float(J(np.zeros((1, k, 8)))[0])
        for _ in range(cfg.refine_iters):
            Jpm = J(np.concatenate([E, -E]))
            g = np.einsum("n,nkd->kd", (Jpm[:n] - Jpm[n:]) / (2 * cfg.refine_eps), E / cfg.refine_eps)
            gmax = np.abs(g).max()
            if not np.isfinite(gmax) or gmax < 1e-12:
                return
            cand = -steps[:, None, None] * (g / gmax)[None]
            Jc = J(cand)
            b = int(np.argmin(Jc))
            if not Jc[b] < j0:
                return
            self.U = np.clip(self.U + A @ cand[b], lo, hi)
            j0 = float(Jc[b])

    def shift(self) -> None:
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]
