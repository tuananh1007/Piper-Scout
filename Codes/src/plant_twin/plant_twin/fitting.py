"""Per-frame fitter: alternates leaf and stem least-squares solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from .leaf import LeafModel
from .solver import levenberg_marquardt
from .stem import StemModel


@dataclass
class FrameObservation:
    leaf_points: np.ndarray                    # (N, 3) world frame
    stem_points: np.ndarray                    # (M, 3) world frame
    contact_point: Optional[np.ndarray] = None  # gripper fingertip if touching
    pulling: bool = False                       # gripper closed and moving


@dataclass
class FitResult:
    leaf_params: np.ndarray
    stem_params: np.ndarray
    leaf_cost: float
    stem_cost: float


class PlantTwinFitter:
    """Tracks one leaf + stem across frames.

    ``solver``:
      'lm'  — own damped Gauss-Newton on the normal equations (default; one
              fused residual+Jacobian evaluation per step, P×P solve).
      'trf' — scipy least_squares with the analytic Jacobian (reference).
      'fd'  — scipy least_squares with finite-difference Jacobian (slow
              reference for debugging the analytic one).
    ``max_iter`` is the per-frame step budget (LM iterations or scipy nfev).
    Each frame is warm-started from a constant-velocity prediction of the
    previous two leaf solutions.
    """

    def __init__(self, leaf: LeafModel, stem: StemModel, max_iter: int = 10,
                 solver: str = "lm", predict: bool = True, ftol: float = 1e-3,
                 max_nfev: Optional[int] = None, analytic_jac: Optional[bool] = None):
        self.leaf, self.stem = leaf, stem
        self.ftol = ftol
        # Backwards-compatible aliases from the first version.
        if max_nfev is not None:
            max_iter = max_nfev
        if analytic_jac is False:
            solver = "fd"
        if solver not in ("lm", "trf", "fd"):
            raise ValueError(f"unknown solver {solver!r}")
        self.max_iter = max_iter
        self.solver = solver
        self.predict = predict
        self._prev_leaf = None
        self.leaf_params = leaf.initial_params()
        self.stem_params = stem.initial_params()
        self._first = True

    # --------------------------------------------------------------- solvers
    def _solve(self, model, x0, args):
        if self.solver == "lm":
            res = levenberg_marquardt(
                lambda x: model.evaluate(x, *args, want_jac=True), x0,
                max_iter=self.max_iter, ftol=self.ftol)
            return res.x, res.cost
        jac = model.jacobian if self.solver == "trf" else "2-point"
        r = least_squares(model.residuals, x0, method="trf", jac=jac,
                          max_nfev=self.max_iter, args=args)
        return r.x, float(r.cost)

    # ------------------------------------------------------------------ step
    def step(self, obs: FrameObservation, outer_iters: int = 1) -> FitResult:
        prev_leaf = None if self._first else self.leaf_params.copy()
        prev_stem = None if self._first else self.stem_params.copy()
        if self.predict and self._prev_leaf is not None:
            self.leaf_params = self.leaf_params + (self.leaf_params - self._prev_leaf)
        self._prev_leaf = prev_leaf
        leaf_cost = stem_cost = 0.0

        for _ in range(outer_iters):
            self.leaf_params, leaf_cost = self._solve(
                self.leaf, self.leaf_params,
                (obs.leaf_points, prev_leaf, obs.contact_point))

            tip = self.leaf.tip_point(self.leaf_params)
            self.stem_params, stem_cost = self._solve(
                self.stem, self.stem_params,
                (obs.stem_points, tip, prev_stem, obs.pulling))

        self._first = False
        return FitResult(self.leaf_params.copy(), self.stem_params.copy(),
                         leaf_cost, stem_cost)

    def export(self) -> Dict[str, np.ndarray]:
        return {
            "leaf_vertices": self.leaf.vertices(self.leaf_params),
            "leaf_faces": self.leaf.faces,
            "stem_curve": self.stem.curve(self.stem_params),
            "stem_radius": np.array(self.stem.radius_m),
            "leaf_colors": self.leaf.colors,          # None until textured
        }
