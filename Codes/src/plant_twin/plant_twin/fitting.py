"""Per-frame fitter: alternates leaf and stem least-squares solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from .leaf import LeafModel
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
    def __init__(self, leaf: LeafModel, stem: StemModel, max_nfev: int = 10,
                 analytic_jac: bool = True, predict: bool = True):
        self.leaf, self.stem = leaf, stem
        # Tracking budget: ~10 Gauss-Newton evaluations per frame, warm-started
        # from a constant-velocity prediction of the previous two solutions.
        self.max_nfev = max_nfev
        self.predict = predict
        self._prev_leaf = None
        # Analytic Jacobians hold ICP correspondences fixed per evaluation;
        # numerical ('2-point') is kept as a reference / fallback.
        self.leaf_jac = leaf.jacobian if analytic_jac else "2-point"
        self.stem_jac = stem.jacobian if analytic_jac else "2-point"
        self.leaf_params = leaf.initial_params()
        self.stem_params = stem.initial_params()
        self._first = True

    def step(self, obs: FrameObservation, outer_iters: int = 1) -> FitResult:
        prev_leaf = None if self._first else self.leaf_params.copy()
        prev_stem = None if self._first else self.stem_params.copy()
        if self.predict and self._prev_leaf is not None:
            self.leaf_params = self.leaf_params + (self.leaf_params - self._prev_leaf)
        self._prev_leaf = prev_leaf
        leaf_cost = stem_cost = 0.0

        for _ in range(outer_iters):
            r = least_squares(
                self.leaf.residuals, self.leaf_params, method="trf",
                jac=self.leaf_jac, max_nfev=self.max_nfev,
                args=(obs.leaf_points, prev_leaf, obs.contact_point),
            )
            self.leaf_params, leaf_cost = r.x, float(r.cost)

            tip = self.leaf.tip_point(self.leaf_params)
            r = least_squares(
                self.stem.residuals, self.stem_params, method="trf",
                jac=self.stem_jac, max_nfev=self.max_nfev,
                args=(obs.stem_points, tip, prev_stem, obs.pulling),
            )
            self.stem_params, stem_cost = r.x, float(r.cost)

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
