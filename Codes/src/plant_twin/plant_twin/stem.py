"""Stem model: 3D centreline curve with circular thickness.

Parameters: K control points (K×3, flattened). The base control point is
anchored (soft, strongly weighted) to a fixed world point; the tip is
attached to the leaf's petiole point supplied each frame.

Residuals:
  data        — tracked skeleton/cloud points to nearest curve sample.
  base        — ctrl[0] vs anchor.
  tip         — ctrl[-1] vs leaf attachment point.
  length      — total polyline length vs rest length.
  smooth      — second differences of control points (bending energy).
  temporal    — ctrl vs previous frame.
  stationary  — ctrl vs rest shape, only while no pull is happening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .geometry import catmull_rom_basis, polyline_length


@dataclass
class StemWeights:
    data: float = 1.0
    base: float = 50.0
    tip: float = 10.0
    length: float = 20.0
    smooth: float = 5.0
    temporal: float = 2.0
    stationary: float = 5.0


@dataclass
class StemModel:
    rest_ctrl: np.ndarray                     # (K, 3) initial centreline
    radius_m: float = 0.003
    samples_per_seg: int = 8
    weights: StemWeights = field(default_factory=StemWeights)

    def __post_init__(self) -> None:
        self.rest_ctrl = np.asarray(self.rest_ctrl, dtype=float).reshape(-1, 3)
        self.K = len(self.rest_ctrl)
        self.n_params = 3 * self.K
        self.anchor = self.rest_ctrl[0].copy()
        self._basis = catmull_rom_basis(self.K, self.samples_per_seg)
        self.rest_length = polyline_length(self.curve(self.rest_ctrl.ravel()))

    def initial_params(self) -> np.ndarray:
        return self.rest_ctrl.ravel().copy()

    def ctrl(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=float).reshape(self.K, 3)

    def curve(self, params: np.ndarray) -> np.ndarray:
        return self._basis @ self.ctrl(params)

    def residuals(
        self,
        params: np.ndarray,
        observed: np.ndarray,
        tip_target: Optional[np.ndarray] = None,
        prev_params: Optional[np.ndarray] = None,
        pulling: bool = False,
    ) -> np.ndarray:
        w = self.weights
        c = self.ctrl(params)
        curve = self.curve(params)
        blocks = []

        observed = np.asarray(observed, dtype=float).reshape(-1, 3)
        if len(observed):
            _, nn = cKDTree(curve).query(observed)
            blocks.append(w.data * (observed - curve[nn]).ravel())

        blocks.append(w.base * (c[0] - self.anchor))
        if tip_target is not None:
            blocks.append(w.tip * (c[-1] - np.asarray(tip_target, dtype=float)))

        blocks.append(np.atleast_1d(w.length * (polyline_length(curve) - self.rest_length)))

        if self.K >= 3:
            blocks.append(w.smooth * (c[2:] - 2 * c[1:-1] + c[:-2]).ravel())

        if prev_params is not None:
            blocks.append(w.temporal * (params - prev_params))

        if not pulling:
            blocks.append(w.stationary * (params - self.rest_ctrl.ravel()))

        return np.concatenate(blocks)

    # -------------------------------------------------------------- jacobian
    def jacobian(
        self,
        params: np.ndarray,
        observed: np.ndarray,
        tip_target: Optional[np.ndarray] = None,
        prev_params: Optional[np.ndarray] = None,
        pulling: bool = False,
    ) -> np.ndarray:
        """Analytic Jacobian of ``residuals``. The curve is linear in the
        control points (curve = M @ ctrl), so d curve / d params = M ⊗ I3."""
        w = self.weights
        P = self.n_params
        curve = self.curve(params)
        M = self._basis                                      # (S, K)
        Dc = np.kron(M, np.eye(3))                           # (3S, 3P/3)
        blocks = []

        observed = np.asarray(observed, dtype=float).reshape(-1, 3)
        if len(observed):
            _, nn = cKDTree(curve).query(observed)
            rows = (3 * nn[:, None] + np.arange(3)[None, :]).ravel()
            blocks.append(-w.data * Dc[rows])

        Jb = np.zeros((3, P)); Jb[:, :3] = np.eye(3)
        blocks.append(w.base * Jb)
        if tip_target is not None:
            Jt = np.zeros((3, P)); Jt[:, -3:] = np.eye(3)
            blocks.append(w.tip * Jt)

        seg = np.diff(curve, axis=0)
        u = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-12)
        dL_dcurve = np.zeros_like(curve)
        dL_dcurve[1:] += u
        dL_dcurve[:-1] -= u
        blocks.append(w.length * (dL_dcurve.ravel() @ Dc)[None, :])

        if self.K >= 3:
            S2 = np.zeros((self.K - 2, self.K))
            for i in range(self.K - 2):
                S2[i, i:i + 3] = [1.0, -2.0, 1.0]
            blocks.append(w.smooth * np.kron(S2, np.eye(3)))

        if prev_params is not None:
            blocks.append(w.temporal * np.eye(P))
        if not pulling:
            blocks.append(w.stationary * np.eye(P))

        return np.vstack(blocks)
