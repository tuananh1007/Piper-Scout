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
from typing import Optional, Tuple

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
        self._Dc = np.kron(self._basis, np.eye(3))          # d curve / d params
        S2 = np.zeros((max(self.K - 2, 0), self.K))
        for i in range(self.K - 2):
            S2[i, i:i + 3] = [1.0, -2.0, 1.0]
        self._S2 = np.kron(S2, np.eye(3))                   # second differences
        self.rest_length = polyline_length(self.curve(self.rest_ctrl.ravel()))

    def initial_params(self) -> np.ndarray:
        return self.rest_ctrl.ravel().copy()

    def ctrl(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=float).reshape(self.K, 3)

    def curve(self, params: np.ndarray) -> np.ndarray:
        return self._basis @ self.ctrl(params)

    def evaluate(
        self,
        params: np.ndarray,
        observed: np.ndarray,
        tip_target: Optional[np.ndarray] = None,
        prev_params: Optional[np.ndarray] = None,
        pulling: bool = False,
        want_jac: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Stacked residuals and (optionally) the analytic Jacobian.

        The curve is linear in the control points (curve = M @ ctrl), so
        d curve / d params = M ⊗ I3 and every block but ``length`` is constant.
        """
        w = self.weights
        P = self.n_params
        c = self.ctrl(params)
        curve = self._basis @ c
        Dc = self._Dc if want_jac else None
        r_blocks, J_blocks = [], []

        observed = np.asarray(observed, dtype=float).reshape(-1, 3)
        if len(observed):
            _, nn = cKDTree(curve).query(observed)
            r_blocks.append(w.data * (observed - curve[nn]).ravel())
            if want_jac:
                rows = (3 * nn[:, None] + np.arange(3)[None, :]).ravel()
                J_blocks.append(-w.data * Dc[rows])

        r_blocks.append(w.base * (c[0] - self.anchor))
        if want_jac:
            Jb = np.zeros((3, P)); Jb[:, :3] = np.eye(3)
            J_blocks.append(w.base * Jb)
        if tip_target is not None:
            r_blocks.append(w.tip * (c[-1] - np.asarray(tip_target, dtype=float)))
            if want_jac:
                Jt = np.zeros((3, P)); Jt[:, -3:] = np.eye(3)
                J_blocks.append(w.tip * Jt)

        seg = np.diff(curve, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        r_blocks.append(np.atleast_1d(w.length * (seg_len.sum() - self.rest_length)))
        if want_jac:
            u = seg / np.maximum(seg_len, 1e-12)[:, None]
            dL = np.zeros_like(curve)
            dL[1:] += u
            dL[:-1] -= u
            J_blocks.append(w.length * (dL.ravel() @ Dc)[None, :])

        if self.K >= 3:
            r_blocks.append(w.smooth * (c[2:] - 2 * c[1:-1] + c[:-2]).ravel())
            if want_jac:
                J_blocks.append(w.smooth * self._S2)

        if prev_params is not None:
            r_blocks.append(w.temporal * (params - prev_params))
            if want_jac:
                J_blocks.append(w.temporal * np.eye(P))

        if not pulling:
            r_blocks.append(w.stationary * (params - self.rest_ctrl.ravel()))
            if want_jac:
                J_blocks.append(w.stationary * np.eye(P))

        r = np.concatenate(r_blocks)
        return r, (np.vstack(J_blocks) if want_jac else None)

    def residuals(self, params, observed, tip_target=None, prev_params=None, pulling=False):
        return self.evaluate(params, observed, tip_target, prev_params, pulling, want_jac=False)[0]

    def jacobian(self, params, observed, tip_target=None, prev_params=None, pulling=False):
        return self.evaluate(params, observed, tip_target, prev_params, pulling, want_jac=True)[1]
