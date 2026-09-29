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

from .geometry import catmull_rom, polyline_length


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
        self.rest_length = polyline_length(self.curve(self.rest_ctrl.ravel()))

    def initial_params(self) -> np.ndarray:
        return self.rest_ctrl.ravel().copy()

    def ctrl(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=float).reshape(self.K, 3)

    def curve(self, params: np.ndarray) -> np.ndarray:
        return catmull_rom(self.ctrl(params), self.samples_per_seg)

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
