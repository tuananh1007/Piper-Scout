"""Per-frame fitter: alternates leaf and stem least-squares solves.

``fit_quality`` scores each frame's fit from the data alone (point-to-plane
distance of the leaf points to the mesh, point-to-curve distance of the stem
points to the centreline minus its radius): RMS, inlier fraction and a
confidence in [0, 1] that downstream users (active perception's U_twin, the
operator GUI) read from /plant_twin/state."""

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
    quality: Optional[Dict[str, float]] = None   # fit_quality of this frame


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _point_to_polyline(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    a, b = C[:-1], C[1:]
    ab = b - a
    t = np.clip(((P[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)[None], 0, 1)
    d = np.linalg.norm(P[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1)
    return d.min(1)


def fit_quality(leaf_vertices: np.ndarray, leaf_faces: np.ndarray, stem_curve: np.ndarray, stem_radius: float,
                leaf_points: np.ndarray, stem_points: np.ndarray, inlier_m: float = 0.005,
                scale_m: float = 0.005, min_points: int = 20) -> Dict[str, float]:
    """Data-only quality of a fit: per part RMS distance (m) and inlier
    fraction (|d| < ``inlier_m``); confidence = worse inlier fraction ×
    exp(−worse RMS / ``scale_m``); valid when confidence ≥ 0.5 with at least
    ``min_points`` points per part."""
    from scipy.spatial import cKDTree  # noqa: PLC0415
    out = {"n_leaf": int(len(leaf_points)), "n_stem": int(len(stem_points))}
    if len(leaf_points):
        nrm = _vertex_normals(leaf_vertices, leaf_faces)
        _, i = cKDTree(leaf_vertices).query(leaf_points)
        dl = np.abs(((leaf_points - leaf_vertices[i]) * nrm[i]).sum(1))
    else:
        dl = np.zeros(0)
    ds = np.abs(_point_to_polyline(stem_points, stem_curve) - stem_radius) if len(stem_points) else np.zeros(0)
    for name, d in (("leaf", dl), ("stem", ds)):
        out[f"{name}_rms_m"] = float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan")
        out[f"{name}_inlier"] = float(np.mean(d < inlier_m)) if len(d) else 0.0
    rms = max(out["leaf_rms_m"], out["stem_rms_m"])
    conf = min(out["leaf_inlier"], out["stem_inlier"]) * float(np.exp(-rms / scale_m)) if np.isfinite(rms) else 0.0
    out["confidence"] = float(np.clip(conf, 0.0, 1.0))
    out["valid"] = bool(out["confidence"] >= 0.5 and len(dl) >= min_points and len(ds) >= min_points)
    return out


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
        q = fit_quality(self.leaf.vertices(self.leaf_params), self.leaf.faces, self.stem.curve(self.stem_params),
                        float(self.stem.radius_m), np.asarray(obs.leaf_points, float),
                        np.asarray(obs.stem_points, float))
        return FitResult(self.leaf_params.copy(), self.stem_params.copy(), leaf_cost, stem_cost, q)

    def export(self) -> Dict[str, np.ndarray]:
        return {
            "leaf_vertices": self.leaf.vertices(self.leaf_params),
            "leaf_faces": self.leaf.faces,
            "stem_curve": self.stem.curve(self.stem_params),
            "stem_radius": np.array(self.stem.radius_m),
            "leaf_colors": self.leaf.colors,          # None until textured
        }
