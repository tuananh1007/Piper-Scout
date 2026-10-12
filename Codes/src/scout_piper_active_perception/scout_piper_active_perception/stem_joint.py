"""Stem joint estimate and its Fisher information (P4.2).

The grasp needs the axis of the stem (or peduncle) at the grasp point: its
position to a couple of millimetres and its direction (the gripper closes
across it). The joint is a cylinder of radius r around a line; the estimate
is a known-direction-free fit of that cylinder to the target-class points.

Parameters, local around the current estimate (p₀, d, r):
  θ = (a₁, a₂, b₁, b₂, δr):  p = p₀ + B a,  d' ∝ d + B b,  r' = r + δr
with B (3×2) a basis orthogonal to d. A surface point x with arc length
s = (x − p₀)·d and outward normal n has residual e = ‖(I − ddᵀ)(x − p₀)‖ − r and
  ∂e/∂θ = [−nᵀB, −s nᵀB, −1].
A depth pixel's error is σ_z along its ray u and one pixel σ_lat = z/f across
it; along the normal that is σ_n² = σ_z²(n·u)² + σ_lat²(1 − (n·u)²). Each point
contributes JᵀJ/σ_n² to its view's information I_v (FisherRF's idea, with the
parameters of the joint instead of a radiance field's). The camera pose of a
view (FK, hand-eye) is off by a translation (σ_t) and a rotation about the
camera centre (σ_rot), the same for all of the view's points, so more points
of one view cannot average it out. Mapped to the joint parameters
(``pose_covariance``: a = Bᵀ(δt − [r]ₓε), b = −Bᵀ[d]ₓε with r the axis point
from the camera) it adds P_v to the view's covariance, and the views fuse as
independent estimates: I = I_prior + Σ_v (I + I_v P_v)⁻¹ I_v.

``predicted_information`` gives the information a candidate camera would add:
the current cylinder's surface sampled on rings, kept where it faces the
camera, lies in the image and depth range, weighted by the expected number of
pixels (area · cos / (z/f)²), by the sensor's thin-structure survival
probability and by the transmittance through the occupancy grid; capped at the
number of points a view contributes to the fit. The view's gain is
½ log det(I + I_view) − ½ log det(I), the expected entropy drop of the joint.

``axis_sigma`` is the worst 1-σ lateral uncertainty of the axis over
±half_len around the grasp point, ``axis_error`` the matching true error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def _basis(d: np.ndarray) -> np.ndarray:
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(d, e1)], 1)                     # (3, 2)


@dataclass
class Cylinder:
    p: np.ndarray            # (3,) axis point at the grasp point
    d: np.ndarray            # (3,) unit direction
    r: float


@dataclass
class NoiseModel:
    sigma0: float = 0.0005
    k2: float = 0.002
    f_px: float = 223.0      # focal length of the depth image used
    thin_lo_px: float = 2.0
    thin_hi_px: float = 6.0
    pose_m: float = 0.0005               # per-view camera translation error (1 σ)
    pose_rad: float = float(np.radians(0.1))   # per-view camera rotation error (1 σ)

    def pose2(self, z):
        return self.pose_m ** 2 + (np.asarray(z) * self.pose_rad) ** 2

    def sigma_n2(self, z, cos_nu):
        sz = self.sigma0 + self.k2 * z ** 2
        sl = z / self.f_px
        c2 = cos_nu ** 2
        return sz ** 2 * c2 + sl ** 2 * (1.0 - c2)

    def survival(self, r, z):
        w = 2.0 * r * self.f_px / np.maximum(z, 1e-6)
        return np.clip((w - self.thin_lo_px) / (self.thin_hi_px - self.thin_lo_px), 0.0, 1.0)


def point_jacobian(cyl: Cylinder, x: np.ndarray):
    """Residuals (N,), Jacobians (N, 5), normals (N, 3) for points x (N, 3)."""
    B = _basis(cyl.d)
    rel = x - cyl.p
    s = rel @ cyl.d
    radial = rel - s[:, None] * cyl.d
    rho = np.linalg.norm(radial, axis=1)
    n = radial / np.maximum(rho, 1e-9)[:, None]
    nB = n @ B                                                    # (N, 2)
    J = np.concatenate([-nB, -s[:, None] * nB, -np.ones((len(x), 1))], 1)
    return rho - cyl.r, J, n


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def pose_covariance(cyl: Cylinder, cam: np.ndarray, noise: NoiseModel) -> np.ndarray:
    """Covariance (5, 5) of the joint parameters caused by one view's camera
    pose error (translation σ_t, rotation σ_rot about the camera centre)."""
    B = _basis(cyl.d)
    r = cyl.p - np.asarray(cam, float)
    G = np.zeros((5, 6))
    G[0:2, 0:3] = B.T
    G[0:2, 3:6] = -B.T @ _skew(r)
    G[2:4, 3:6] = -B.T @ _skew(cyl.d)
    S = np.diag([noise.pose_m ** 2] * 3 + [noise.pose_rad ** 2] * 3)
    return G @ S @ G.T


def view_information(I_v: np.ndarray, P_v: np.ndarray) -> np.ndarray:
    """(I_v⁻¹ + P_v)⁻¹ = (I + I_v P_v)⁻¹ I_v, also for a singular I_v."""
    M = np.linalg.solve(np.eye(5) + I_v @ P_v, I_v)
    return 0.5 * (M + M.T)


def information(cyl: Cylinder, x: np.ndarray, cam: np.ndarray, noise: NoiseModel, weights=None) -> np.ndarray:
    """Σ w JᵀJ/σ_n² of points x seen from camera centres ``cam`` ((3,) or
    (N, 3)), without the pose term."""
    if not len(x):
        return np.zeros((5, 5))
    _, J, n = point_jacobian(cyl, x)
    ray = x - cam
    z = np.linalg.norm(ray, axis=-1)
    u = ray / z[:, None]
    w = 1.0 / noise.sigma_n2(z, np.abs((n * u).sum(1)))
    if weights is not None:
        w = w * weights
    return (J * w[:, None]).T @ J


def prior_information(sigma_p: float = 0.01, sigma_d: float = 0.3, sigma_r: float = 0.003) -> np.ndarray:
    return np.diag([sigma_p ** -2, sigma_p ** -2, sigma_d ** -2, sigma_d ** -2, sigma_r ** -2])


def axis_sigma(cov: np.ndarray, half_len: float) -> float:
    """Worst 1-σ lateral axis uncertainty over arc lengths ±half_len."""
    worst = 0.0
    for s in (-half_len, 0.0, half_len):
        A = np.concatenate([np.eye(2), s * np.eye(2)], 1)          # lateral offset at s
        worst = max(worst, float(np.sqrt(max(np.linalg.eigvalsh(A @ cov[:4, :4] @ A.T).max(), 0.0))))
    return worst


def axis_error(cyl: Cylinder, p_true: np.ndarray, d_true: np.ndarray, half_len: float) -> float:
    """Max distance from the true axis to the estimated one over ±half_len."""
    d_true = d_true / np.linalg.norm(d_true)
    err = 0.0
    for s in np.linspace(-half_len, half_len, 5):
        q = p_true + s * d_true
        rel = q - cyl.p
        err = max(err, float(np.linalg.norm(rel - (rel @ cyl.d) * cyl.d)))
    return err


class StemJointEstimator:
    """Accumulates target-class points with the camera they were seen from and
    fits the joint cylinder (Gauss-Newton with a prior on the radius)."""

    def __init__(self, target_guess: np.ndarray, half_len: float = 0.03, r0: float = 0.004,
                 noise: Optional[NoiseModel] = None, max_points_per_view: int = 400,
                 sigma_p: float = 0.01, sigma_d: float = 0.3, sigma_r: float = 0.003, direction_guess=None,
                 rng: Optional[np.random.Generator] = None):
        d0 = np.array([0.0, 0.0, 1.0]) if direction_guess is None else np.asarray(direction_guess, float)
        self.cyl = Cylinder(np.asarray(target_guess, float).copy(), d0 / np.linalg.norm(d0), float(r0))
        self.r0, self.sigma_r = float(r0), float(sigma_r)
        self.half_len = float(half_len)
        self.noise = noise or NoiseModel()
        self.max_pts = int(max_points_per_view)
        self.prior = prior_information(sigma_p, sigma_d, sigma_r)
        self.pts = np.zeros((0, 3))
        self.cams = np.zeros((0, 3))
        self.view = np.zeros(0, np.int64)
        self.n_views = 0
        self.fitted = False
        self.rng = rng or np.random.default_rng(0)

    # ------------------------------------------------------------ data
    def add_view(self, points: np.ndarray, cam_center: np.ndarray) -> int:
        """Target points (N, 3) of one view; keeps those within 1.5·half_len of
        the current grasp point along the axis (all of them before the first
        fit), at most ``max_points_per_view``."""
        P = np.asarray(points, float).reshape(-1, 3)
        if self.fitted and len(P):
            P = P[np.abs((P - self.cyl.p) @ self.cyl.d) <= 1.5 * self.half_len]
        if len(P) > self.max_pts:
            P = P[self.rng.choice(len(P), self.max_pts, replace=False)]
        if len(P):
            self.pts = np.concatenate([self.pts, P])
            self.cams = np.concatenate([self.cams, np.broadcast_to(cam_center, P.shape)])
            self.view = np.concatenate([self.view, np.full(len(P), self.n_views)])
        self.n_views += 1
        return len(P)



    def _init_from_points(self) -> None:
        """Direction from the points' principal axis when they are elongated,
        axis about r behind their centroid (they lie on the camera-facing
        half), grasp point kept where the guess projects onto the axis."""
        P, guess = self.pts, self.cyl.p.copy()
        c = P.mean(0)
        if len(P) >= 10:
            w, V = np.linalg.eigh(np.cov((P - c).T))
            if w[-1] > 4.0 * w[-2]:
                self.cyl.d = V[:, -1] * (np.sign(V[2, -1]) or 1.0)
        d = self.cyl.d
        view = c - self.cams.mean(0)
        view -= (view @ d) * d
        nv = np.linalg.norm(view)
        c = c + (self.r0 * view / nv if nv > 1e-9 else 0.0)
        self.cyl.p = c + ((guess - c) @ d) * d

    def fit(self, iters: int = 15, trim_sigma: float = 3.0) -> Cylinder:
        if len(self.pts) < 8:
            return self.cyl
        if not self.fitted:
            self._init_from_points()
        keep = np.ones(len(self.pts), bool)
        for it in range(iters):
            e, J, n = point_jacobian(self.cyl, self.pts)
            ray = self.pts - self.cams
            z = np.linalg.norm(ray, axis=1)
            s2 = self.noise.sigma_n2(z, np.abs((n * ray).sum(1)) / z)
            pose2 = self.noise.pose2(z)
            if it >= 3:
                keep = np.abs(e) < trim_sigma * np.sqrt(s2 + pose2) + 0.001
            w = keep / (s2 + pose2)
            H = (J * w[:, None]).T @ J
            g = (J * w[:, None]).T @ e
            H[4, 4] += self.sigma_r ** -2
            g[4] += (self.cyl.r - self.r0) * self.sigma_r ** -2
            H += 1e-9 * np.eye(5)
            step = -np.linalg.solve(H + 1e-6 * np.diag(np.diag(H)), g)
            B = _basis(self.cyl.d)
            d = self.cyl.d + B @ step[2:4]
            self.cyl = Cylinder(self.cyl.p + B @ step[:2], d / np.linalg.norm(d),
                                float(np.clip(self.cyl.r + step[4], 0.0005, 0.03)))
            if np.abs(step).max() < 1e-6:
                break
        self.keep = keep
        self.fitted = True
        return self.cyl

    # ------------------------------------------------------- information
    def information(self) -> np.ndarray:
        I = self.prior.copy()
        if self.fitted and len(self.pts):
            k = getattr(self, "keep", np.ones(len(self.pts), bool))
            if len(k) != len(self.pts):
                k = np.ones(len(self.pts), bool)
            for v in np.unique(self.view[k]):
                sel = k & (self.view == v)
                I_v = information(self.cyl, self.pts[sel], self.cams[sel], self.noise)
                I += view_information(I_v, pose_covariance(self.cyl, self.cams[sel][0], self.noise))
        return I

    def covariance(self) -> np.ndarray:
        return np.linalg.inv(self.information())

    def sigma(self) -> float:
        return axis_sigma(self.covariance(), self.half_len)

    def predicted_information(self, T_world_cam: np.ndarray, K: np.ndarray, image_hw, **kw) -> np.ndarray:
        return self.predicted_view(T_world_cam, K, image_hw, **kw)[0]

    def predicted_view(self, T_world_cam: np.ndarray, K: np.ndarray, image_hw, near_m: float = 0.07,
                       far_m: float = 1.0, transmittance: Optional[Callable] = None,
                       n_s: int = 9, n_phi: int = 16):
        """(information (5, 5), expected target pixels with a depth return)."""
        cyl = self.cyl
        B = _basis(cyl.d)
        s = np.linspace(-self.half_len, self.half_len, n_s)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        n = (np.cos(phi)[:, None] * B[:, 0] + np.sin(phi)[:, None] * B[:, 1])          # (n_phi, 3)
        X = cyl.p + s[:, None, None] * cyl.d + cyl.r * n[None]                         # (n_s, n_phi, 3)
        N = np.broadcast_to(n[None], X.shape).reshape(-1, 3)
        X = X.reshape(-1, 3)
        c = T_world_cam[:3, 3]
        ray = X - c
        dist = np.linalg.norm(ray, axis=1)
        u = ray / dist[:, None]
        cos_view = -(N * u).sum(1)
        pc = (X - c) @ T_world_cam[:3, :3]
        z = pc[:, 2]
        H, W = image_hw
        with np.errstate(divide="ignore", invalid="ignore"):
            px = K[0, 0] * pc[:, 0] / z + K[0, 2]
            py = K[1, 1] * pc[:, 1] / z + K[1, 2]
        ok = (cos_view > 0.05) & (z > near_m) & (z < far_m) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not ok.any():
            return np.zeros((5, 5)), 0.0
        area = (2 * self.half_len / n_s) * (2 * np.pi * cyl.r / n_phi)
        pix = area * cos_view[ok] * (K[0, 0] / z[ok]) ** 2
        pix *= self.noise.survival(cyl.r, z[ok])
        if transmittance is not None:
            pix *= transmittance(np.broadcast_to(c, X[ok].shape), X[ok])
        total = float(pix.sum())
        if total > self.max_pts:
            pix = pix * (self.max_pts / total)
        I_v = information(cyl, X[ok], c, self.noise, weights=pix)
        return view_information(I_v, pose_covariance(cyl, c, self.noise)), total

    def fisher_gain(self, I_view: np.ndarray, I_now: Optional[np.ndarray] = None) -> float:
        I0 = self.information() if I_now is None else I_now
        return 0.5 * float(np.linalg.slogdet(I0 + I_view)[1] - np.linalg.slogdet(I0)[1])

    def target_exclusion(self, margin_m: float = 0.015) -> Callable[[np.ndarray], np.ndarray]:
        """Points within r + margin of the joint axis over ±1.5·half_len: the
        target's own voxels, which must not occlude the target."""
        cyl, L = self.cyl, 1.5 * self.half_len

        def fn(P):
            rel = P - cyl.p
            s = rel @ cyl.d
            rho = np.linalg.norm(rel - s[:, None] * cyl.d, axis=1)
            return (np.abs(s) <= L) & (rho <= cyl.r + margin_m)
        return fn
