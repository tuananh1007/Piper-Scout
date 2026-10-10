"""Phase 2B: safety-bounded local MPPI visual servo (ROADMAP Phase 2B, P2.1–P2.2).

The final-approach controller that replaces the adaptive IBVS of stem_grasp
(``core.FullAdaptiveServoController`` stays as the fallback baseline): the
whole-body MPPI in ``arm_only`` mode (Scout frozen) over the six Piper joint
velocities, with the Phase 2B costs

  image      ρ(‖ŝ_k − s*_k‖) over the horizon plus a terminal weight: ŝ the
             predicted image position of the grasp target, s* where the gripper
             approach axis crosses the target's depth (stem_grasp servo_geometry),
             ρ(e) = √(1 + (e/c)²) − 1 (pseudo-Huber, c = ``image_scale_px``):
             linear far away, quadratic within a few pixels, so the pull does not
             vanish near the goal as a squared normalised error does
  view       quadratic barrier inside ``fov_margin_px`` of the image border and
             a large cost for the target behind the camera (visibility)
  joints     quadratic barrier within 0.05 rad of the joint limits
  manip      1 / manipulability (Yoshikawa, translational)
  clearance  optional ``distance_fn`` (semantic scene), as WholeBodyCost
  smooth     ‖ΔU‖² and continuity with the last command
  force      above ``contact_force_n``, motion that brings the TCP closer to the
             target along the gripper axis costs w_force · (f / f_max) per metre
  distance   optional ``desired_distance_m``: (d_H − d*)² for the stepwise final
             approach (d = target minus TCP along the gripper axis)

and any ``extra`` terms (e.g. a Piper-JEPA visibility cost).

Image prediction (P2.1.2), ``mode``:
  projective  the metric target (from depth, or the last valid estimate) is
              projected into the camera of every planned state (FK, link6 →
              camera transform, pinhole); the measured pixel minus the predicted
              pixel at the current state is kept as a bias over the horizon
              (calibration and target-estimate error)
  jacobian    ŝ_k = s_meas + J_s (q_k − q_0): J_s (2 × 6) is the projective model
              linearised at q_0 (finite differences), or, without a metric
              target, the online Broyden estimate (``ImageJacobianEstimator``)
              learnt from measured (Δs, Δq); s* is then given by the caller

Sampling starts from the better of the shifted last plan and a resolved-rate
nominal (``nominal_plan``): at every planned step, damped least squares on the
TCP Jacobian for a TCP velocity that pulls the TCP axis onto the target line
and moves along it to the desired distance. Random joint-space samples (and a
constant joint velocity, whose Cartesian path curves) rarely produce that
straight-line motion; near the stem, where 1 mm sideways is ~4 px, the final
approach then stalls. MPPI shapes the nominal with every cost above.

Hard limits stay outside the optimiser: ``MppiVisualServo`` passes every command
through the whole-body ``SafetyFilter`` (velocity, acceleration, joint limits,
clearance when a distance field is given), independently of any learned
term; the stem_grasp pipeline keeps its force, mask-age and approach aborts.

The cost has a torch twin (``torch_cost``), so ``TorchMPPI`` samples on CUDA
(P2.1.1: 512 trajectories × 20 steps); ``benchmarks/visual_servo_timing.py``
measures the cycle time against the 10 ms budget (P2.2.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .costs.terms import DistanceFn, ExtraTerm
from .dynamics.piper import PiperKinematics
from .dynamics.whole_body import ArmParams, WholeBodyModel
from .safety.projection import SafetyFilter
from .solvers.mppi import MPPI, MPPIConfig


# ---------------------------------------------------------------- inputs
@dataclass
class ServoTarget:
    """One control step's measurement and geometry, all in the Scout base_link
    frame (the model's world with the base at the origin)."""
    uv_meas: np.ndarray                         # (2,) measured image position of the target
    K: np.ndarray                               # (3, 3) colour intrinsics
    image_hw: Tuple[int, int]
    T_flange_cam: np.ndarray                    # (4, 4) link6 → camera optical frame
    p_base: Optional[np.ndarray] = None         # (3,) metric target; None: jacobian mode only
    uv_desired: Optional[np.ndarray] = None     # (2,) needed without p_base
    force_n: float = 0.0
    desired_distance_m: Optional[float] = None  # final approach: target − TCP along the gripper axis


@dataclass
class VisualServoWeights:
    image: float = 4.0                  # running and terminal equal: a terminal-heavy cost swings
    terminal: float = 4.0
    image_scale_px: float = 5.0
    fov: float = 50.0
    fov_margin_px: float = 20.0
    behind: float = 1e4
    joint_limit: float = 1e3
    manip: float = 0.002
    clearance: float = 1e4
    d_safe: float = 0.02
    smooth: float = 0.5
    force: float = 200.0
    contact_force_n: float = 0.15
    max_force_n: float = 2.0
    distance: float = 2e3


def _project(K, p):
    z = p[..., 2]
    zs = np.where(z > 1e-4, z, 1e-4)
    return np.stack([K[0, 0] * p[..., 0] / zs + K[0, 2], K[1, 1] * p[..., 1] / zs + K[1, 2]], -1), z


# ------------------------------------------------------------------ model
def _manipulability(F: np.ndarray) -> np.ndarray:
    """Yoshikawa measure of the translational Jacobian from link frames F (..., 8, 4, 4)."""
    p = F[..., -1, :3, 3]
    Jv = np.stack([np.cross(F[..., i + 1, :3, 2], p - F[..., i + 1, :3, 3]) for i in range(6)], -1)
    return np.sqrt(np.clip(np.linalg.det(Jv @ np.swapaxes(Jv, -1, -2)), 0, None))


def camera_geometry(model: WholeBodyModel, X: np.ndarray, T_fc: np.ndarray, K: np.ndarray, p_base: np.ndarray,
                    F: Optional[np.ndarray] = None):
    """Per state X (..., 9): target pixel uv, target depth z, desired pixel
    (target on the approach axis), and distance along the axis. ``F``: the
    states' link frames if already computed."""
    F = model.world_frames(X) if F is None else F                  # (..., 8, 4, 4)
    T = F[..., 6, :, :] @ T_fc
    R, t = T[..., :3, :3], T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    pc = (Rt @ (np.asarray(p_base, float) - t)[..., None])[..., 0]
    tcp = (Rt @ (F[..., 7, :3, 3] - t)[..., None])[..., 0]
    axis = (Rt @ F[..., 6, :3, 2][..., None])[..., 0]
    uv, z = _project(K, pc)
    az = np.where(np.abs(axis[..., 2]) > 0.1, axis[..., 2], 0.1)
    pd = tcp + ((pc[..., 2] - tcp[..., 2]) / az)[..., None] * axis
    uv_d, _ = _project(K, pd)
    dist = ((pc - tcp) * axis).sum(-1)
    return uv, z, uv_d, dist


def image_jacobian(model: WholeBodyModel, q: np.ndarray, T_fc: np.ndarray, K: np.ndarray, p_base: np.ndarray,
                   eps: float = 1e-4) -> np.ndarray:
    """d(target pixel)/dq (2 × 6) of the projective model at q (central differences)."""
    X = np.zeros((12, 9))
    X[:, 3:] = q
    for i in range(6):
        X[2 * i, 3 + i] += eps
        X[2 * i + 1, 3 + i] -= eps
    uv = camera_geometry(model, X, T_fc, K, p_base)[0]
    return ((uv[0::2] - uv[1::2]) / (2 * eps)).T


@dataclass
class ImageJacobianEstimator:
    """Online (Broyden) estimate of d(pixel)/dq from measured steps; the
    fallback when no metric target exists (D405 depth gone below ~7 cm)."""
    alpha: float = 0.5
    min_step_rad: float = 1e-4
    J: Optional[np.ndarray] = None
    _last: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def reset(self, J0: Optional[np.ndarray] = None) -> None:
        self.J, self._last = (None if J0 is None else np.array(J0, float)), None

    def update(self, q: np.ndarray, uv: np.ndarray) -> Optional[np.ndarray]:
        q, uv = np.asarray(q, float), np.asarray(uv, float)
        if self._last is not None and self.J is not None:
            dq, ds = q - self._last[0], uv - self._last[1]
            n2 = float(dq @ dq)
            if n2 > self.min_step_rad ** 2:
                self.J = self.J + self.alpha * np.outer(ds - self.J @ dq, dq) / n2
        self._last = (q, uv)
        return self.J


def nominal_velocity(model: WholeBodyModel, q: np.ndarray, p_base: np.ndarray,
                     desired_distance_m: Optional[float], k_lat: float = 2.0, k_dist: float = 1.0,
                     v_lat_max: float = 0.05, v_approach: float = 0.02, damping: float = 1e-3) -> np.ndarray:
    """Joint velocities (6,) for a TCP that keeps its orientation, moves its
    axis onto the target (≤ ``v_lat_max``) and, with ``desired_distance_m``,
    along it (≤ ``v_approach``)."""
    F = model.kin.link_frames(q)
    t, a = F[-1, :3, 3], F[-1, :3, 2]
    r = np.asarray(p_base, float) - t
    d = float(r @ a)
    v = k_lat * (r - d * a)
    n = np.linalg.norm(v)
    if n > v_lat_max:
        v *= v_lat_max / n
    if desired_distance_m is not None:
        v = v + np.clip(k_dist * (d - desired_distance_m), -v_approach, v_approach) * a
    Jm = model.kin.jacobian(q)
    qd = Jm.T @ np.linalg.solve(Jm @ Jm.T + damping * np.eye(6), np.r_[v, 0.0, 0.0, 0.0])
    s = np.abs(qd).max() / model.arm.qd_max
    return qd / s if s > 1 else qd


def nominal_plan(model: WholeBodyModel, q: np.ndarray, p_base: np.ndarray, desired_distance_m: Optional[float],
                 horizon: int, **kw) -> np.ndarray:
    """Resolved-rate rollout (horizon, 8): ``nominal_velocity`` recomputed at each planned joint state."""
    U = np.zeros((horizon, 8))
    q = np.asarray(q, float).copy()
    for k in range(horizon):
        U[k, 2:] = nominal_velocity(model, q, p_base, desired_distance_m, **kw)
        q = np.clip(q + model.dt * U[k, 2:], model.kin.lower, model.kin.upper)
    return U


# ------------------------------------------------------------------- cost
@dataclass
class VisualServoCost:
    model: WholeBodyModel
    target: ServoTarget
    x0: np.ndarray
    w: VisualServoWeights = field(default_factory=VisualServoWeights)
    mode: str = "projective"                    # projective | jacobian
    J_s: Optional[np.ndarray] = None            # jacobian mode without a metric target
    distance_fn: Optional[DistanceFn] = None
    extra: List[ExtraTerm] = field(default_factory=list)

    def __post_init__(self) -> None:
        tg = self.target
        self.K = np.asarray(tg.K, float)
        H, W = tg.image_hw
        self.diag = float(np.hypot(H, W))
        self.q0 = np.asarray(self.x0, float)[3:]
        self.bias = np.zeros(2)
        if tg.p_base is not None:
            uv0, _, uv_d0, d0 = camera_geometry(self.model, np.asarray(self.x0, float), tg.T_flange_cam,
                                                self.K, tg.p_base)
            self.bias = np.asarray(tg.uv_meas, float) - uv0
            self.uv_desired0, self.distance0 = uv_d0, float(d0)
        else:
            if self.mode != "jacobian" or tg.uv_desired is None or self.J_s is None:
                raise ValueError("without a metric target: mode 'jacobian' with uv_desired and J_s")
            self.uv_desired0, self.distance0 = np.asarray(tg.uv_desired, float), None
        if self.mode == "jacobian" and self.J_s is None:
            self.J_s = image_jacobian(self.model, self.q0, tg.T_flange_cam, self.K, tg.p_base)
        self.error_px = float(np.linalg.norm(np.asarray(tg.uv_meas, float) - self.uv_desired0))

    def predict(self, X: np.ndarray, F: Optional[np.ndarray] = None):
        """(uv, depth or None, desired uv, distance or None) for states X (B, H, 9)."""
        tg = self.target
        if self.mode == "projective":
            uv, z, uv_d, dist = camera_geometry(self.model, X, tg.T_flange_cam, self.K, tg.p_base, F)
            return uv + self.bias, z, uv_d, dist
        uv = np.asarray(tg.uv_meas, float) + (X[..., 3:] - self.q0) @ self.J_s.T
        if tg.p_base is not None:
            _, z, uv_d, dist = camera_geometry(self.model, X, tg.T_flange_cam, self.K, tg.p_base, F)
            return uv, z, uv_d, dist
        return uv, None, np.broadcast_to(self.uv_desired0, uv.shape), None

    def terms(self, X: np.ndarray, U: np.ndarray, u_prev: Optional[np.ndarray] = None) -> dict:
        """Per-sample value (B,) of every cost term (diagnostics; ``__call__`` sums them)."""
        w, m, tg = self.w, self.model, self.target
        Xk = X[:, 1:]
        F = m.world_frames(Xk)                                       # once: camera, manipulability
        uv, z, uv_d, dist = self.predict(Xk, F)
        rho = np.sqrt(1.0 + (((uv - uv_d) / w.image_scale_px) ** 2).sum(-1)) - 1.0     # (B, H)
        out = {"image": w.image * rho.mean(1) + w.terminal * rho[:, -1]}
        H_img, W_img = tg.image_hw
        mg = w.fov_margin_px
        edge = np.stack([uv[..., 0], W_img - uv[..., 0], uv[..., 1], H_img - uv[..., 1]], -1)
        out["view"] = w.fov * ((np.clip(mg - edge, 0, None) / mg) ** 2).sum((1, 2))
        if z is not None:
            out["view"] = out["view"] + w.behind * (z < 0.02).sum(1)
        q = Xk[..., 3:]
        lim = np.clip(m.kin.lower + 0.05 - q, 0, None) + np.clip(q - (m.kin.upper - 0.05), 0, None)
        out["joints"] = w.joint_limit * (lim ** 2).sum((1, 2))
        out["manip"] = w.manip * (1.0 / (_manipulability(F) + 1e-3)).mean(1)
        if self.distance_fn is not None:
            C, r = m.collision_spheres(Xk)
            d, _ = self.distance_fn(C.reshape(-1, 3))
            out["clearance"] = w.clearance * (np.clip(w.d_safe - (d.reshape(C.shape[:-1]) - r), 0, None) ** 2).sum((1, 2))
        smooth = w.smooth * (np.diff(U, axis=1) ** 2).sum((1, 2))
        if u_prev is not None:
            smooth = smooth + w.smooth * ((U[:, 0] - u_prev) ** 2).sum(-1)
        out["smooth"] = smooth
        if dist is not None:
            if tg.force_n > w.contact_force_n:
                closer = np.clip(-np.diff(np.concatenate([np.full((len(X), 1), self.distance0), dist], 1), axis=1),
                                 0, None).sum(1)
                out["force"] = w.force * min(tg.force_n / w.max_force_n, 1.0) * closer
            if tg.desired_distance_m is not None:
                out["distance"] = w.distance * (dist[:, -1] - tg.desired_distance_m) ** 2
        for i, term in enumerate(self.extra):
            out[f"extra{i}"] = term(X, U)
        return out

    def __call__(self, X: np.ndarray, U: np.ndarray, u_prev: Optional[np.ndarray] = None) -> np.ndarray:
        return sum(self.terms(X, U, u_prev).values())

    # ------------------------------------------------------------- torch twin
    def torch_cost(self, tm, field=None) -> Callable:
        """The same cost on torch tensors (``TorchMPPI``); ``extra`` terms run in numpy."""
        torch = tm.torch
        w, tg = self.w, self.target
        K = tm.tensor(self.K)
        T_fc = tm.tensor(tg.T_flange_cam)
        p = None if tg.p_base is None else tm.tensor(tg.p_base)
        bias = tm.tensor(self.bias)
        uv_meas = tm.tensor(tg.uv_meas)
        q0 = tm.tensor(self.q0)
        Js = None if self.J_s is None else tm.tensor(self.J_s)
        uvd0 = tm.tensor(self.uv_desired0)
        H_img, W_img = tg.image_hw
        dist_fn = self.distance_fn

        def proj(P):
            zs = torch.clamp(P[..., 2], min=1e-4)
            return torch.stack([K[0, 0] * P[..., 0] / zs + K[0, 2], K[1, 1] * P[..., 1] / zs + K[1, 2]], -1)

        def geometry(Xk):
            Tb = tm.planar_T(Xk[..., :3])
            F = Tb[..., None, :, :] @ tm.link_frames(Xk[..., 3:])
            T = F[..., 6, :, :] @ T_fc
            Rt = T[..., :3, :3].transpose(-1, -2)
            t = T[..., :3, 3]
            pc = (Rt @ (p - t)[..., None])[..., 0]
            tcp = (Rt @ (F[..., 7, :3, 3] - t)[..., None])[..., 0]
            axis = (Rt @ F[..., 6, :3, 2][..., None])[..., 0]
            az = torch.where(axis[..., 2].abs() > 0.1, axis[..., 2], torch.full_like(axis[..., 2], 0.1))
            pd = tcp + ((pc[..., 2] - tcp[..., 2]) / az)[..., None] * axis
            return proj(pc), pc[..., 2], proj(pd), ((pc - tcp) * axis).sum(-1), F

        def cost(X, U, u_prev=None):
            Xk = X[:, 1:]
            q = Xk[..., 3:]
            F = None
            if p is not None:
                uv_p, z, uv_d, dist, F = geometry(Xk)
            if self.mode == "projective":
                uv = uv_p + bias
            else:
                uv = uv_meas + (q - q0) @ Js.T
                if p is None:
                    z = dist = None
                    uv_d = uvd0.expand(uv.shape)
            rho = torch.sqrt(1.0 + (((uv - uv_d) / w.image_scale_px) ** 2).sum(-1)) - 1.0
            J = w.image * rho.mean(1) + w.terminal * rho[:, -1]
            mg = w.fov_margin_px
            edge = torch.stack([uv[..., 0], W_img - uv[..., 0], uv[..., 1], H_img - uv[..., 1]], -1)
            J = J + w.fov * ((torch.clamp(mg - edge, min=0) / mg) ** 2).sum((1, 2))
            if z is not None:
                J = J + w.behind * (z < 0.02).to(J.dtype).sum(1)
            lim = torch.clamp(tm.lower + 0.05 - q, min=0) + torch.clamp(q - (tm.upper - 0.05), min=0)
            J = J + w.joint_limit * (lim ** 2).sum((1, 2))
            if F is None:
                Tb = tm.planar_T(Xk[..., :3])
                F = Tb[..., None, :, :] @ tm.link_frames(q)
            J = J + w.manip * (1.0 / (tm.manipulability(F) + 1e-3)).mean(1)
            if dist_fn is not None or field is not None:
                Fa = F[..., tm.arm_idx, :, :]
                C = Fa[..., :3, 3]
                r = tm.radii[:len(tm.arm_idx)]
                if field is not None:
                    d = field.distance(C.reshape(-1, 3))[0].reshape(C.shape[:-1])
                else:
                    d = tm.tensor(dist_fn(C.reshape(-1, 3).detach().cpu().numpy().astype(float))[0]).reshape(C.shape[:-1])
                J = J + w.clearance * (torch.clamp(w.d_safe - (d - r), min=0) ** 2).sum((1, 2))
            dU = U[:, 1:] - U[:, :-1]
            J = J + w.smooth * (dU ** 2).sum((1, 2))
            if u_prev is not None:
                J = J + w.smooth * ((U[:, 0] - u_prev) ** 2).sum(-1)
            if dist is not None:
                if tg.force_n > w.contact_force_n:
                    d_all = torch.cat([torch.full_like(dist[:, :1], self.distance0), dist], 1)
                    closer = torch.clamp(-(d_all[:, 1:] - d_all[:, :-1]), min=0).sum(1)
                    J = J + w.force * min(tg.force_n / w.max_force_n, 1.0) * closer
                if tg.desired_distance_m is not None:
                    J = J + w.distance * (dist[:, -1] - tg.desired_distance_m) ** 2
            if self.extra:
                Xn, Un = X.detach().cpu().numpy().astype(float), U.detach().cpu().numpy().astype(float)
                for term in self.extra:
                    J = J + tm.tensor(term(Xn, Un))
            return J

        return cost


# -------------------------------------------------------------- controller
@dataclass
class VisualServoConfig:
    dt: float = 0.05                    # control period (one step per new mask)
    horizon: int = 20
    samples: int = 512
    iterations: int = 1
    temperature: float = 0.2
    noise_knots: int = 4                # temporally smooth noise
    noise_arm: float = 0.15             # rad/s sample σ per joint
    refine_iters: int = 1               # gradient step after sampling: the last pixels (sampling
                                        # noise moves the target ~7 px per step)
    qd_max: float = 0.5                 # rad/s, near-contact speed limit
    qdd_max: float = 4.0                # rad/s², enforced by the safety filter (2 lags the plan)
    tcp_offset_m: float = 0.14
    approach_speed_mps: float = 0.02    # nominal TCP speed along the gripper axis (stem_grasp)
    nominal_gain: float = 2.0           # 1/s, nominal pull of the TCP axis onto the target (4 overshoots)
    mode: str = "projective"            # projective | jacobian
    backend: str = "numpy"              # numpy | torch
    device: str = "auto"
    seed: int = 0


class MppiVisualServo:
    """Arm-only MPPI visual servo: ``step(q, target)`` -> joint velocities (6,)."""

    def __init__(self, cfg: VisualServoConfig = VisualServoConfig(), weights: Optional[VisualServoWeights] = None,
                 distance_fn: Optional[DistanceFn] = None):
        self.cfg = cfg
        self.w = weights or VisualServoWeights()
        self.model = WholeBodyModel(dt=cfg.dt, arm=ArmParams(qd_max=cfg.qd_max, qdd_max=cfg.qdd_max),
                                    kin=PiperKinematics(tcp_offset_m=cfg.tcp_offset_m))
        mcfg = MPPIConfig(horizon=cfg.horizon, samples=cfg.samples, iterations=cfg.iterations,
                          temperature=cfg.temperature, noise_knots=cfg.noise_knots, noise_arm=cfg.noise_arm,
                          refine_iters=cfg.refine_iters,
                          arm_only=True, seed=cfg.seed)
        if cfg.backend == "torch":
            from .torch_backend import TorchMPPI, default_device  # noqa: PLC0415 — needs torch
            self.mppi = TorchMPPI(self.model, mcfg, device=default_device() if cfg.device == "auto" else cfg.device)
        else:
            self.mppi = MPPI(self.model, mcfg)
        self.distance_fn = distance_fn
        self.safety = SafetyFilter(self.model, distance_fn=distance_fn)
        self.jac = ImageJacobianEstimator()
        self.u_prev = np.zeros(8)
        self.extra: List[ExtraTerm] = []

    def reset(self) -> None:
        self.mppi.reset()
        self.jac.reset()
        self.u_prev = np.zeros(8)

    def _seed_with_nominal(self, x0: np.ndarray, q: np.ndarray, target: ServoTarget, cost) -> None:
        """Sampling mean := the cheaper of the shifted plan and the resolved-rate nominal."""
        U_nom = nominal_plan(self.model, q, target.p_base, target.desired_distance_m, self.cfg.horizon,
                             k_lat=self.cfg.nominal_gain, v_approach=self.cfg.approach_speed_mps)
        U_now = self.mppi.U
        torch_plan = not isinstance(U_now, np.ndarray)
        U_cur = U_now.detach().cpu().numpy().astype(float) if torch_plan else np.asarray(U_now, float)
        Us = np.stack([U_cur, U_nom])
        J = cost(self.model.rollout(x0, Us), Us, self.u_prev)
        if J[1] < J[0]:
            self.mppi.U = U_now.new_tensor(U_nom) if torch_plan else U_nom
        self._last_nominal = {k: round(float(v[1]), 4) for k, v in cost.terms(self.model.rollout(x0, Us), Us,
                                                                             self.u_prev).items()}

    def step(self, q: np.ndarray, target: ServoTarget, max_staleness_s: float = 0.0):
        """One control step from the measured joints and image target.

        Returns (qdot (6,), diag): diag has the image error, the predicted
        terminal error, the image prediction mode used, solve ms and the
        safety reason."""
        t0 = time.perf_counter()
        q = np.asarray(q, float)
        x0 = np.r_[0.0, 0.0, 0.0, q]
        mode, J_s = self.cfg.mode, None
        if target.p_base is not None:
            # keep the online estimate seeded by the model while a metric target exists
            if self.jac.J is None or mode == "projective":
                self.jac.reset(image_jacobian(self.model, q, target.T_flange_cam, np.asarray(target.K, float),
                                              target.p_base))
            self.jac.update(q, target.uv_meas)
        else:
            mode, J_s = "jacobian", self.jac.update(q, target.uv_meas)
            if J_s is None:
                return np.zeros(6), {"mode": "no_jacobian", "error_px": float("nan"), "safety": "no_model"}
        cost = VisualServoCost(self.model, target, x0, self.w, mode=mode, J_s=J_s,
                               distance_fn=self.distance_fn, extra=list(self.extra))
        if target.p_base is not None:
            self._seed_with_nominal(x0, q, target, cost)
        u_prev = self.u_prev
        u = self.mppi.solve(x0, cost, u_prev)
        U_plan = self.mppi.U
        U_plan = np.array(U_plan.detach().cpu().numpy() if not isinstance(U_plan, np.ndarray) else U_plan, float)
        rep = self.safety.project(x0, u, u_prev, state_age_s=max_staleness_s)
        self.u_prev = rep.u
        self.mppi.shift()
        X = self.model.rollout(x0, U_plan[None])
        uv, _, uv_d, _ = cost.predict(X[:, -1:])
        e_term = float(np.linalg.norm(uv[0, -1] - uv_d[0, -1]))
        plan_terms = {k: round(float(v[0]), 4) for k, v in cost.terms(X, U_plan[None], u_prev).items()}
        return rep.u[2:].copy(), {
            "mode": mode, "error_px": cost.error_px, "predicted_terminal_px": e_term,
            "distance_m": cost.distance0, "solve_ms": 1e3 * (time.perf_counter() - t0),
            "safety": rep.reason, "scale": float(rep.scale), "plan_terms": plan_terms,
            "nominal_terms": getattr(self, "_last_nominal", {})}
