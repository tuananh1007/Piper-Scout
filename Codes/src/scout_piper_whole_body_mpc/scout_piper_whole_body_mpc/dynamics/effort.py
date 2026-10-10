"""Contact force at the gripper from the Piper's joint efforts.

The platform has no wrist force/torque sensor (research/README.md), so force
gates and contact detection use an estimate. In quasi-static motion each joint
effort is

    τ_i = a_i (g_i(q) − (Jᵀ F)_i) + b_i + c_i tanh(q̇_i / v_s) + d_i q̇_i + noise

with g(q) the torque that holds the arm against gravity (link masses and
centres of mass of the URDF, ``INERTIALS``), J the translational Jacobian of
the TCP and F the force the environment applies there. a_i absorbs the
driver's effort scale and sign, b_i an offset, c_i / d_i Coulomb and viscous
friction. ``fit_effort_model`` fits (a, b, c, d) per joint by least squares on
free motion (F = 0; ``calibrate_effort`` records it), and reports the residual
σ_i. Gravity loads neither joint 1 (vertical axis) nor, much, joint 6 (roll
about the gripper axis), so their a_i cannot be fitted; they take the median
gain of the identifiable joints, i.e. the driver is assumed to report all
joint efforts in one unit and sign convention (``identified`` says which were
fitted). ``ContactForceEstimator`` turns the residual into F by weighted least
squares (weights 1/σ²), low-pass filters it, and can be tared just before a
contact phase; ``force_noise`` gives the per-axis 1-σ of the estimate at a
pose, which is what contact thresholds have to clear (the 0.15 N of a real
sensor is far below it).

Contact is assumed at the TCP (the grasp point between the fingers); a force
elsewhere on the hand gives a biased F but still a non-zero residual.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .piper import PiperKinematics

GRAVITY = 9.81

# (frame index in PiperKinematics.link_frames, mass kg, centre of mass in that frame) — from
# _piper_arm.xacro; the gripper base and both fingers are lumped on link6 (fixed joint at its
# origin; fingers at mid opening). ``test_effort`` re-parses the xacro to keep them in sync.
INERTIALS: List[Tuple[int, float, Tuple[float, float, float]]] = [
    (1, 0.71, (0.00032, -0.00041, -0.00348)),
    (2, 1.16, (0.21852, -0.00861, 0.00126)),
    (3, 0.5, (-0.01999, -0.14808, -0.00053)),
    (4, 0.38, (0.00013, -0.00076, -0.00394)),
    (5, 0.383, (0.00018, -0.06104, -0.00227)),
    (6, 0.507, (-0.000164, 0.000072, 0.037709)),          # link6 0.007 + gripper base 0.45 + fingers 2 x 0.025
]


def gravity_torques(kin: PiperKinematics, q: np.ndarray, payload_kg: float = 0.0,
                    payload_com: Sequence[float] = (0.05, 0.0, 0.05),
                    inertials: Sequence = INERTIALS) -> np.ndarray:
    """Joint torques (..., 6) that hold the arm against gravity: ∂V/∂q.

    ``payload_kg`` at ``payload_com`` in link6 (e.g. the D405, ~0.06 kg)."""
    F = kin.link_frames(q)
    tau = np.zeros(np.shape(q)[:-1] + (6,))
    items = list(inertials) + ([(6, payload_kg, tuple(payload_com))] if payload_kg > 0 else [])
    for fi, m, com in items:
        c = F[..., fi, :3, :3] @ np.asarray(com, float) + F[..., fi, :3, 3]
        for j in range(fi):                       # joint j turns frame j+1 about its z
            z, o = F[..., j + 1, :3, 2], F[..., j + 1, :3, 3]
            tau[..., j] += m * GRAVITY * np.cross(z, c - o)[..., 2]
    return tau


@dataclass
class EffortModel:
    """Per-joint τ = a·(g − Jᵀ F) + b + c·tanh(q̇ / v_s) + d·q̇ (see module doc)."""
    a: List[float] = field(default_factory=lambda: [1.0] * 6)
    b: List[float] = field(default_factory=lambda: [0.0] * 6)
    c: List[float] = field(default_factory=lambda: [0.0] * 6)
    d: List[float] = field(default_factory=lambda: [0.0] * 6)
    sigma: List[float] = field(default_factory=lambda: [0.1] * 6)    # residual 1-σ per joint
    v_s: float = 0.05                                                 # rad/s, friction smoothing
    payload_kg: float = 0.0
    payload_com: List[float] = field(default_factory=lambda: [0.05, 0.0, 0.05])
    samples: int = 0                                                  # 0: not calibrated
    identified: List[bool] = field(default_factory=lambda: [False] * 6)   # a_i fitted (gravity-loaded)

    def free_torque(self, kin: PiperKinematics, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Predicted effort without contact (..., 6)."""
        g = gravity_torques(kin, q, self.payload_kg, self.payload_com)
        qd = np.asarray(qd, float)
        return (np.asarray(self.a) * g + np.asarray(self.b) + np.asarray(self.c) * np.tanh(qd / self.v_s)
                + np.asarray(self.d) * qd)

    def torque(self, kin: PiperKinematics, q: np.ndarray, qd: np.ndarray,
               force: Optional[Sequence[float]] = None) -> np.ndarray:
        """Effort with ``force`` (3,) N applied at the TCP (base_link frame)."""
        tau = self.free_torque(kin, q, qd)
        if force is not None:
            J = kin.jacobian(np.asarray(q, float))[..., :3, :]
            tau = tau - np.asarray(self.a) * (np.swapaxes(J, -1, -2) @ np.asarray(force, float))
        return tau

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "EffortModel":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def fit_effort_model(kin: PiperKinematics, Q: np.ndarray, QD: np.ndarray, TAU: np.ndarray,
                     v_s: float = 0.05, payload_kg: float = 0.0, prior_weight: float = 1e-5,
                     min_gravity_spread: float = 0.1) -> EffortModel:
    """Least squares per joint on free-motion samples (N, 6) each. Joints
    whose gravity torque varies by less than ``min_gravity_spread`` N·m (std)
    over the samples get the median gain of the others and only (b, c, d)
    fitted; a small pull toward (1, 0, 0, 0) keeps every solve well posed."""
    Q, QD, TAU = (np.asarray(v, float) for v in (Q, QD, TAU))
    G = gravity_torques(kin, Q, payload_kg)
    m = EffortModel(v_s=v_s, payload_kg=payload_kg, samples=len(Q))
    lam = prior_weight * len(Q)

    def solve(A, y, prior):
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y + lam * prior)

    fric = lambda i: [np.ones(len(Q)), np.tanh(QD[:, i] / v_s), QD[:, i]]    # noqa: E731
    m.identified = [bool(np.std(G[:, i]) >= min_gravity_spread) for i in range(6)]
    for i in (i for i in range(6) if m.identified[i]):
        x = solve(np.column_stack([G[:, i], *fric(i)]), TAU[:, i], np.array([1.0, 0, 0, 0]))
        m.a[i], m.b[i], m.c[i], m.d[i] = (float(v) for v in x)
    shared = float(np.median([m.a[i] for i in range(6) if m.identified[i]])) if any(m.identified) else 1.0
    for i in range(6):
        if not m.identified[i]:
            y = TAU[:, i] - shared * G[:, i]
            m.a[i] = shared
            m.b[i], m.c[i], m.d[i] = (float(v) for v in solve(np.column_stack(fric(i)), y, np.zeros(3)))
    for i in range(6):
        A = np.column_stack([G[:, i], *fric(i)])
        m.sigma[i] = float(max(np.std(TAU[:, i] - A @ np.array([m.a[i], m.b[i], m.c[i], m.d[i]])), 1e-4))
    return m


def effort_excitation(q0: np.ndarray, t: float, duration: float,
                      amplitude: Sequence[float] = (0.4, 0.35, 0.35, 0.8, 0.6, 0.8),
                      period: Sequence[float] = (41.0, 23.0, 29.0, 37.0, 19.0, 31.0),
                      ramp: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calibration motion: (q_des, q̇_des) at time t of a slow multi-sine
    around q0. Incommensurate periods spread the poses (and so the gravity
    torques) over the run; the amplitude ramps up from and back down to 0 over
    ``ramp`` s, so the arm starts and ends at q0."""
    A, w = np.asarray(amplitude, float), 2 * np.pi / np.asarray(period, float)
    t = float(np.clip(t, 0.0, duration))
    r = min(ramp, duration / 2)
    if t < r:
        env, denv = 0.5 - 0.5 * np.cos(np.pi * t / r), 0.5 * np.pi / r * np.sin(np.pi * t / r)
    elif t > duration - r:
        u = (duration - t) / r
        env, denv = 0.5 - 0.5 * np.cos(np.pi * u), -0.5 * np.pi / r * np.sin(np.pi * u)
    else:
        env, denv = 1.0, 0.0
    s, c = np.sin(w * t), np.cos(w * t)
    return np.asarray(q0, float) + env * A * s, denv * A * s + env * A * w * c


def joint_move(q_start: np.ndarray, q_goal: np.ndarray, t: float, speed: float = 0.15
               ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Straight joint-space move with a cosine speed profile peaking at
    ``speed`` rad/s on the joint that moves most: (q, q̇, total time)."""
    q_start, dq = np.asarray(q_start, float), np.asarray(q_goal, float) - np.asarray(q_start, float)
    T = max(0.5 * np.pi * float(np.abs(dq).max()) / speed, 1e-6)
    u = float(np.clip(t / T, 0.0, 1.0))
    return (q_start + dq * (0.5 - 0.5 * np.cos(np.pi * u)),
            dq * 0.5 * np.pi / T * np.sin(np.pi * u) if 0.0 < t < T else np.zeros_like(dq), T)


def motion_clearance(Q: np.ndarray, kin: Optional[PiperKinematics] = None, deck_z: float = 0.18,
                     footprint: Tuple[float, float] = (0.47, 0.35), limit_margin: float = 0.15) -> dict:
    """Joint limits (``limit_margin`` inside them: moveit_servo halts 0.1 rad
    before a limit, servo.yaml joint_limit_margin) and the clearance of the distal arm spheres (elbow onward,
    WholeBodyModel.arm_spheres) over joint positions Q (N, 6): above the
    Scout's top plate (``deck_z`` over the ``footprint`` half-extents, base_link
    frame; the arm mount plate is at 0.18 m) and above the base_link plane
    elsewhere. The mount itself and nearby objects are not modelled: the area
    around the robot must be clear."""
    from .whole_body import WholeBodyModel  # noqa: PLC0415
    model = WholeBodyModel(kin=kin or PiperKinematics())
    Q = np.atleast_2d(np.asarray(Q, float))
    C, r = model.collision_spheres(np.concatenate([np.zeros((len(Q), 3)), Q], axis=1))
    distal = [i for i, (fi, _) in enumerate(model.arm_spheres) if fi >= 3]
    c, rd = C[:, distal], r[distal]
    bottom = c[..., 2] - rd
    over = (np.abs(c[..., 0]) <= footprint[0] + rd) & (np.abs(c[..., 1]) <= footprint[1] + rd)
    margin = np.minimum(Q - model.kin.lower, model.kin.upper - Q).min()
    return {"within_limits": bool(margin >= limit_margin), "min_limit_margin_rad": float(margin),
            "min_deck_clearance_m": float(np.where(over, bottom - deck_z, np.inf).min()),
            "min_height_m": float(bottom.min())}


def excitation_check(q0: np.ndarray, duration: float, kin: Optional[PiperKinematics] = None,
                     step: float = 0.25, **kw) -> dict:
    """``motion_clearance`` of the calibration motion around q0, its peak
    joint speed and the spread (std) of the gravity torque per joint."""
    ts = np.arange(0.0, duration + 1e-9, step)
    Q, QD = (np.array(v) for v in zip(*(effort_excitation(q0, t, duration, **kw) for t in ts)))
    out = motion_clearance(Q, kin)
    out.update({"peak_speed_rad_s": float(np.abs(QD).max()),
                "gravity_spread_nm": np.std(gravity_torques(kin or PiperKinematics(), Q), axis=0).round(3).tolist()})
    return out


@dataclass
class ForceEstimate:
    force: np.ndarray            # (3,) N on the TCP, Scout base_link frame
    magnitude: float
    residual: np.ndarray         # (6,) joint-torque residual after the free-motion model


@dataclass
class ContactForceEstimator:
    kin: PiperKinematics
    model: EffortModel = field(default_factory=EffortModel)
    filter_s: float = 0.1        # low-pass time constant on the residual
    damping: float = 1e-6

    def __post_init__(self) -> None:
        self._r: Optional[np.ndarray] = None
        self._offset = np.zeros(6)

    def _scaled(self, r: np.ndarray):
        """Residual and σ in units of (Jᵀ F) per joint (divide out a_i and its sign)."""
        a = np.asarray(self.model.a, float)
        a = np.where(np.abs(a) > 1e-3, a, 1e-3)
        return -r / a, np.asarray(self.model.sigma, float) / np.abs(a)

    def _solve(self, q: np.ndarray, rhs: np.ndarray, sig: np.ndarray) -> np.ndarray:
        J = self.kin.jacobian(q)[:3]                         # (3, 6) TCP translation
        W = 1.0 / sig ** 2
        return np.linalg.solve((J * W) @ J.T + self.damping * np.eye(3), (J * W) @ rhs)

    def update(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, dt: float) -> ForceEstimate:
        r = np.asarray(tau, float) - self.model.free_torque(self.kin, q, qd)
        alpha = 1.0 if self._r is None else min(max(dt, 0.0) / (self.filter_s + max(dt, 0.0)), 1.0)
        self._r = r if self._r is None else self._r + alpha * (r - self._r)
        rhs, sig = self._scaled(self._r - self._offset)
        F = self._solve(np.asarray(q, float), rhs, sig)
        return ForceEstimate(F, float(np.linalg.norm(F)), self._r - self._offset)

    def tare(self) -> None:
        """Zero the current residual (call with nothing touching the gripper)."""
        if self._r is not None:
            self._offset = self._r.copy()

    def force_noise(self, q: np.ndarray) -> np.ndarray:
        """Per-axis 1-σ (3,) of the force estimate at q from the joint residual σ."""
        _, sig = self._scaled(np.zeros(6))
        J = self.kin.jacobian(np.asarray(q, float))[:3]
        cov = np.linalg.inv((J / sig ** 2) @ J.T + self.damping * np.eye(3))
        return np.sqrt(np.diag(cov))
