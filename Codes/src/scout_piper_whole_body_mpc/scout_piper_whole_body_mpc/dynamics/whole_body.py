"""Whole-body state x = [x_b, y_b, θ_b, q₁..q₆], control u = [v, ω, q̇₁..q̇₆]."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .piper import PiperKinematics
from .scout import ScoutParams, planar_T, rollout_base


@dataclass
class ArmParams:
    qd_max: float = 0.6            # rad/s per joint (hardware allows more; slow near plants)
    qdd_max: float = 2.0           # rad/s²


@dataclass
class WholeBodyModel:
    dt: float = 0.1
    scout: ScoutParams = field(default_factory=ScoutParams)
    arm: ArmParams = field(default_factory=ArmParams)
    kin: PiperKinematics = field(default_factory=PiperKinematics)
    # collision spheres: (frame index in link_frames, radius); 0 = arm base
    arm_spheres: tuple = ((2, 0.05), (3, 0.045), (4, 0.04), (6, 0.04), (7, 0.03))
    # Scout body as spheres in base_link (x, y, z, r)
    base_spheres: tuple = ((0.25, 0.0, 0.15, 0.25), (-0.25, 0.0, 0.15, 0.25))

    @property
    def u_low(self) -> np.ndarray:
        return np.r_[-self.scout.v_max, -self.scout.omega_max, -self.arm.qd_max * np.ones(6)]

    @property
    def u_high(self) -> np.ndarray:
        return -self.u_low

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """x0 (9,), U (B, H, 8) -> X (B, H+1, 9). Arm is clamped to joint limits."""
        U = np.asarray(U, float)
        B, H, _ = U.shape
        X = np.empty((B, H + 1, 9))
        X[..., :3] = rollout_base(x0[:3], U[..., :2], self.dt, self.scout)
        q = np.cumsum(U[..., 2:] * self.dt, axis=1) + x0[3:]
        X[:, 0, 3:] = x0[3:]
        X[:, 1:, 3:] = np.clip(q, self.kin.lower, self.kin.upper)
        return X

    def world_frames(self, X: np.ndarray) -> np.ndarray:
        """X (..., 9) -> link frames in world (..., 8, 4, 4)."""
        Tb = planar_T(X[..., :3])
        return Tb[..., None, :, :] @ self.kin.link_frames(X[..., 3:])

    def tcp_world(self, X: np.ndarray) -> np.ndarray:
        return planar_T(X[..., :3]) @ self.kin.tcp(X[..., 3:])

    def collision_spheres(self, X: np.ndarray):
        """World centres (..., S, 3) and radii (S,) for arm + base spheres."""
        F = self.world_frames(X)
        arm_c = np.stack([F[..., i, :3, 3] for i, _ in self.arm_spheres], axis=-2)
        Tb = planar_T(X[..., :3])
        bp = np.array([s[:3] for s in self.base_spheres])
        base_c = (Tb[..., None, :3, :3] @ bp[..., None])[..., 0] + Tb[..., None, :3, 3]
        r = np.r_[[r for _, r in self.arm_spheres], [s[3] for s in self.base_spheres]]
        return np.concatenate([arm_c, base_c], axis=-2), r
