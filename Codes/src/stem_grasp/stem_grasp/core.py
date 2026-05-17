"""Visual servo control core — ROS 2 Humble port skeleton.

Ports stem_grasp_ros1/src/stem_grasp_ros1/core.py. The math here is identical
to the ROS 1 version — only the ROS plumbing changed. In Phase 2 of the
ROADMAP this whole module is replaced by an MPPI-VS controller.

Classes provided (skeleton):
    StemVelocityObserver        Kalman filter on pixel velocity (sway estimator)
    OnlineJacobianEstimator     EMA-blended image Jacobian
    FullAdaptiveServoController Exponential-gain IBVS with proximity damping

TODO(port-from-ros1): copy class bodies verbatim from ROS 1 core.py — they
have no rospy dependencies, so the port is mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ServoStepResult:
    velocity: np.ndarray
    diagnostics: dict


class StemVelocityObserver:
    """Kalman filter on stem pixel velocity. TODO(port-from-ros1)."""


class OnlineJacobianEstimator:
    """EMA-blended image Jacobian estimator. TODO(port-from-ros1)."""


class FullAdaptiveServoController:
    """Exponential-gain IBVS with proximity damping. TODO(port-from-ros1).

    Phase 2 replaces this with MPPI-VS; preserve the same `step()` signature
    so the pipeline node doesn't need to change.
    """

    def step(
        self,
        raw_uv: np.ndarray,
        desired_uv: np.ndarray,
        depth_z: float,
        force_n: float,
        commanded_vel: np.ndarray | None,
    ) -> ServoStepResult:
        return ServoStepResult(velocity=np.zeros(3), diagnostics={})
