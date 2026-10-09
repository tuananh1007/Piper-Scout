"""Reach-then-servo handoff between stem_grasp and the whole-body MPC (P0.4.11).

ROS 1 planned and executed the move to the pre-grasp pose with MoveIt; on
ROS 2 Humble ``moveit_py`` has no binary, so ``pipeline_node`` can hand the
pre-grasp pose to ``scout_piper_whole_body_mpc`` instead
(``reach_executor: whole_body_mpc``):

    SCANNING --best candidate--> REACHING --close enough for settle_s--> SERVOING
                                     |--timeout / MPC lost--> SCANNING (goal cancelled)

The MPC drives base and arm to the pre-grasp position with the candidate's
approach direction as a soft orientation goal; the pipeline then cancels the
MPC (so it stops sending JointJog to servo) and the image-based servo takes
over through the same servo node. This module holds the ROS-free parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def candidate_goal_in_world(pre_pos: np.ndarray, quat_xyzw: np.ndarray,
                            T_world_planning: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-grasp pose of a candidate (planning frame) -> world frame.

    Returns (position (3,), quaternion xyzw (4,)). The candidate's z axis is
    the gripper approach direction (core.select_grasp_candidates)."""
    R_wp = T_world_planning[:3, :3]
    p = R_wp @ np.asarray(pre_pos, float) + T_world_planning[:3, 3]
    R = R_wp @ Rotation.from_quat(np.asarray(quat_xyzw, float)).as_matrix()
    return p, Rotation.from_matrix(R).as_quat()


@dataclass
class ReachMonitor:
    """Decides when the MPC has delivered the arm to the pre-grasp pose.

    ``update`` takes the latest MPC status (mode string, TCP error and
    approach-axis error, and its receive time) and returns "wait", "handoff"
    or "timeout". Handoff: close enough held for ``settle_s``, where close
    enough is the MPC's own "reached" (1 cm, 15 deg) or a TCP error within
    ``tcp_tolerance_m`` and an axis error within ``angle_tolerance_deg``: the
    image-based servo and the final approach correct the last centimetre, and
    the MPC can creep just outside its 1 cm for many seconds. Timeout: no
    handoff within ``timeout_s``, no MPC status at all within
    ``first_status_timeout_s`` (MPC not running or not subscribed), or the
    status stopped for ``status_timeout_s``.
    """

    start_t: float
    timeout_s: float = 60.0
    settle_s: float = 1.0
    first_status_timeout_s: float = 10.0
    status_timeout_s: float = 2.0
    tcp_tolerance_m: float = 0.02
    angle_tolerance_deg: float = 15.0
    reached_since: Optional[float] = None

    def update(self, mode: Optional[str], status_t: Optional[float], now: float,
               tcp_error_m: Optional[float] = None, angle_deg: Optional[float] = None) -> str:
        elapsed = now - self.start_t
        if elapsed > self.timeout_s:
            return "timeout"
        if status_t is None or status_t < self.start_t:
            return "timeout" if elapsed > self.first_status_timeout_s else "wait"
        if now - status_t > self.status_timeout_s:
            return "timeout"
        close = mode == "reached" or (
            mode in ("whole_body", "handoff_ready") and tcp_error_m is not None
            and tcp_error_m <= self.tcp_tolerance_m
            and (angle_deg is None or angle_deg <= self.angle_tolerance_deg))
        if not close:
            self.reached_since = None
            return "wait"
        if self.reached_since is None:
            self.reached_since = status_t
        return "handoff" if status_t - self.reached_since >= self.settle_s else "wait"
