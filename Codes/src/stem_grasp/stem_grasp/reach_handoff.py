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


def pick_operator_candidate(candidates, target: np.ndarray, max_dist_m: float):
    """Index of the grasp candidate nearest the operator's target (planning
    frame), or None when even that one is farther than ``max_dist_m``: the
    segmented stem is not the one the operator chose, so nothing is reached."""
    if not candidates:
        return None
    d = [float(np.linalg.norm(np.asarray(c["pos"], float) - target)) for c in candidates]
    i = int(np.argmin(d))
    return i if d[i] <= max_dist_m else None


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


@dataclass
class ViewGate:
    """In SCANNING: ask active perception for another view first, or go on.

    ``decide`` returns "request" (call the planner), "wait" (a request or a
    view is pending, or the scene is settling after one) or "proceed" (grasp
    as before). It proceeds when active perception is not running (no status
    within ``status_max_age_s``), reports that no view is needed, has given
    ``max_views`` views already, or did not publish a view within
    ``request_timeout_s`` of a request or refused one (``failed``). After a
    view it waits ``settle_s`` and then for a status received after that, so
    the decision uses the map and estimate that include the new view. Until
    the first status arrives it waits up to ``startup_wait_s`` (from the first
    decision after a reset), so a planner that is still starting is not taken
    for one that is not running; a status without ``needs_view`` (the planner
    has no target estimate yet) is waited on for up to ``startup_wait_s`` too.

    A view is chosen for what it adds to the map and the stem estimate, and
    leaves the planner does not know of can hide the target from it; the grasp
    needs a stem cloud from the current pose. ``should_return`` says when to
    go back to the pose the views started from (the scan pose, which saw the
    target): no stem cloud since the last view ended within ``return_after_s``.
    After that return no more views are asked for."""

    max_views: int = 4
    status_max_age_s: float = 3.0
    settle_s: float = 1.5
    request_timeout_s: float = 10.0
    startup_wait_s: float = 5.0
    return_after_s: float = 3.0
    views: int = 0
    settle_until: float = float("-inf")
    requested_t: Optional[float] = None
    gave_up: str = ""
    started_t: Optional[float] = None
    unassessed_t: Optional[float] = None
    view_done_t: Optional[float] = None
    returned: bool = False

    def reset(self) -> None:
        self.views, self.settle_until, self.requested_t, self.gave_up = 0, float("-inf"), None, ""
        self.started_t, self.unassessed_t, self.view_done_t, self.returned = None, None, None, False

    def decide(self, now: float, status: Optional[dict], status_t: Optional[float]) -> str:
        if self.gave_up or self.returned or self.views >= self.max_views:
            return "proceed"
        if self.started_t is None:
            self.started_t = now
        if now < self.settle_until:
            return "wait"
        if status is None and now - self.started_t < self.startup_wait_s:
            return "wait"
        if self.requested_t is not None:
            if now - self.requested_t > self.request_timeout_s:
                self.failed("no view published")
                return "proceed"
            return "wait"
        if status is None or status_t is None or now - status_t > self.status_max_age_s:
            return "proceed"
        if status_t < self.settle_until:
            return "wait"
        if "needs_view" not in status:                 # running, no target estimate yet
            self.unassessed_t = self.unassessed_t if self.unassessed_t is not None else now
            return "wait" if now - self.unassessed_t < self.startup_wait_s else "proceed"
        return "request" if status["needs_view"] else "proceed"

    def requested(self, now: float) -> None:
        self.requested_t = now

    def view_started(self) -> None:
        self.views += 1
        self.requested_t = None

    def view_done(self, now: float) -> None:
        self.settle_until = now + self.settle_s
        self.view_done_t = now

    def should_return(self, now: float, cloud_t: Optional[float]) -> bool:
        """After the views (``decide`` said "proceed"): True when no stem cloud
        arrived since the last view ended (``cloud_t``: its receive time) within
        ``return_after_s``; once only."""
        if self.views == 0 or self.returned or self.view_done_t is None:
            return False
        if cloud_t is not None and cloud_t >= self.view_done_t:
            return False
        return now - self.view_done_t > self.return_after_s

    def failed(self, reason: str) -> None:
        self.gave_up = reason or "failed"
        self.requested_t = None
