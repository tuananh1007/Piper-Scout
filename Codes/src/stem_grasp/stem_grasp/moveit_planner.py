"""MoveIt 2 planner wrapper — ROS 2 Humble port skeleton.

Ports stem_grasp_ros1/src/stem_grasp_ros1/moveit_planner.py.
ROS 1 used moveit_commander (Python wrapper). In ROS 2 the equivalent path is
either:

    (a) moveit_py — the new Python bindings shipped with MoveIt 2 Humble.
    (b) MoveGroup C++/Python action client over /move_group + /execute_trajectory.

We target (a) for Phase 0 to keep the migration short.

TODO(port-from-ros1): line-for-line port of plan_to_pose_with_diagnostics,
execute, set_velocity_scaling, set_orientation_constraint, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PlanDiagnostics:
    success: bool
    planning_time_s: float
    error_code: int
    n_waypoints: int


class MoveItPlanner:
    """Stateful wrapper around moveit_py.

    Phase 0 surface area mirrors moveit_planner.py from the ROS 1 stack.
    """

    def __init__(
        self,
        group_name: str = "arm",
        planning_time: float = 3.0,
        planning_attempts: int = 8,
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.35,
        velocity_scale: float = 0.12,
        acceleration_scale: float = 0.12,
    ) -> None:
        self.group_name = group_name
        self.planning_time = planning_time
        self.planning_attempts = planning_attempts
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.velocity_scale = velocity_scale
        self.acceleration_scale = acceleration_scale
        # TODO: instantiate MoveItPy and grab the planning component.

    def plan_to_pose_with_diagnostics(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> Tuple[Optional[object], PlanDiagnostics]:
        """Plan to (position, quaternion) goal.

        TODO(port-from-ros1): see moveit_planner.py:plan_to_pose_with_diagnostics.
        Phase 0 implementation should use moveit_py's PlanningComponent.plan().
        """
        return None, PlanDiagnostics(False, 0.0, -1, 0)

    def execute(self, plan: object) -> bool:
        """Execute a previously-planned trajectory; wait for completion."""
        # TODO: planning_component.execute(plan) with /execute_trajectory action.
        return False
