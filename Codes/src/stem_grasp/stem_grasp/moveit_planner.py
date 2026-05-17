"""MoveIt 2 planner wrapper.

Ports the ROS 1 stem_grasp_ros1/src/stem_grasp_ros1/moveit_planner.py.
ROS 1 used moveit_commander; ROS 2 has two paths:

  (a) moveit_py — the new Python bindings shipped with MoveIt 2 (preferred).
  (b) MoveGroup action client over /move_group + /execute_trajectory.

This module uses (a). The public surface mirrors the ROS 1 wrapper so the
pipeline node's call sites need only the smallest possible edits.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    # moveit_py is the official MoveIt 2 Python binding (Humble+).
    from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
    from moveit.core.robot_state import RobotState
    _MOVEIT_PY_OK = True
except Exception:  # pragma: no cover — moveit_py not installed in dev envs
    MoveItPy = None  # type: ignore[assignment]
    MultiPipelinePlanRequestParameters = None  # type: ignore[assignment]
    RobotState = None  # type: ignore[assignment]
    _MOVEIT_PY_OK = False

from geometry_msgs.msg import Pose, PoseStamped


@dataclass
class PlanDiagnostics:
    """Plan attempt diagnostics, kept structurally identical to the ROS 1 dict
    returned by plan_to_pose_with_diagnostics for drop-in compatibility."""

    success: bool = False
    planning_time: Optional[float] = None
    error_code: Optional[int] = None
    error_name: str = "UNKNOWN"


class MoveItPlanner:
    """Stateful wrapper around moveit_py.PlanningComponent.

    Constructor parameters mirror those read from ROS params by the ROS 1
    StemGraspPlannerRos1. In the ROS 2 stack, the caller (pipeline_node) is
    responsible for reading params and passing them here.

    Phase 0 surface used by pipeline_node:
        plan_to_pose_with_diagnostics(position, quaternion) -> (plan, diag)
        execute(plan) -> bool
        add_mesh(name, o3d_mesh, frame)
    """

    def __init__(
        self,
        node: "rclpy.node.Node",  # type: ignore[name-defined]
        group_name: str = "arm",
        planning_frame: str = "piper_base_link",
        planning_time: float = 3.0,
        planning_attempts: int = 8,
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.35,
        velocity_scale: float = 0.12,
        acceleration_scale: float = 0.12,
        moveit_config_package: str = "piper_moveit",
    ) -> None:
        self.node = node
        self.group_name = group_name
        self.planning_frame = planning_frame
        self.planning_time = planning_time
        self.planning_attempts = planning_attempts
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.velocity_scale = velocity_scale
        self.acceleration_scale = acceleration_scale

        if not _MOVEIT_PY_OK:
            node.get_logger().warn(
                "moveit_py not available — MoveItPlanner is non-functional. "
                "Install ros-humble-moveit-py or build moveit2 from source."
            )
            self._moveit = None
            self._planning_component = None
            return

        # MoveItPy reads moveit config from a YAML or from set_node_parameters.
        # For Phase 0 we let the caller (pipeline_node) wire the relevant
        # parameters into the rclpy node, then hand the node here.
        self._moveit = MoveItPy(node_name="moveit_py_inline", launch_params=None)
        self._planning_component = self._moveit.get_planning_component(group_name)

    # ----------------------------------------------------------------- public
    def plan_to_pose(self, position, quaternion):
        plan, _ = self.plan_to_pose_with_diagnostics(position, quaternion)
        return plan

    def plan_to_pose_with_diagnostics(
        self, position, quaternion
    ) -> Tuple[Optional[object], PlanDiagnostics]:
        diag = PlanDiagnostics()
        if self._planning_component is None:
            return None, diag

        target = PoseStamped()
        target.header.frame_id = self.planning_frame
        target.pose.position.x = float(position[0])
        target.pose.position.y = float(position[1])
        target.pose.position.z = float(position[2])
        target.pose.orientation.x = float(quaternion[0])
        target.pose.orientation.y = float(quaternion[1])
        target.pose.orientation.z = float(quaternion[2])
        target.pose.orientation.w = float(quaternion[3])

        self._planning_component.set_start_state_to_current_state()
        self._planning_component.set_goal_state(
            pose_stamped_msg=target,
            pose_link=self._eef_link_for_group(),
        )

        # planning_time / attempts are set via MultiPipelinePlanRequestParameters
        # in moveit_py. For Phase 0 keep it simple with the default planner.
        result = self._planning_component.plan()
        diag.success = bool(getattr(result, "trajectory", None))
        if hasattr(result, "planning_time"):
            diag.planning_time = self._coerce_planning_time(result.planning_time)
        if hasattr(result, "error_code"):
            err = result.error_code
            err_val = getattr(err, "val", err)
            diag.error_code = int(err_val) if err_val is not None else None
            diag.error_name = self._moveit_error_name(diag.error_code)
        else:
            diag.error_name = "SUCCESS" if diag.success else "FAILURE"

        return (result.trajectory if diag.success else None), diag

    def execute(self, plan) -> bool:
        if plan is None or self._moveit is None:
            return False
        try:
            self._moveit.execute(plan, controllers=[])
            return True
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"execute failed: {exc}")
            return False

    def add_mesh(self, name: str, o3d_mesh, frame: str = "base_link") -> None:
        """Add an Open3D mesh to the MoveIt 2 planning scene.

        Writes the mesh to a temp .ply and calls PlanningSceneMonitor through
        moveit_py's planning_scene_monitor interface.
        """
        if self._moveit is None or o3d_mesh is None:
            return
        try:
            import open3d as o3d  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"open3d unavailable: {exc}")
            return

        fname = os.path.join(tempfile.gettempdir(), f"{name}.ply")
        try:
            o3d.io.write_triangle_mesh(fname, o3d_mesh)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"failed to write mesh to {fname}: {exc}")
            return

        try:
            with self._moveit.get_planning_scene_monitor().read_write() as scene:
                # moveit_py exposes the PlanningScene via this RAII guard;
                # adding a mesh CO requires building a CollisionObject.
                from moveit_msgs.msg import CollisionObject
                from shape_msgs.msg import Mesh, MeshTriangle
                from geometry_msgs.msg import Point

                co = CollisionObject()
                co.id = name
                co.header.frame_id = frame
                mesh = Mesh()
                for tri in np.asarray(o3d_mesh.triangles):
                    t = MeshTriangle()
                    t.vertex_indices = [int(tri[0]), int(tri[1]), int(tri[2])]
                    mesh.triangles.append(t)
                for v in np.asarray(o3d_mesh.vertices):
                    p = Point()
                    p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
                    mesh.vertices.append(p)
                co.meshes.append(mesh)
                co.mesh_poses.append(Pose(orientation=_unit_quat()))
                co.operation = CollisionObject.ADD
                scene.apply_collision_object(co)
                scene.current_state.update()
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"add_mesh failed: {exc}")

    # ---------------------------------------------------------------- helpers
    def _eef_link_for_group(self) -> str:
        # piper_moveit defines the end-effector tip; in Phase 0 we hardcode
        # the convention used by the unified URDF.
        return "piper_link6"

    @staticmethod
    def _coerce_planning_time(value):
        try:
            numeric = float(value)
        except Exception:  # noqa: BLE001
            return None
        if not math.isfinite(numeric):
            return None
        if numeric < 0.0 or numeric > 1e4:
            return None
        if 0.0 < numeric < 1e-6:
            return None
        return numeric

    @staticmethod
    def _moveit_error_name(code):
        if code is None:
            return "UNKNOWN"
        try:
            from moveit_msgs.msg import MoveItErrorCodes
        except Exception:  # noqa: BLE001
            return str(code)
        code_map = {
            value: name
            for name, value in MoveItErrorCodes.__dict__.items()
            if name.isupper() and isinstance(value, int)
        }
        return code_map.get(int(code), str(code))


def _unit_quat():
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.w = 1.0
    return q
