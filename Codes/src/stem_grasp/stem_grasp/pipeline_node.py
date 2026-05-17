"""stem_grasp pipeline node — ROS 2 Humble port skeleton.

This is a structural port of the ROS 1 node at:
    ../../../../src/stem_grasp_ros1/scripts/pipeline_node.py

GOAL OF THIS FILE
-----------------
1. Establish the rclpy node lifecycle (parameter loading, subs, pubs, timers).
2. Preserve every topic name and parameter key from the ROS 1 launch so that
   tuning carries over (see config/pipeline.yaml and bringup system.yaml).
3. Provide clearly-scoped TODO blocks for each algorithm chunk to be ported
   one at a time, referencing the line range in the ROS 1 source.

WHAT IS *NOT* HERE YET
----------------------
- The skeleton/candidate selection math (port from ROS 1 lines ~600–900).
- The iterative approach state machine (port from ROS 1 lines ~1070–1220).
- The multi-view ICP capture loop (port from ROS 1 lines ~1500–2000).
- MoveIt 2 motion planning wrapper (a separate `moveit_planner.py` module
  will replace the ROS 1 moveit_commander wrapper).
- The visual servo step (port from src/stem_grasp_ros1/src/stem_grasp_ros1/core.py).

These are tracked in PROGRESS.md at the workspace root.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped, WrenchStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import Float32, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray


# ---------------------------------------------------------------------------
# Constants & enums
# ---------------------------------------------------------------------------


class PipelineState(Enum):
    IDLE = auto()
    SCANNING = auto()
    SERVOING = auto()
    ABORTED = auto()


@dataclass
class PipelineParams:
    """All node parameters in one struct. Loaded from ROS 2 parameters at start.

    Names mirror the ROS 1 launch arguments so tuning carries over verbatim.
    See config/pipeline.yaml for defaults and units.
    """

    # --- Loop rates ---
    outer_loop_hz: float = 5.0
    inner_loop_hz: float = 100.0

    # --- Frames ---
    planning_frame: str = "piper_base_link"
    eef_frame: str = "piper_link6"
    camera_optical_frame: str = "camera_color_optical_frame"
    move_group: str = "arm"

    # --- Force / safety ---
    max_force_n: float = 2.0
    contact_threshold_n: float = 0.15

    # --- Approach ---
    approach_target_distance: float = 0.25
    approach_step: float = 0.05
    approach_max_steps: int = 8
    approach_settle_sec: float = 0.4
    approach_distance_tolerance: float = 0.01
    approach_mask_wait_sec: float = 2.0
    approach_min_leaf_pixels: int = 60
    approach_goal_orientation_tolerance: float = 0.05

    # --- Skeleton stability ---
    skeleton_min_stem_points: int = 6
    skeleton_max_endpoint_jump_m: float = 0.05
    skeleton_cache_max_age_sec: float = 2.0

    # --- Planning ---
    goal_position_tolerance: float = 0.01
    goal_orientation_tolerance: float = 0.35
    target_position_offset_m: float = 0.12
    velocity_scale: float = 0.12
    acceleration_scale: float = 0.12
    grasp_strategy: str = "nearest_target"
    target_point_timeout_sec: float = 3.5

    # --- Servo ---
    servo_cmd_topic: str = "/servo_node/delta_twist_cmds"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class StemGraspPipeline(Node):
    """Scan-plan-servo pipeline orchestrator.

    State machine:
        IDLE ──start_request──▶ SCANNING ──plan_executed──▶ SERVOING
         ▲                                                       │
         └──────────────  force_limit_or_done  ──────────────────┘
    """

    def __init__(self) -> None:
        super().__init__("stem_grasp_pipeline")

        self.params = self._load_params()
        self.state = PipelineState.IDLE
        self.current_force: float = 0.0
        self.current_target: Optional[dict] = None
        self.last_mask_centroid_uv: Optional[tuple[float, float]] = None
        self.last_mask_stamp: Optional[float] = None
        self.last_target_point_stamp: Optional[float] = None

        # Concurrent callbacks: TF lookups + force monitor must not block planning.
        self.cb_group = ReentrantCallbackGroup()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos_sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)

        # ---------------- Subscribers (sensor + perception inputs) ----------
        self.create_subscription(
            WrenchStamped,
            "/ft_sensor/raw",
            self._on_wrench,
            qos_sensor,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Image,
            "/stem_grasp/stem_mask",
            self._on_stem_mask,
            qos_sensor,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            PointStamped,
            "/stem_grasp/target_point",
            self._on_target_point,
            qos_sensor,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            qos_reliable,
            callback_group=self.cb_group,
        )

        # ---------------- Publishers ---------------------------------------
        self.pub_state = self.create_publisher(String, "/stem_grasp/pipeline_state", 10)
        self.pub_best_score = self.create_publisher(
            Float32, "/stem_grasp/best_grasp_score", 10
        )
        self.pub_target_pose = self.create_publisher(
            PoseStamped, "/stem_grasp/target_pose", 10
        )
        self.pub_static_cloud = self.create_publisher(
            PointCloud2, "/static_cloud_out", qos_reliable
        )
        self.pub_skeleton_markers = self.create_publisher(
            MarkerArray, "/stem_grasp/skeleton_markers", 10
        )
        self.pub_leaf_markers = self.create_publisher(
            MarkerArray, "/stem_grasp/leaf_markers", 10
        )
        self.pub_servo_cmd = self.create_publisher(
            TwistStamped, self.params.servo_cmd_topic, 5
        )

        # ---------------- Timers -------------------------------------------
        outer_period = 1.0 / max(self.params.outer_loop_hz, 1e-3)
        inner_period = 1.0 / max(self.params.inner_loop_hz, 1e-3)
        self.create_timer(outer_period, self._outer_loop, callback_group=self.cb_group)
        self.create_timer(inner_period, self._inner_loop, callback_group=self.cb_group)
        self.create_timer(0.5, self._publish_state, callback_group=self.cb_group)

        self.get_logger().info(
            "stem_grasp pipeline (ROS 2 port skeleton) ready. "
            f"outer={self.params.outer_loop_hz} Hz, inner={self.params.inner_loop_hz} Hz."
        )
        self._set_state(PipelineState.SCANNING)

    # ------------------------------------------------------------------ params
    def _load_params(self) -> PipelineParams:
        """Declare and read all parameters defined in PipelineParams.

        Uses the dataclass field defaults if not overridden by the launch file.
        """
        params = PipelineParams()
        for field_name, field_def in PipelineParams.__dataclass_fields__.items():
            default = getattr(params, field_name)
            self.declare_parameter(field_name, default)
            value = self.get_parameter(field_name).value
            setattr(params, field_name, value)
        return params

    # --------------------------------------------------------------- callbacks
    def _on_wrench(self, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        self.current_force = float(np.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))

    def _on_stem_mask(self, msg: Image) -> None:
        """Update the latest stem-mask centroid.

        TODO(port-from-ros1): line ~520-580 of pipeline_node.py
            Compute the centroid (u, v) from the mask image, store with stamp,
            apply moving-average filter, and trigger the candidate refresh
            condition variable.
        """
        self.last_mask_stamp = self.get_clock().now().nanoseconds * 1e-9
        # TODO: actually compute centroid; placeholder None for now.

    def _on_target_point(self, msg: PointStamped) -> None:
        """Cache the target (e.g., flower/fruit) 3D point from segmentation node."""
        self.last_target_point_stamp = self.get_clock().now().nanoseconds * 1e-9
        # TODO(port-from-ros1): line ~610-650 — convert to planning_frame, cache.

    def _on_joint_state(self, msg: JointState) -> None:
        # Used for stationary-gate of segmentation and for IK seeds.
        # TODO(port-from-ros1): line ~480-510 — joint velocity threshold gate.
        pass

    # ----------------------------------------------------------------- loops
    def _outer_loop(self) -> None:
        """Slow loop: scan -> select candidate -> plan -> execute -> servo.

        Mirrors the outer state machine of the ROS 1 pipeline (lines ~830-920
        of pipeline_node.py).
        """
        if self.state != PipelineState.SCANNING:
            return

        # TODO(port-from-ros1): skeleton extraction (lines ~660-700)
        # TODO(port-from-ros1): candidate selection (lines ~720-800)
        # TODO(port-from-ros1): plan_and_execute_to_candidate (lines ~830-850)
        # If plan succeeds, transition: self._set_state(PipelineState.SERVOING)

    def _inner_loop(self) -> None:
        """Fast loop: visual servo step.

        Mirrors ROS 1 inner_loop (lines ~871-901). In Phase 0 we publish to
        /servo_node/delta_twist_cmds where moveit_servo is now an actual
        subscriber (unlike the ROS 1 stack where the topic had no consumer).
        """
        if self.state != PipelineState.SERVOING:
            return

        if self.current_force > self.params.max_force_n:
            self.get_logger().warn(
                f"Force limit exceeded: {self.current_force:.3f} N — back to SCANNING."
            )
            self._set_state(PipelineState.SCANNING)
            return

        # TODO(port-from-ros1): centroid acquisition + projection + MPPI step.
        # For Phase 0 we leave a zero-twist publish so the topic has traffic
        # and downstream subscribers come up cleanly.
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = self.params.planning_frame
        self.pub_servo_cmd.publish(twist)

    # ----------------------------------------------------------------- state
    def _set_state(self, new_state: PipelineState) -> None:
        if new_state == self.state:
            return
        self.get_logger().info(f"{self.state.name} -> {new_state.name}")
        self.state = new_state

    def _publish_state(self) -> None:
        msg = String()
        msg.data = self.state.name
        self.pub_state.publish(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StemGraspPipeline()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
