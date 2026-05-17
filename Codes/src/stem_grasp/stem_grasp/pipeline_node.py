"""stem_grasp pipeline node — ROS 2 Humble port.

Port of stem_grasp_ros1/scripts/pipeline_node.py to rclpy. The state machine,
parameter wiring, topic plumbing, and the inner-loop visual servo are functional
end-to-end (modulo Phase 0 hardware bring-up). Remaining stubs:

  * _outer_loop — skeleton extraction + candidate selection are wired through
    core.skeletonize_plant_points / extract_main_stem / select_grasp_candidates,
    but the plan-and-execute path needs a stem cloud subscriber + the live
    target-pose handoff to moveit_planner. See TODO(P0.4.11).
  * Iterative approach state machine (TODO P0.4.13) and multi-view ring
    (deferred to Phase 4).

References between ports and ROS 1 source (file line numbers):
  - core.py classes:                  same module
  - moveit_planner.MoveItPlanner:     replaces moveit_commander wrapper
  - pointcloud_node:                  /stem_grasp/filtered_cloud
                                      /stem_grasp/leaf_filtered_cloud
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    PointStamped,
    PoseStamped,
    TwistStamped,
    WrenchStamped,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from std_msgs.msg import Float32, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from stem_grasp import core
from stem_grasp.moveit_planner import MoveItPlanner


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class PipelineState(Enum):
    IDLE = auto()
    SCANNING = auto()
    SERVOING = auto()
    ABORTED = auto()


@dataclass
class PipelineParams:
    """All parameters; names mirror the ROS 1 launch arguments verbatim so
    tuning carries over without translation."""

    # Loop rates
    outer_loop_hz: float = 5.0
    inner_loop_hz: float = 100.0

    # Frames
    planning_frame: str = "piper_base_link"
    eef_frame: str = "piper_link6"
    camera_optical_frame: str = "camera_color_optical_frame"
    move_group: str = "arm"

    # Force / safety
    max_force_n: float = 2.0
    contact_threshold_n: float = 0.15

    # Approach
    approach_target_distance: float = 0.25
    approach_step: float = 0.05
    approach_max_steps: int = 8
    approach_settle_sec: float = 0.4
    approach_distance_tolerance: float = 0.01
    approach_mask_wait_sec: float = 2.0
    approach_min_leaf_pixels: int = 60
    approach_goal_orientation_tolerance: float = 0.05

    # Skeleton stability
    skeleton_min_stem_points: int = 6
    skeleton_max_endpoint_jump_m: float = 0.05
    skeleton_cache_max_age_sec: float = 2.0

    # Planning
    goal_position_tolerance: float = 0.01
    goal_orientation_tolerance: float = 0.35
    target_position_offset_m: float = 0.12
    velocity_scale: float = 0.12
    acceleration_scale: float = 0.12
    grasp_strategy: str = "nearest_target"
    target_point_timeout_sec: float = 3.5

    # Servo
    servo_cmd_topic: str = "/servo_node/delta_twist_cmds"

    # Camera intrinsics — populated at runtime from CameraInfo
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class StemGraspPipeline(Node):
    def __init__(self) -> None:
        super().__init__("stem_grasp_pipeline")

        self.params = self._load_params()
        self.state = PipelineState.IDLE
        self.state_lock = threading.Lock()

        # Latest observations
        self.current_force: float = 0.0
        self.last_mask_centroid_uv: Optional[np.ndarray] = None
        self.last_mask_stamp: Optional[float] = None
        self.last_target_point: Optional[np.ndarray] = None  # in planning_frame
        self.last_target_stamp: Optional[float] = None
        self.last_stem_cloud: Optional[np.ndarray] = None  # (N, 3) in planning_frame
        self.last_stem_cloud_stamp: Optional[float] = None
        self.camera_info_ready = False

        # Skeleton cache
        self.cached_skeleton: Optional[np.ndarray] = None
        self.cached_skeleton_stamp: Optional[float] = None

        # Components
        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Servo controller — built lazily once camera intrinsics arrive
        self.servo: Optional[core.FullAdaptiveServoController] = None

        # MoveIt 2 wrapper — best-effort; if moveit_py unavailable the planner
        # plan_to_pose_with_diagnostics returns (None, success=False).
        self.planner = MoveItPlanner(
            node=self,
            group_name=self.params.move_group,
            planning_frame=self.params.planning_frame,
            position_tolerance=self.params.goal_position_tolerance,
            orientation_tolerance=self.params.goal_orientation_tolerance,
            velocity_scale=self.params.velocity_scale,
            acceleration_scale=self.params.acceleration_scale,
        )

        qos_sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)

        # ---------- subscribers ----------
        self.create_subscription(
            WrenchStamped, "/ft_sensor/raw", self._on_wrench,
            qos_sensor, callback_group=self.cb_group,
        )
        self.create_subscription(
            Image, "/stem_grasp/mask", self._on_stem_mask,
            qos_sensor, callback_group=self.cb_group,
        )
        self.create_subscription(
            PointStamped, "/stem_grasp/target_point", self._on_target_point,
            qos_sensor, callback_group=self.cb_group,
        )
        self.create_subscription(
            PointCloud2, "/stem_grasp/filtered_cloud", self._on_stem_cloud,
            qos_sensor, callback_group=self.cb_group,
        )
        self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self._on_camera_info,
            qos_sensor, callback_group=self.cb_group,
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_state,
            qos_reliable, callback_group=self.cb_group,
        )

        # ---------- publishers ----------
        self.pub_state = self.create_publisher(
            String, "/stem_grasp/pipeline_state", 10
        )
        self.pub_best_score = self.create_publisher(
            Float32, "/stem_grasp/best_grasp_score", 10
        )
        self.pub_target_pose = self.create_publisher(
            PoseStamped, "/stem_grasp/target_pose", 10
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

        # ---------- timers ----------
        outer_period = 1.0 / max(self.params.outer_loop_hz, 1e-3)
        inner_period = 1.0 / max(self.params.inner_loop_hz, 1e-3)
        self.create_timer(outer_period, self._outer_loop, callback_group=self.cb_group)
        self.create_timer(inner_period, self._inner_loop, callback_group=self.cb_group)
        self.create_timer(0.5, self._publish_state, callback_group=self.cb_group)

        self.get_logger().info(
            "stem_grasp pipeline (ROS 2) ready. "
            f"outer={self.params.outer_loop_hz} Hz, inner={self.params.inner_loop_hz} Hz."
        )
        self._set_state(PipelineState.SCANNING)

    # ----------------------------------------------------------------- params
    def _load_params(self) -> PipelineParams:
        params = PipelineParams()
        for field_name in PipelineParams.__dataclass_fields__:
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
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"mask conversion failed: {exc}")
            return
        # Centroid of the largest connected component (proxy for stem centroid)
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            self.last_mask_centroid_uv = None
            return
        u = float(xs.mean())
        v = float(ys.mean())
        self.last_mask_centroid_uv = np.array([u, v])
        self.last_mask_stamp = self._now()

    def _on_target_point(self, msg: PointStamped) -> None:
        # Convert to planning_frame for reuse downstream
        try:
            tf = self.tf_buffer.lookup_transform(
                self.params.planning_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception:
            return
        # Apply translation+rotation manually (avoids tf2_geometry_msgs dep)
        t = tf.transform.translation
        r = tf.transform.rotation
        # Quaternion rotation: v' = q v q*; use scipy for clarity.
        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        p_in = np.array([msg.point.x, msg.point.y, msg.point.z])
        p_out = rot @ p_in + np.array([t.x, t.y, t.z])
        self.last_target_point = p_out
        self.last_target_stamp = self._now()

    def _on_stem_cloud(self, msg: PointCloud2) -> None:
        # Pull XYZ points; transform to planning_frame.
        pts = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )
        if len(pts) == 0:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.params.planning_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception:
            return
        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_quat(
            [tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w]
        ).as_matrix()
        t = np.array([tf.transform.translation.x,
                      tf.transform.translation.y,
                      tf.transform.translation.z])
        self.last_stem_cloud = (rot @ pts.T).T + t
        self.last_stem_cloud_stamp = self._now()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.params.fx, self.params.fy = float(msg.k[0]), float(msg.k[4])
        self.params.cx, self.params.cy = float(msg.k[2]), float(msg.k[5])
        if self.servo is None:
            self.servo = core.FullAdaptiveServoController(
                fx=self.params.fx,
                fy=self.params.fy,
                cx=self.params.cx,
                cy=self.params.cy,
            )
            self.get_logger().info(
                f"servo controller initialized: fx={self.params.fx:.1f} "
                f"fy={self.params.fy:.1f}"
            )
        self.camera_info_ready = True

    def _on_joint_state(self, msg: JointState) -> None:
        # TODO(P0.4.13): use joint vel for the stationary-gate of segmentation
        # and as an IK seed for moveit_py replan calls.
        pass

    # ------------------------------------------------------------------ loops
    def _outer_loop(self) -> None:
        if self.state != PipelineState.SCANNING:
            return
        if self.last_stem_cloud is None:
            return
        if self._age(self.last_stem_cloud_stamp) > self.params.skeleton_cache_max_age_sec:
            return

        # 1) Extract 3D skeleton from the masked stem cloud
        G, sk_pts = core.skeletonize_plant_points(self.last_stem_cloud, voxel_size=0.003)
        if len(sk_pts) < self.params.skeleton_min_stem_points:
            return
        stem = core.extract_main_stem(G, sk_pts)
        if len(stem) < self.params.skeleton_min_stem_points:
            return

        # 2) Skeleton stability gate (endpoint jump)
        if self.cached_skeleton is not None and len(self.cached_skeleton) > 0:
            prev_tip = self.cached_skeleton[-1]
            new_tip = stem[-1]
            if np.linalg.norm(new_tip - prev_tip) > self.params.skeleton_max_endpoint_jump_m:
                # Re-use the cached skeleton; current is unstable
                stem = self.cached_skeleton
        self.cached_skeleton = stem
        self.cached_skeleton_stamp = self._now()
        self._publish_skeleton_markers(stem)

        # 3) Select candidates (ratio mode or nearest_target mode)
        target_offset = self.params.target_position_offset_m
        candidates = core.select_grasp_candidates(
            stem, num_candidates=5, target_offset=target_offset
        )
        if not candidates:
            return

        if (self.params.grasp_strategy == "nearest_target"
                and self.last_target_point is not None
                and self._age(self.last_target_stamp) < self.params.target_point_timeout_sec):
            tgt = self.last_target_point
            candidates.sort(key=lambda c: np.linalg.norm(c["pos"] - tgt))

        best = candidates[0]

        # 4) Publish best target pose for downstream consumers + score
        self._publish_target_pose(best)
        score_msg = Float32()
        score_msg.data = float(best["score"])
        self.pub_best_score.publish(score_msg)

        # TODO(P0.4.11): planner.plan_to_pose_with_diagnostics(best["pre_pos"], best["quat"])
        # When moveit_py is wired up, call plan -> execute -> set_state(SERVOING).

    def _inner_loop(self) -> None:
        if self.state != PipelineState.SERVOING:
            return
        if self.current_force > self.params.max_force_n:
            self.get_logger().warn(
                f"Force limit exceeded ({self.current_force:.3f} N) — back to SCANNING."
            )
            self._set_state(PipelineState.SCANNING)
            return
        if self.servo is None or self.last_mask_centroid_uv is None:
            return
        if self.last_target_point is None:
            return

        # Drive the IBVS toward the image-plane centroid; the desired uv is
        # the projection of the 3D target onto the image (here approximated
        # as image center; the proper projection comes in TODO P0.4.12 once
        # the target_point projection helper is ported).
        desired_uv = np.array([self.params.cx, self.params.cy])
        depth_z = float(max(self.last_target_point[2], 0.05))
        vel, diag = self.servo.step(
            raw_uv=self.last_mask_centroid_uv,
            desired_uv=desired_uv,
            depth_z=depth_z,
            force_n=self.current_force,
            commanded_vel=None,
        )
        self._publish_twist(vel)

    # ----------------------------------------------------------------- helpers
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _age(self, t: Optional[float]) -> float:
        if t is None:
            return float("inf")
        return self._now() - t

    def _publish_twist(self, vel: np.ndarray) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.params.planning_frame
        msg.twist.linear.x = float(vel[0])
        msg.twist.linear.y = float(vel[1])
        msg.twist.linear.z = float(vel[2])
        self.pub_servo_cmd.publish(msg)

    def _publish_target_pose(self, candidate: dict) -> None:
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.params.planning_frame
        ps.pose.position.x = float(candidate["pre_pos"][0])
        ps.pose.position.y = float(candidate["pre_pos"][1])
        ps.pose.position.z = float(candidate["pre_pos"][2])
        ps.pose.orientation.x = float(candidate["quat"][0])
        ps.pose.orientation.y = float(candidate["quat"][1])
        ps.pose.orientation.z = float(candidate["quat"][2])
        ps.pose.orientation.w = float(candidate["quat"][3])
        self.pub_target_pose.publish(ps)

    def _publish_skeleton_markers(self, stem_pts: np.ndarray) -> None:
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = self.params.planning_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "skeleton"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.003
        m.color.r = 0.2
        m.color.g = 1.0
        m.color.b = 0.2
        m.color.a = 1.0
        for p in stem_pts:
            pt = Point()
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
            m.points.append(pt)
        ma.markers.append(m)
        self.pub_skeleton_markers.publish(ma)

    # ----------------------------------------------------------------- state
    def _set_state(self, new_state: PipelineState) -> None:
        with self.state_lock:
            if new_state == self.state:
                return
            self.get_logger().info(f"{self.state.name} -> {new_state.name}")
            self.state = new_state

    def _publish_state(self) -> None:
        msg = String()
        msg.data = self.state.name
        self.pub_state.publish(msg)


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
