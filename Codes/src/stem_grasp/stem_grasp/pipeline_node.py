"""stem_grasp pipeline node — ROS 2 Humble port.

Port of stem_grasp_ros1/scripts/pipeline_node.py to rclpy. The state machine,
parameter wiring, topic plumbing, and the inner-loop visual servo are functional
end-to-end (modulo Phase 0 hardware bring-up). Remaining stubs:

  * _outer_loop — skeleton extraction + candidate selection are wired through
    core.skeletonize_plant_points / extract_main_stem / select_grasp_candidates.
    The move to the pre-grasp pose (P0.4.11) runs through the whole-body MPC
    with ``reach_executor: whole_body_mpc`` (SCANNING -> REACHING -> SERVOING,
    see reach_handoff.py); the default ``none`` only publishes the target pose.
    MoveIt (moveit_planner) stays unwired: moveit_py has no Humble binary.
  * _inner_loop — image-based servo of the stem onto the gripper approach
    axis, one step per new mask (P0.4.12, see servo_geometry.py).
  * Iterative approach (P0.4.13, approach.py): with ``approach_enabled``
    the handoff goes to APPROACHING, which servoes and advances along the
    gripper axis in steps until AT_GRASP (or ABORTED); with
    ``grasp_close_gripper`` it then closes the gripper (GRASPING) until the
    opening settles on the stem (GRASPED). ``~/release`` opens the gripper
    (RELEASING), backs out along the gripper axis (RETREATING) and goes IDLE;
    ``~/scan`` starts scanning again.
  * Multi-view ring (deferred to Phase 4).

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

import json
import signal

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
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from std_msgs.msg import Empty, Float32, Float64, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from stem_grasp import core
from stem_grasp.approach import ApproachConfig, GripperCloseMonitor, IterativeApproach, RetreatMonitor
from stem_grasp.moveit_planner import MoveItPlanner
from stem_grasp.reach_handoff import ReachMonitor, candidate_goal_in_world
from stem_grasp.servo_geometry import desired_uv as servo_desired_uv
from stem_grasp.servo_geometry import project, stem_feature_uv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class PipelineState(Enum):
    IDLE = auto()
    SCANNING = auto()
    REACHING = auto()
    SERVOING = auto()
    APPROACHING = auto()   # servo + stepwise advance along the gripper axis (P0.4.13)
    AT_GRASP = auto()      # grasp point between the fingers; no more commands
    GRASPING = auto()      # gripper closing (grasp_close_gripper)
    GRASPED = auto()       # gripper settled on the stem; holding
    RELEASING = auto()     # ~/release: gripper opening
    RETREATING = auto()    # backing out along the gripper axis, then IDLE
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
    skeleton_vertical_axis: int = 2      # "up" axis of planning_frame (z for piper_base_link;
                                         # ROS 1 worked in the camera optical frame, axis 1)
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
    # Single-view stem clouds show only the camera-facing half, so skeleton
    # points sit on that surface; fit the cross-section and grasp the axis.
    grasp_point_on_stem_axis: bool = True
    stem_radius_max_m: float = 0.02

    # Servo
    servo_cmd_topic: str = "/servo_node/delta_twist_cmds"
    tcp_offset_m: float = 0.14            # eef_frame -> grasp point along its z (as the MPC)
    servo_lambda_0: float = 0.8           # IBVS gain at zero image error (as ROS 1)
    servo_lambda_inf: float = 0.5         # IBVS gain at large image error (ROS 1: 0.07)
    servo_rho: float = 0.3                # gain decay per pixel of error
    servo_row_band_px: int = 12           # mask rows around the target row for the stem position
    target_point_max_age_sec: float = 0.5  # live /stem_grasp/target_point preferred when fresher
    servo_mask_max_age_sec: float = 0.3   # no servo command on an older mask (servo then halts)
    # Phase 2B: "ibvs" (FullAdaptiveServoController, camera twists) or "mppi"
    # (scout_piper_whole_body_mpc visual_servo: arm-only MPPI on joint velocities,
    # JointJog on servo_joint_cmd_topic; the Scout stays still)
    servo_controller: str = "ibvs"
    servo_joint_cmd_topic: str = "/servo_node/delta_joint_cmds"
    arm_joint_names: str = "piper_joint1,piper_joint2,piper_joint3,piper_joint4,piper_joint5,piper_joint6"
    robot_base_frame: str = "base_link"   # the MPPI model's frame (Scout base_link)
    mppi_samples: int = 256
    mppi_horizon: int = 20
    mppi_dt: float = 0.05                 # planning step (about one mask period)
    mppi_backend: str = "numpy"           # numpy | torch (GPU with mppi_device cuda / auto)
    mppi_device: str = "auto"
    mppi_qd_max: float = 0.5              # rad/s near the stem
    joint_state_max_age_sec: float = 0.2  # no MPPI command on older joint states

    # Iterative final approach (P0.4.13, approach.py): after the handoff, servo
    # and advance along the gripper axis in approach_step steps until the grasp
    # point is within approach_distance_tolerance. A new design: the ROS 1
    # source is not in the repository, so the approach_* names above keep their
    # ROS 1 names with the meanings given in approach.py
    # (approach_goal_orientation_tolerance, a MoveIt goal tolerance, is unused).
    approach_enabled: bool = False        # false: servo only, no motion toward the stem
    approach_speed_mps: float = 0.02      # advance speed along the gripper axis
    approach_align_tolerance_px: float = 8.0  # image error that allows the next step
    approach_timeout_sec: float = 60.0

    # Grasp (after AT_GRASP): open the gripper before the approach, close it on
    # arrival through piper_servo_bridge (~/gripper_cmd, enabled bridge only)
    # and wait for the measured opening to settle. Off by default.
    grasp_close_gripper: bool = False
    grasp_gripper_topic: str = "/piper_servo_bridge/gripper_cmd"
    grasp_open_width_m: float = 0.06
    grasp_closed_width_m: float = 0.0     # target; the stem stops the fingers
    grasp_settle_sec: float = 0.5
    grasp_timeout_sec: float = 5.0
    grasp_min_object_m: float = 0.002     # settled below this: closed on nothing
    gripper_finger_joints: str = "piper_joint7,piper_joint8"   # opening = first - second

    # Release (~/release from GRASPED or AT_GRASP): open the gripper to
    # grasp_open_width_m, then back the gripper straight out along its axis by
    # release_retreat_m and go IDLE (~/scan starts scanning again).
    release_retreat_m: float = 0.10
    release_speed_mps: float = 0.03
    release_open_timeout_sec: float = 5.0
    release_timeout_sec: float = 15.0

    # Reach to the pre-grasp pose (P0.4.11): "none" only publishes the target
    # pose; "whole_body_mpc" hands it to scout_piper_whole_body_mpc
    reach_executor: str = "none"
    mpc_world_frame: str = "odom"
    mpc_goal_pose_topic: str = "/whole_body_mpc/goal_pose"
    mpc_status_topic: str = "/whole_body_mpc/status"
    mpc_cancel_topic: str = "/whole_body_mpc/cancel"
    reach_timeout_sec: float = 60.0
    reach_settle_sec: float = 1.0
    reach_handoff_tolerance_m: float = 0.02   # TCP error accepted for the handoff (MPC "reached": 1 cm)
    reach_handoff_angle_deg: float = 15.0     # approach-axis error accepted for the handoff

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
        self.last_mask: Optional[np.ndarray] = None
        self.last_mask_stamp: Optional[float] = None
        self._servo_mask_stamp: Optional[float] = None   # mask the servo last stepped on
        self._servo_cam_prev = None    # (time, world <- camera) at the previous servo step
        self.approach: Optional[IterativeApproach] = None
        self.grip_monitor: Optional[GripperCloseMonitor] = None
        self.gripper_width: Optional[float] = None
        self.gripper_width_t: Optional[float] = None
        self._inner_lock = threading.Lock()
        self.grasp_target_world: Optional[np.ndarray] = None   # in mpc_world_frame
        self.last_target_point: Optional[np.ndarray] = None  # in planning_frame
        self.last_target_stamp: Optional[float] = None
        self.last_stem_cloud: Optional[np.ndarray] = None  # (N, 3) in planning_frame
        self.last_stem_cloud_stamp: Optional[float] = None
        self.camera_info_ready = False

        # Reach handoff (REACHING state)
        self.mpc_mode: Optional[str] = None
        self.mpc_status_t: Optional[float] = None
        self.mpc_tcp_error: Optional[float] = None
        self.mpc_angle: Optional[float] = None
        self.reach_monitor: Optional[ReachMonitor] = None
        self.reach_goal: Optional[PoseStamped] = None

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
        self.arm_q: Optional[np.ndarray] = None
        self.arm_q_t: Optional[float] = None

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
        self.create_subscription(
            String, self.params.mpc_status_topic, self._on_mpc_status,
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
        self.mppi_servo = None                 # Phase 2B controller (servo_controller: mppi)
        self._mppi_lock = threading.Lock()     # camera_info callbacks run concurrently
        self.pub_joint_cmd = None
        if self.params.servo_controller == "mppi":
            from control_msgs.msg import JointJog  # noqa: PLC0415
            self._JointJog = JointJog
            self.pub_joint_cmd = self.create_publisher(JointJog, self.params.servo_joint_cmd_topic, 5)
        elif self.params.servo_controller != "ibvs":
            raise ValueError(f"servo_controller {self.params.servo_controller!r}: ibvs or mppi")
        self.pub_mpc_goal = self.create_publisher(
            PoseStamped, self.params.mpc_goal_pose_topic, 5
        )
        self.pub_mpc_cancel = self.create_publisher(
            Empty, self.params.mpc_cancel_topic, 5
        )
        self.pub_servo_status = self.create_publisher(
            String, "/stem_grasp/servo_status", 10
        )
        self.pub_gripper = self.create_publisher(Float64, self.params.grasp_gripper_topic, 5)
        self.release_t: Optional[float] = None
        self.retreat: Optional[RetreatMonitor] = None
        self._retreat_pub_t = 0.0
        self.create_service(Trigger, "~/release", self._release_srv, callback_group=self.cb_group)
        self.create_service(Trigger, "~/scan", self._scan_srv, callback_group=self.cb_group)
        self._servo_status_t = 0.0

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
        self.last_mask = mask > 0
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_mask_stamp = stamp if stamp > 0 else self._now()

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
        # Humble's read_points returns a structured array; read_points_numpy
        # gives the plain (N, 3) array.
        pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(np.float32)
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
        self.image_hw = (int(msg.height), int(msg.width))
        if self.params.servo_controller == "mppi" and self.mppi_servo is None:
            with self._mppi_lock:
                if self.mppi_servo is None:
                    self.mppi_servo = self._make_mppi_servo()
        if self.servo is None:
            self.servo = self._make_servo()
            self.get_logger().info(
                f"servo controller initialized: fx={self.params.fx:.1f} "
                f"fy={self.params.fy:.1f}"
            )
        self.camera_info_ready = True

    def _make_servo(self) -> "core.FullAdaptiveServoController":
        return core.FullAdaptiveServoController(
            fx=self.params.fx,
            fy=self.params.fy,
            cx=self.params.cx,
            cy=self.params.cy,
            lambda_0=float(self.params.servo_lambda_0),
            lambda_inf=float(self.params.servo_lambda_inf),
            rho=float(self.params.servo_rho),
        )

    def _make_mppi_servo(self):
        """Phase 2B arm-only MPPI visual servo (scout_piper_whole_body_mpc)."""
        from scout_piper_whole_body_mpc.visual_servo import MppiVisualServo, VisualServoConfig  # noqa: PLC0415
        p = self.params
        servo = MppiVisualServo(VisualServoConfig(
            dt=float(p.mppi_dt), horizon=int(p.mppi_horizon), samples=int(p.mppi_samples),
            qd_max=float(p.mppi_qd_max), tcp_offset_m=float(p.tcp_offset_m),
            approach_speed_mps=float(p.approach_speed_mps), backend=str(p.mppi_backend),
            device=str(p.mppi_device)))
        self.get_logger().info(f"MPPI visual servo: {p.mppi_samples} samples x {p.mppi_horizon} steps "
                               f"of {p.mppi_dt} s on {p.mppi_backend}")
        return servo

    def _approach_config(self) -> ApproachConfig:
        p = self.params
        return ApproachConfig(
            step_m=float(p.approach_step), max_steps=int(p.approach_max_steps),
            settle_s=float(p.approach_settle_sec),
            distance_tolerance_m=float(p.approach_distance_tolerance),
            max_start_distance_m=float(p.approach_target_distance),
            mask_wait_s=float(p.approach_mask_wait_sec),
            min_mask_pixels=int(p.approach_min_leaf_pixels),
            align_tolerance_px=float(p.approach_align_tolerance_px),
            speed_mps=float(p.approach_speed_mps), timeout_s=float(p.approach_timeout_sec),
            contact_force_n=float(p.contact_threshold_n))

    def _on_mpc_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
            self.mpc_mode = status.get("mode")
        except (ValueError, AttributeError):
            return
        self.mpc_tcp_error = status.get("tcp_error_m")
        self.mpc_angle = status.get("approach_error_deg")
        self.mpc_status_t = self._now()

    def _on_joint_state(self, msg: JointState) -> None:
        # gripper opening from the relay's finger joints (each half the opening)
        names = [n.strip() for n in self.params.gripper_finger_joints.split(",")]
        pos = dict(zip(msg.name, msg.position))
        arm = [n.strip() for n in self.params.arm_joint_names.split(",")]
        if all(n in pos for n in arm):
            self.arm_q = np.array([pos[n] for n in arm], float)
            self.arm_q_t = self._now()
        if len(names) == 2 and all(n in pos for n in names):
            self.gripper_width = float(pos[names[0]] - pos[names[1]])
            self.gripper_width_t = self._now()

    # ------------------------------------------------------------------ loops
    def _outer_loop(self) -> None:
        if self.state == PipelineState.REACHING:
            self._reach_tick()
            return
        if self.state == PipelineState.GRASPING:
            self._grasp_tick()
            return
        if self.state == PipelineState.RELEASING:
            self._release_tick()
            return
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
        stem = core.extract_main_stem(G, sk_pts, vertical_axis=int(self.params.skeleton_vertical_axis))
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
        # approach from the arm's side: from the stem toward planning_frame's origin
        candidates = core.select_grasp_candidates(
            stem, num_candidates=5, target_offset=target_offset,
            approach_hint=-np.mean(stem, axis=0),
        )
        if not candidates:
            return
        if self.params.grasp_point_on_stem_axis:
            self._centre_on_stem_axis(candidates)

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

        # 5) Move to the pre-grasp pose (P0.4.11)
        if self.params.reach_executor == "whole_body_mpc":
            self._start_reach(best)

    def _centre_on_stem_axis(self, candidates: list) -> None:
        """Move each grasp point from the visible surface to the stem axis."""
        from scipy.spatial.transform import Rotation as R_scipy
        for c in candidates:
            stem_dir = R_scipy.from_quat(c["quat"]).as_matrix()[:, 1]   # candidate y = stem direction
            fit = core.stem_axis_point(self.last_stem_cloud, c["pos"], stem_dir,
                                       r_max=float(self.params.stem_radius_max_m))
            if fit is None:
                continue
            shift = fit[0] - c["pos"]
            c["pos"], c["pre_pos"], c["stem_radius"] = fit[0], c["pre_pos"] + shift, fit[1]

    # ------------------------------------------------------------- reaching
    def _start_reach(self, candidate: dict) -> None:
        """Send the candidate's pre-grasp pose to the whole-body MPC."""
        world = self.params.mpc_world_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                world, self.params.planning_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"no TF {world} <- {self.params.planning_frame}: {exc}",
                                   throttle_duration_sec=5.0)
            return
        from scipy.spatial.transform import Rotation as R_scipy
        q = tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y,
                    tf.transform.translation.z]
        p, quat = candidate_goal_in_world(candidate["pre_pos"], candidate["quat"], T)
        # the grasp point itself, kept world-fixed for the servo (the base moves while reaching)
        self.grasp_target_world = T[:3, :3] @ np.asarray(candidate["pos"], float) + T[:3, 3]
        goal = PoseStamped()
        goal.header.frame_id = world
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = map(float, p)
        (goal.pose.orientation.x, goal.pose.orientation.y,
         goal.pose.orientation.z, goal.pose.orientation.w) = map(float, quat)
        self.reach_goal = goal
        self.reach_monitor = ReachMonitor(start_t=self._now(),
                                          timeout_s=self.params.reach_timeout_sec,
                                          settle_s=self.params.reach_settle_sec,
                                          tcp_tolerance_m=self.params.reach_handoff_tolerance_m,
                                          angle_tolerance_deg=self.params.reach_handoff_angle_deg)
        self._set_state(PipelineState.REACHING)
        radius = candidate.get("stem_radius")
        self.get_logger().info(f"reaching pre-grasp {np.round(p, 3)} in {world} via the whole-body MPC"
                               + (f"; stem diameter {2000 * radius:.1f} mm" if radius else
                                  "; grasp point not centred on the stem"))
        self._reach_tick()

    def _reach_tick(self) -> None:
        if self.reach_monitor is None or self.reach_goal is None:
            self._set_state(PipelineState.SCANNING)
            return
        decision = self.reach_monitor.update(self.mpc_mode, self.mpc_status_t, self._now(),
                                             self.mpc_tcp_error, self.mpc_angle)
        if decision == "wait":
            # republished every outer cycle: the goal topic is best effort and the
            # MPC keeps its warm start for a repeated goal. Never after the
            # decision, or the goal could overtake the cancel on the other topic.
            self.reach_goal.header.stamp = self.get_clock().now().to_msg()
            self.pub_mpc_goal.publish(self.reach_goal)
            return
        self.pub_mpc_cancel.publish(Empty())       # MPC stops and releases servo
        self.reach_monitor = None
        if decision == "handoff" and self.params.approach_enabled:
            self.get_logger().info("pre-grasp reached; servo and stepwise approach take over")
            if self.params.grasp_close_gripper:          # fingers open before moving onto the stem
                self._command_gripper(self.params.grasp_open_width_m)
            self._set_state(PipelineState.APPROACHING)
        elif decision == "handoff":
            self.get_logger().info("pre-grasp reached; image-based servo takes over")
            self._set_state(PipelineState.SERVOING)
        else:
            self.get_logger().warn("reach timed out or the MPC stopped reporting; back to SCANNING")
            self._set_state(PipelineState.SCANNING)

    def _inner_loop(self) -> None:
        # the timer is in a reentrant group: never step the servo twice at once
        if not self._inner_lock.acquire(blocking=False):
            return
        try:
            if self.state == PipelineState.RETREATING:
                self._retreat_step()
            else:
                self._servo_step()
        finally:
            self._inner_lock.release()

    def _servo_step(self) -> None:
        if self.state not in (PipelineState.SERVOING, PipelineState.APPROACHING):
            return
        if self.current_force > self.params.max_force_n:
            self.get_logger().warn(
                f"Force limit exceeded ({self.current_force:.3f} N) — back to SCANNING."
            )
            self._set_state(PipelineState.SCANNING)
            return
        if self.servo is None or self.last_mask is None:
            return
        if (self.state == PipelineState.APPROACHING
                and self._age(self.last_mask_stamp) > self.params.approach_mask_wait_sec):
            self._end_approach("abort", f"no stem mask for {self.params.approach_mask_wait_sec} s")
            return
        # one control step per new mask: repeated measurements would read as
        # a stopped stem. A stale mask gets no command, so servo halts.
        if self.last_mask_stamp == self._servo_mask_stamp:
            return
        if self._age(self.last_mask_stamp) > self.params.servo_mask_max_age_sec:
            return
        dt = None
        if self._servo_mask_stamp is not None:
            dt = float(np.clip(self.last_mask_stamp - self._servo_mask_stamp, 1e-3, 0.5))
        self._servo_mask_stamp = self.last_mask_stamp
        geo = self._servo_geometry()
        if geo is None:
            return
        target_cam, tcp_cam, approach_cam = geo
        camera_vel = self._camera_velocity()
        depth_z = float(target_cam[2])
        if depth_z < 0.05:
            return
        K = np.array([[self.params.fx, 0.0, self.params.cx],
                      [0.0, self.params.fy, self.params.cy], [0.0, 0.0, 1.0]])
        target_uv = project(K, target_cam)
        # P0.4.12: the target must appear where the gripper approach axis
        # crosses its depth (image centre only as a fallback)
        desired = servo_desired_uv(K, tcp_cam, approach_cam, depth_z)
        if desired is None:
            desired = np.array([self.params.cx, self.params.cy])
        raw = stem_feature_uv(self.last_mask, target_uv[1], int(self.params.servo_row_band_px))
        # grasp point minus TCP along the gripper axis, and its distance from the axis
        distance = float((target_cam - tcp_cam) @ approach_cam)
        miss = float(np.linalg.norm(np.cross(target_cam - tcp_cam, approach_cam)))
        if raw is None:
            if self.approach is not None:
                self._approach_speed(float("inf"), distance, 0)   # holds; aborts if it persists
            return
        if self.mppi_servo is not None:
            self._mppi_servo_step(K, target_cam, raw, desired, distance, miss, depth_z)
            return
        vel, diag = self.servo.step(
            raw_uv=raw,
            desired_uv=desired,
            depth_z=depth_z,
            force_n=self.current_force,
            commanded_vel=None,
            dt=dt,
            camera_vel=camera_vel,
        )
        status = {}
        if self.approach is not None:
            speed = self._approach_speed(diag["error_norm"], distance,
                                         int(np.count_nonzero(self.last_mask)))
            if speed is None:                      # approach ended; servo halts
                return
            vel = vel + speed * approach_cam       # advance along the gripper axis
            status = {"approach": self.approach.phase, "step": self.approach.steps}
        # the IBVS velocity is a camera translation in the camera optical frame
        self._publish_twist(vel, self.params.camera_optical_frame)
        now = self._now()
        if now - self._servo_status_t > 0.1:
            self._servo_status_t = now
            self.pub_servo_status.publish(String(data=json.dumps({
                "error_px": diag["error_norm"], "depth_m": depth_z,
                "raw_uv": [float(v) for v in raw], "desired_uv": [float(v) for v in desired],
                "vel_cam": [float(v) for v in vel], "lambda": diag["lambda"],
                "sway_px_s": diag["sway_px_s"], "distance_m": distance, "axis_miss_m": miss,
                **status})))

    def _mppi_servo_step(self, K, target_cam, raw, desired, distance, miss, depth_z) -> None:
        """Phase 2B: one arm-only MPPI step, JointJog out (servo_controller: mppi)."""
        from scout_piper_whole_body_mpc.visual_servo import ServoTarget  # noqa: PLC0415
        p = self.params
        if self.arm_q is None or self._age(self.arm_q_t) > p.joint_state_max_age_sec:
            return                                            # no fresh joints: no command (servo halts)
        cam = p.camera_optical_frame
        T_bc = self._lookup(p.robot_base_frame, cam)          # camera -> Scout base_link
        T_fc = self._lookup(p.eef_frame, cam)                 # camera -> link6
        if T_bc is None or T_fc is None:
            return
        error_px = float(np.linalg.norm(np.asarray(raw) - np.asarray(desired)))
        desired_distance = distance                           # SERVOING: hold the distance
        status = {}
        if self.approach is not None:
            speed = self._approach_speed(error_px, distance, int(np.count_nonzero(self.last_mask)))
            if speed is None:                                 # approach ended; servo halts
                return
            desired_distance = distance - speed * p.mppi_horizon * p.mppi_dt
            status = {"approach": self.approach.phase, "step": self.approach.steps}
        target = ServoTarget(uv_meas=np.asarray(raw, float), K=K, image_hw=getattr(self, "image_hw", (480, 640)),
                             T_flange_cam=T_fc, p_base=T_bc[:3, :3] @ target_cam + T_bc[:3, 3],
                             force_n=float(self.current_force), desired_distance_m=float(desired_distance))
        qd, diag = self.mppi_servo.step(self.arm_q, target)
        msg = self._JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "stem_grasp"                   # tells these apart from the whole-body MPC's
        msg.joint_names = [n.strip() for n in p.arm_joint_names.split(",")]
        msg.velocities = [float(v) for v in qd]
        self.pub_joint_cmd.publish(msg)
        now = self._now()
        if now - self._servo_status_t > 0.1:
            self._servo_status_t = now
            self.pub_servo_status.publish(String(data=json.dumps({
                "controller": "mppi", "error_px": error_px, "depth_m": depth_z,
                "raw_uv": [float(v) for v in raw], "desired_uv": [float(v) for v in desired],
                "qd": [float(v) for v in qd], "solve_ms": diag["solve_ms"], "mode": diag["mode"],
                "safety": diag["safety"], "distance_m": distance, "axis_miss_m": miss,
                "plan_terms": diag.get("plan_terms", {}), "nominal_terms": diag.get("nominal_terms", {}),
                **status})))

    def _approach_speed(self, error_px: float, distance_m: float, pixels: int) -> Optional[float]:
        """Advance speed for this servo step, or None once the approach ended."""
        st = self.approach.update(self._now(), error_px, distance_m, pixels, self.current_force)
        if st.phase in ("done", "abort"):
            self._end_approach(st.phase, st.reason)
            return None
        return st.speed

    def _end_approach(self, phase: str, reason: str) -> None:
        self._publish_twist(np.zeros(3), self.params.camera_optical_frame)
        steps = self.approach.steps if self.approach is not None else 0
        if phase == "done":
            self.get_logger().info(f"approach done after {steps} steps: {reason}")
            self._set_state(PipelineState.AT_GRASP)
            if self.params.grasp_close_gripper:
                self._command_gripper(self.params.grasp_closed_width_m)
                self.grip_monitor = GripperCloseMonitor(
                    start_t=self._now(), settle_s=float(self.params.grasp_settle_sec),
                    timeout_s=float(self.params.grasp_timeout_sec),
                    min_object_m=float(self.params.grasp_min_object_m))
                self._set_state(PipelineState.GRASPING)
        else:
            self.get_logger().warn(f"approach aborted after {steps} steps: {reason}")
            self._set_state(PipelineState.ABORTED)

    # --------------------------------------------------------------- release
    def _release_srv(self, req, res):
        if self.state not in (PipelineState.GRASPED, PipelineState.AT_GRASP):
            res.success, res.message = False, f"release needs GRASPED or AT_GRASP, not {self.state.name}"
            return res
        self._command_gripper(self.params.grasp_open_width_m)
        self.release_t = self._now()
        self._set_state(PipelineState.RELEASING)
        res.success, res.message = True, "releasing"
        return res

    def _scan_srv(self, req, res):
        if self.state not in (PipelineState.IDLE, PipelineState.ABORTED):
            res.success, res.message = False, f"scan needs IDLE or ABORTED, not {self.state.name}"
            return res
        self._set_state(PipelineState.SCANNING)
        res.success, res.message = True, "scanning"
        return res

    def _release_tick(self) -> None:
        width = self.gripper_width if self._age(self.gripper_width_t) < 0.5 else None
        if width is not None and width >= self.params.grasp_open_width_m - 0.005:
            tcp = self._tcp_world()
            if tcp is None:
                self.get_logger().warn("release: no TF for the TCP; not retreating")
                self._set_state(PipelineState.ABORTED)
                return
            self.retreat = RetreatMonitor(
                start_t=self._now(), start_tcp=tcp[0], axis=tcp[1],
                distance_m=float(self.params.release_retreat_m),
                timeout_s=float(self.params.release_timeout_sec))
            self.get_logger().info(f"gripper open ({1000 * width:.1f} mm); retreating "
                                   f"{100 * self.params.release_retreat_m:.0f} cm")
            self._set_state(PipelineState.RETREATING)
        elif self._age(self.release_t) > self.params.release_open_timeout_sec:
            # never pull away with the stem still held
            self.get_logger().warn("release: the gripper did not open; not retreating")
            self._set_state(PipelineState.ABORTED)

    def _tcp_world(self):
        """(TCP position, approach axis) in mpc_world_frame, or None."""
        T = self._lookup(self.params.mpc_world_frame, self.params.eef_frame)
        if T is None:
            return None
        return T[:3, 3] + self.params.tcp_offset_m * T[:3, 2], T[:3, 2]

    def _retreat_step(self) -> None:
        now = self._now()
        if now - self._retreat_pub_t < 0.05 or self.retreat is None:   # 20 Hz is plenty for servo
            return
        self._retreat_pub_t = now
        tcp = self._tcp_world()
        decision = self.retreat.update(now, None if tcp is None else tcp[0])
        if decision != "wait":
            self._publish_twist(np.zeros(3), self.params.camera_optical_frame)
            moved = self.retreat.moved
            self.retreat = None
            if decision == "done":
                self.get_logger().info(f"retreated {100 * moved:.1f} cm; IDLE (call ~/scan to start again)")
                self._set_state(PipelineState.IDLE)
            else:
                self.get_logger().warn(f"retreat timed out after {100 * moved:.1f} cm")
                self._set_state(PipelineState.ABORTED)
            return
        T_ce = self._lookup(self.params.camera_optical_frame, self.params.eef_frame)
        if T_ce is not None:   # straight back along the gripper axis, in the camera frame
            self._publish_twist(-self.params.release_speed_mps * T_ce[:3, 2],
                                self.params.camera_optical_frame)

    def _command_gripper(self, width: float) -> None:
        self.pub_gripper.publish(Float64(data=float(width)))
        self.get_logger().info(f"gripper -> {1000 * width:.1f} mm")

    def _grasp_tick(self) -> None:
        if self.grip_monitor is None:
            self._set_state(PipelineState.ABORTED)
            return
        width = self.gripper_width if self._age(self.gripper_width_t) < 0.5 else None
        decision = self.grip_monitor.update(self._now(), width)
        if decision == "wait":
            return
        w = self.grip_monitor.width
        self.grip_monitor = None
        if decision == "grasped":
            self.get_logger().info(f"grasped: gripper settled at {1000 * w:.1f} mm")
            self._set_state(PipelineState.GRASPED)
        elif decision == "empty":
            self.get_logger().warn(f"gripper closed to {1000 * w:.1f} mm: nothing between the fingers")
            self._set_state(PipelineState.ABORTED)
        else:
            self.get_logger().warn("gripper did not settle (no finger joint states?)")
            self._set_state(PipelineState.ABORTED)

    def _camera_velocity(self) -> Optional[np.ndarray]:
        """Measured camera translation velocity (camera optical frame) since the
        previous servo step, from TF; None on the first step or without TF."""
        T = self._lookup(self.params.mpc_world_frame, self.params.camera_optical_frame)
        if T is None:
            return None
        now, prev = self._now(), self._servo_cam_prev
        self._servo_cam_prev = (now, T)
        if prev is None or now - prev[0] < 1e-3 or now - prev[0] > 0.5:
            return None
        return T[:3, :3].T @ (T[:3, 3] - prev[1][:3, 3]) / (now - prev[0])

    def _lookup(self, target: str, source: str) -> Optional[np.ndarray]:
        """4x4 transform target <- source (latest), or None."""
        try:
            tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        from scipy.spatial.transform import Rotation as R_scipy
        q = tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = R_scipy.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y,
                    tf.transform.translation.z]
        return T

    def _servo_geometry(self):
        """(target, TCP, approach axis) in the camera optical frame, or None.

        Target: the live /stem_grasp/target_point when fresh, else the grasp
        point chosen before the reach (world-fixed)."""
        cam = self.params.camera_optical_frame
        if (self.last_target_point is not None
                and self._age(self.last_target_stamp) < self.params.target_point_max_age_sec):
            T = self._lookup(cam, self.params.planning_frame)
            target = None if T is None else T[:3, :3] @ self.last_target_point + T[:3, 3]
        elif self.grasp_target_world is not None:
            T = self._lookup(cam, self.params.mpc_world_frame)
            target = None if T is None else T[:3, :3] @ self.grasp_target_world + T[:3, 3]
        else:
            return None
        T_ce = self._lookup(cam, self.params.eef_frame)
        if target is None or T_ce is None:
            return None
        approach = T_ce[:3, 2]
        tcp = T_ce[:3, 3] + self.params.tcp_offset_m * approach
        return target, tcp, approach

    # ----------------------------------------------------------------- helpers
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _age(self, t: Optional[float]) -> float:
        if t is None:
            return float("inf")
        return self._now() - t

    def _publish_twist(self, vel: np.ndarray, frame_id: str) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
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
            servoing = (PipelineState.SERVOING, PipelineState.APPROACHING)
            if new_state in servoing and self.servo is not None:
                # fresh observer and last command for every servo phase
                self.servo = self._make_servo()
                self._servo_mask_stamp = None
                self._servo_cam_prev = None
            if new_state in servoing and self.mppi_servo is not None and self.state not in servoing:
                self.mppi_servo.reset()
            self.approach = None
            if new_state == PipelineState.APPROACHING:
                self.approach = IterativeApproach(self._approach_config(), start_t=self._now())
            self.state = new_state
        self._publish_state()        # every transition, not only on the 2 Hz timer

    def _publish_state(self) -> None:
        msg = String()
        msg.data = self.state.name
        self.pub_state.publish(msg)


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(args=None) -> None:
    # Handle SIGINT/SIGTERM here: with rclpy's handler the context goes away
    # under the multi-threaded executor's worker threads at shutdown.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = StemGraspPipeline()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):   # Ctrl-C / ros2 launch shutdown
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
