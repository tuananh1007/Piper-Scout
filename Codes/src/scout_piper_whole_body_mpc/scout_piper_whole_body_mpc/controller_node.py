"""ROS 2 whole-body MPC node (geometry-only baseline, W3/W4).

Pipeline per cycle:  state (odom + joint_states) → MPPI → safety filter → commands.

Inputs
  /odom                          nav_msgs/Odometry       Scout pose in world_frame
  /joint_states                  sensor_msgs/JointState  arm joints (joint_names)
  goal_topic                     geometry_msgs/PointStamped grasp point (world_frame);
                                 e.g. /piper_jepa/target_point
  goal_pose_topic                geometry_msgs/PoseStamped pre-grasp pose (world_frame); its
                                 z axis is the desired TCP approach direction (stem_grasp)
  cancel_topic                   std_msgs/Empty: drop the goal, send one stop, go idle
                                 (hands the arm to another servo client)
  geometry (use_semantic_scene)  in-process SemanticVoxelMap from aligned depth +
                                 /scene_repr/mask/<class> (scout_piper_scene_repr)
  field_topic (if set)           scout_piper_scene_repr/SemanticDistanceField from
                                 scene_query_node (CPU map) or nvblox_field_bridge.py (nvblox,
                                 GPU); used instead of the in-process map

Outputs
  execute == false (default): /whole_body_mpc/preview/cmd_vel, /whole_body_mpc/preview/joint_jog
  execute == true:            /cmd_vel (scout_ros2) and /servo_node/delta_joint_cmds (moveit_servo)
  /whole_body_mpc/status         std_msgs/String JSON (mode, TCP and approach-angle error,
                                 safety reason, timing, ages); mode "idle" after a cancel
  /whole_body_mpc/plan           nav_msgs/Path of the predicted TCP trajectory

Executing on hardware requires the Phase 0 exit criteria (E-stop and
stop-and-zero validated) — keep execute false until then.

On exit (Ctrl-C, SIGTERM from ros2 launch, or an exception) the node publishes
zero base and joint velocities before shutting down: neither scout_ros2 nor
ugv_sdk stops the Scout when /cmd_vel stops arriving, so without this the base
would keep its last command (INSTALL.md 10.5).
"""

from __future__ import annotations

import collections
import json
import signal
import time
from typing import Optional

import numpy as np
import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String

from .costs.terms import CostWeights, Goal, WholeBodyCost, same_goal
from .dynamics.piper import PiperKinematics
from .dynamics.scout import ScoutParams
from .dynamics.whole_body import ArmParams, WholeBodyModel
from .safety.projection import SafetyFilter
from .scene_adapter import semantic_distance_fn, semantic_leaf_fn
from .solvers.mppi import MPPI, MPPIConfig


def _yaw(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


class WholeBodyMpcNode(Node):
    def __init__(self) -> None:
        super().__init__("whole_body_mpc")
        self.declare_parameters("", [
            ("execute", False),
            ("rate_hz", 10.0),
            ("world_frame", "odom"),
            ("goal_topic", "/whole_body_mpc/goal"),
            ("goal_pose_topic", "/whole_body_mpc/goal_pose"),
            ("cancel_topic", "/whole_body_mpc/cancel"),
            ("joint_names", [f"piper_joint{i}" for i in range(1, 7)]),
            ("tcp_offset_m", 0.14),
            ("horizon", 20), ("samples", 256), ("iterations", 2), ("temperature", 0.1),
            ("v_max", 0.3), ("omega_max", 0.6), ("qd_max", 0.6),
            ("k_v", 1.0), ("k_omega", 1.0),
            ("w_base", 100.0), ("d_safe", 0.02), ("plan_margin_m", 0.01),
            ("refine_iters", 2),
            ("use_semantic_scene", False),
            ("scene_grid_center", [0.6, 0.0, 0.6]), ("scene_half_extent_m", 0.5),
            ("scene_voxel_size_m", 0.01), ("max_state_age_s", 0.2), ("max_geometry_age_s", 1.0),
            ("goal_tolerance_m", 0.01), ("handoff_distance_m", 0.05),
            ("approach_tolerance_deg", 15.0), ("w_orient", 2.0),
            ("w_reach", 1e4), ("reach_max_m", 0.36),
            ("w_reach_advanced", 1e3), ("reach_max_advanced_m", 0.40),
            ("pose_goal_advance_m", 0.0),
            ("field_topic", ""),              # e.g. /scene_repr/distance_field ("" = off)
            ("field_max_voxel_age_s", 30.0),  # voxels older than this count as unknown
            ("unknown_policy", "no_entry"),   # no_entry | stop (see safety/projection.py)
            ("backend", "numpy"),             # numpy | torch (GPU when torch_device is cuda)
            ("torch_device", "auto"),         # auto (cuda if available) | cuda | cpu
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.execute = bool(p("execute"))
        self.world = p("world_frame")
        self.joint_names = list(p("joint_names"))
        self.model = WholeBodyModel(
            dt=1.0 / float(p("rate_hz")),
            scout=ScoutParams(k_v=float(p("k_v")), k_omega=float(p("k_omega")),
                              v_max=float(p("v_max")), omega_max=float(p("omega_max"))),
            arm=ArmParams(qd_max=float(p("qd_max"))),
            kin=PiperKinematics(tcp_offset_m=float(p("tcp_offset_m"))))
        mppi_cfg = MPPIConfig(horizon=int(p("horizon")), samples=int(p("samples")),
                              iterations=int(p("iterations")), temperature=float(p("temperature")),
                              refine_iters=int(p("refine_iters")))
        self.backend = str(p("backend"))
        if self.backend == "torch":
            from .torch_backend import TorchMPPI, default_device  # noqa: PLC0415 — needs torch
            dev = str(p("torch_device"))
            self.mppi = TorchMPPI(self.model, mppi_cfg, device=default_device() if dev == "auto" else dev)
            self.get_logger().info(f"MPPI on torch ({self.mppi.device}), {mppi_cfg.samples} samples")
        elif self.backend == "numpy":
            self.mppi = MPPI(self.model, mppi_cfg)
        else:
            raise ValueError(f"backend {self.backend!r}: numpy or torch")
        self._field_version = -1
        self.unknown_policy = str(p("unknown_policy"))
        self.field_snap = None
        self.field_max_age = float(p("field_max_voxel_age_s"))
        self._field_stamp = None
        # the planner keeps plan_margin_m more clearance than the safety filter enforces
        self.weights = CostWeights(base=float(p("w_base")), orient=float(p("w_orient")),
                                   reach=float(p("w_reach")), reach_max_m=float(p("reach_max_m")),
                                   reach_advanced=float(p("w_reach_advanced")),
                                   reach_max_advanced_m=float(p("reach_max_advanced_m")),
                                   d_safe=float(p("d_safe")) + max(float(p("plan_margin_m")), 0.0))
        self.d_safe = float(p("d_safe"))
        self.max_state_age = float(p("max_state_age_s"))
        self.max_geom_age = float(p("max_geometry_age_s"))
        self.tol = float(p("goal_tolerance_m"))
        self.handoff = float(p("handoff_distance_m"))
        self.approach_tol = float(p("approach_tolerance_deg"))
        self.pose_advance = float(p("pose_goal_advance_m"))

        if p("field_topic"):
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: PLC0415
            from scout_piper_scene_repr.msg import SemanticDistanceField  # noqa: PLC0415
            from scout_piper_scene_repr_py.field import snapshot_from_msg  # noqa: PLC0415
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(SemanticDistanceField, p("field_topic"),
                                     lambda m: setattr(self, "field_snap", snapshot_from_msg(m)), qos)
        self.scene = None
        if p("use_semantic_scene") and p("field_topic"):
            self.get_logger().warn("field_topic set: use_semantic_scene ignored")
        elif p("use_semantic_scene"):
            from scout_piper_scene_repr_py.distance_query import SemanticDistanceQuery  # noqa: PLC0415
            from scout_piper_scene_repr_py.ros_integrator import RosSceneIntegrator  # noqa: PLC0415
            self.scene = RosSceneIntegrator(self, world_frame=self.world,
                                            grid_center=list(p("scene_grid_center")),
                                            half_extent_m=float(p("scene_half_extent_m")),
                                            voxel_size_m=float(p("scene_voxel_size_m")))
            self.query = SemanticDistanceQuery(self.scene.map, max_age_s=self.max_geom_age)

        self.base: Optional[np.ndarray] = None
        self.q: Optional[np.ndarray] = None
        self.t_base = self.t_q = -np.inf
        self.goal: Optional[Goal] = None
        self.u_prev = np.zeros(8)
        self._overruns = collections.deque(maxlen=20)   # solve > 90 % of the period
        # extra cost terms (WholeBodyCost.extra), e.g. Piper-JEPA's visibility cost
        # (scout_piper_jepa predictive_mpc_node subclasses this node)
        self.extra_terms: list = []

        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(JointState, "/joint_states", self._js, 20)
        self.create_subscription(PointStamped, p("goal_topic"), self._goal, 5)
        self.create_subscription(PoseStamped, p("goal_pose_topic"), self._goal_pose, 5)
        self.create_subscription(Empty, p("cancel_topic"), self._cancel, 5)
        pre = "" if self.execute else "/whole_body_mpc/preview"
        self.pub_twist = self.create_publisher(Twist, f"{pre}/cmd_vel", 1)
        self.pub_jog = self.create_publisher(
            JointJog, "/servo_node/delta_joint_cmds" if self.execute else f"{pre}/joint_jog", 1)
        self.pub_status = self.create_publisher(String, "/whole_body_mpc/status", 5)
        self.pub_plan = self.create_publisher(Path, "/whole_body_mpc/plan", 1)
        self.create_timer(self.model.dt, self._tick)
        self.get_logger().info(f"whole-body MPC up (execute={self.execute})")

    # ------------------------------------------------------------- inputs
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom(self, msg: Odometry) -> None:
        pz = msg.pose.pose
        self.base = np.array([pz.position.x, pz.position.y, _yaw(pz.orientation)])
        self.t_base = self._now()

    def _js(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in self.joint_names):
            self.q = np.array([msg.position[idx[n]] for n in self.joint_names])
            self.t_q = self._now()

    def _set_goal(self, frame_id: str, goal: Goal) -> None:
        if frame_id and frame_id != self.world:
            self.get_logger().warn(f"goal in {frame_id}, expected {self.world}; ignored")
            return
        if not same_goal(self.goal, goal):     # republishing a goal keeps the warm start
            self.mppi.reset()
        self.goal = goal

    def _goal(self, msg: PointStamped) -> None:
        self._set_goal(msg.header.frame_id, Goal(p=np.array([msg.point.x, msg.point.y, msg.point.z])))

    def _goal_pose(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        z_axis = np.array([2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x),
                           1 - 2 * (q.x * q.x + q.y * q.y)])
        if not np.isfinite(z_axis).all() or np.linalg.norm(z_axis) < 0.5:
            self.get_logger().warn("goal pose has an invalid orientation; ignored")
            return
        pos = msg.pose.position
        self._set_goal(msg.header.frame_id, Goal(p=np.array([pos.x, pos.y, pos.z]),
                                                 approach_axis=z_axis / np.linalg.norm(z_axis),
                                                 advance_m=self.pose_advance))

    def _cancel(self, msg: Empty) -> None:
        if self.goal is None:
            return
        self.goal = None
        self.mppi.reset()
        self.stop_motion()
        self.u_prev = np.zeros(8)
        self.pub_status.publish(String(data=json.dumps({"mode": "idle", "execute": self.execute})))
        self.get_logger().info("goal cancelled; idle")

    # ---------------------------------------------------------------- loop
    def _send(self, u: np.ndarray) -> None:
        tw = Twist()
        tw.linear.x, tw.angular.z = float(u[0]), float(u[1])
        self.pub_twist.publish(tw)
        jog = JointJog()
        jog.header.stamp = self.get_clock().now().to_msg()
        jog.joint_names = self.joint_names
        jog.velocities = [float(v) for v in u[2:]]
        self.pub_jog.publish(jog)

    def stop_motion(self, repeats: int = 3) -> None:
        """Command zero base and arm velocity (repeated, in case one is dropped)."""
        for _ in range(repeats):
            self._send(np.zeros(8))
            time.sleep(0.02)

    def _tick(self) -> None:
        now = self._now()
        if self.base is None or self.q is None or self.goal is None:
            return
        state_age = now - min(self.t_base, self.t_q)
        dist_fn = leaf_fn = None
        geom_age = 0.0
        if self.field_snap is not None:
            from scout_piper_scene_repr_py.field import snapshot_distance_fn, snapshot_leaf_fn  # noqa: PLC0415
            snap = self.field_snap
            geom_age = now - snap.stamp
            dist_fn = snapshot_distance_fn(snap, now=now, max_voxel_age_s=self.field_max_age)
            leaf_fn = snapshot_leaf_fn(snap) if snap.soft_distance is not None else None
            if self.backend == "torch" and snap.stamp != self._field_stamp:
                from .torch_backend import TorchGridField  # noqa: PLC0415
                self.mppi.field = TorchGridField.from_snapshot(snap, now=now, max_age_s=self.field_max_age,
                                                               device=self.mppi.device)
                self._field_stamp = snap.stamp
        elif self.scene is not None:
            self.scene.integrate_latest()
            geom_age = now - self.scene.map.stamp if np.isfinite(self.scene.map.stamp) else np.inf
            dist_fn = semantic_distance_fn(self.query, now=self.scene.map.stamp)
            leaf_fn = semantic_leaf_fn(self.query)
            if self.backend == "torch" and self.scene.map.version != self._field_version:
                self._update_device_field()
        x = np.r_[self.base, self.q]
        cost = WholeBodyCost(self.model, self.goal, distance_fn=dist_fn,
                             leaf_fn=leaf_fn, w=self.weights, extra=list(self.extra_terms))
        safety = SafetyFilter(self.model, distance_fn=dist_fn, d_safe=self.d_safe,
                              max_state_age_s=self.max_state_age, max_geometry_age_s=self.max_geom_age,
                              unknown_policy=self.unknown_policy)
        T_tcp = self.model.tcp_world(x)
        err = float(np.linalg.norm(T_tcp[:3, 3] - self.goal.p))
        angle = None
        if self.goal.approach_axis is not None:
            angle = float(np.degrees(np.arccos(np.clip(T_tcp[:3, 2] @ self.goal.approach_axis, -1, 1))))
        t0 = time.perf_counter()
        # with a pose goal, "reached" also needs the approach axis within tolerance
        if err < self.tol and (angle is None or angle < self.approach_tol):
            u_mpc, mode = np.zeros(8), "reached"
        else:
            u_mpc = self.mppi.solve(x, cost, self.u_prev)
            mode = "handoff_ready" if err < self.handoff else "whole_body"
        solve_ms = 1e3 * (time.perf_counter() - t0)
        # sustained overruns only (5 of the last 20 steps above 90 % of the
        # period): a single slow step (e.g. the first after a new goal) is normal
        self._overruns.append(solve_ms > 900.0 * self.model.dt)
        if sum(self._overruns) >= 5:
            self.get_logger().warn(
                f"MPPI solves overrun: {sum(self._overruns)} of the last {len(self._overruns)} took "
                f"over 90 % of the {1e3 * self.model.dt:.0f} ms period (last {solve_ms:.0f} ms); "
                "lower samples (profile:=orin) or refine_iters, or use backend torch on the GPU", throttle_duration_sec=5.0)
        rep = safety.project(x, u_mpc, self.u_prev, state_age_s=state_age, geometry_age_s=geom_age)
        self._send(rep.u)
        self.u_prev = rep.u
        self.mppi.shift()
        self._publish(err, angle, mode, rep, solve_ms, state_age, geom_age)

    def _update_device_field(self) -> None:
        """Copy the semantic map's hard / leaf fields to the MPPI device (torch backend)."""
        from scout_piper_scene_repr_py.field import export_field  # noqa: PLC0415
        from .torch_backend import TorchGridField  # noqa: PLC0415
        snap = export_field(self.query)
        if snap is None:
            return
        self.mppi.field = TorchGridField.from_snapshot(
            snap, now=snap.stamp, max_age_s=self.query.max_age_s,
            leaf_weight=self.query.policies["leaf"].cost_weight if "leaf" in self.query.policies else 50.0,
            device=self.mppi.device)
        self._field_version = self.scene.map.version

    def _publish(self, err, angle, mode, rep, solve_ms, state_age, geom_age) -> None:
        self.pub_status.publish(String(data=json.dumps({
            "mode": mode, "tcp_error_m": err, "approach_error_deg": angle,
            "safety": rep.reason, "scale": rep.scale,
            "min_clearance_m": None if not np.isfinite(rep.min_clearance) else rep.min_clearance,
            "cost": self.mppi.last_cost, "solve_ms": round(solve_ms, 1),
            "state_age_s": state_age, "geometry_age_s": None if not np.isfinite(geom_age) else geom_age,
            "execute": self.execute})))
        if self.mppi.last_X is None:
            return
        path = Path()
        path.header.frame_id = self.world
        path.header.stamp = self.get_clock().now().to_msg()
        for T in self.model.tcp_world(self.mppi.last_X):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, T[:3, 3])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_plan.publish(path)


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(args=None) -> None:
    # Handle SIGINT/SIGTERM here instead of in rclpy, so the context is still up
    # when the node commands its final stop.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = WholeBodyMpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motion()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
