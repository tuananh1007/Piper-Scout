"""ROS 2 whole-body MPC node (geometry-only baseline, W3/W4).

Pipeline per cycle:  state (odom + joint_states) → MPPI → safety filter → commands.

Inputs
  /odom                          nav_msgs/Odometry       Scout pose in world_frame
  /joint_states                  sensor_msgs/JointState  arm joints (joint_names)
  goal_topic                     geometry_msgs/PointStamped grasp point (world_frame);
                                 e.g. /piper_jepa/target_point
  geometry (use_semantic_scene)  in-process SemanticVoxelMap from aligned depth +
                                 /scene_repr/mask/<class> (scout_piper_scene_repr)

Outputs
  execute == false (default): /whole_body_mpc/preview/cmd_vel, /whole_body_mpc/preview/joint_jog
  execute == true:            /cmd_vel (scout_ros2) and /servo_node/delta_joint_cmds (moveit_servo)
  /whole_body_mpc/status         std_msgs/String JSON (cost, safety reason, timing, ages)
  /whole_body_mpc/plan           nav_msgs/Path of the predicted TCP trajectory

Executing on hardware requires the Phase 0 exit criteria (E-stop and
stop-and-zero validated) — keep execute false until then.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .costs.terms import CostWeights, Goal, WholeBodyCost
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
        self.mppi = MPPI(self.model, MPPIConfig(horizon=int(p("horizon")), samples=int(p("samples")),
                                                iterations=int(p("iterations")),
                                                temperature=float(p("temperature")),
                                                refine_iters=int(p("refine_iters"))))
        # the planner keeps plan_margin_m more clearance than the safety filter enforces
        self.weights = CostWeights(base=float(p("w_base")),
                                   d_safe=float(p("d_safe")) + max(float(p("plan_margin_m")), 0.0))
        self.d_safe = float(p("d_safe"))
        self.max_state_age = float(p("max_state_age_s"))
        self.max_geom_age = float(p("max_geometry_age_s"))
        self.tol = float(p("goal_tolerance_m"))
        self.handoff = float(p("handoff_distance_m"))

        self.scene = None
        if p("use_semantic_scene"):
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
        self.goal: Optional[np.ndarray] = None
        self.u_prev = np.zeros(8)

        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(JointState, "/joint_states", self._js, 20)
        self.create_subscription(PointStamped, p("goal_topic"), self._goal, 5)
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

    def _goal(self, msg: PointStamped) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.world:
            self.get_logger().warn(f"goal in {msg.header.frame_id}, expected {self.world}; ignored")
            return
        self.goal = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.mppi.reset()

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

    def _tick(self) -> None:
        now = self._now()
        if self.base is None or self.q is None or self.goal is None:
            return
        state_age = now - min(self.t_base, self.t_q)
        dist_fn = leaf_fn = None
        geom_age = 0.0
        if self.scene is not None:
            self.scene.integrate_latest()
            geom_age = now - self.scene.map.stamp if np.isfinite(self.scene.map.stamp) else np.inf
            dist_fn = semantic_distance_fn(self.query, now=self.scene.map.stamp)
            leaf_fn = semantic_leaf_fn(self.query)
        x = np.r_[self.base, self.q]
        cost = WholeBodyCost(self.model, Goal(p=self.goal), distance_fn=dist_fn,
                             leaf_fn=leaf_fn, w=self.weights)
        safety = SafetyFilter(self.model, distance_fn=dist_fn, d_safe=self.d_safe,
                              max_state_age_s=self.max_state_age, max_geometry_age_s=self.max_geom_age)
        err = float(np.linalg.norm(self.model.tcp_world(x)[:3, 3] - self.goal))
        t0 = time.perf_counter()
        if err < self.tol:
            u_mpc, mode = np.zeros(8), "reached"
        else:
            u_mpc = self.mppi.solve(x, cost, self.u_prev)
            mode = "handoff_ready" if err < self.handoff else "whole_body"
        solve_ms = 1e3 * (time.perf_counter() - t0)
        rep = safety.project(x, u_mpc, self.u_prev, state_age_s=state_age, geometry_age_s=geom_age)
        self._send(rep.u)
        self.u_prev = rep.u
        self.mppi.shift()
        self._publish(err, mode, rep, solve_ms, state_age, geom_age)

    def _publish(self, err, mode, rep, solve_ms, state_age, geom_age) -> None:
        self.pub_status.publish(String(data=json.dumps({
            "mode": mode, "tcp_error_m": err, "safety": rep.reason, "scale": rep.scale,
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WholeBodyMpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
