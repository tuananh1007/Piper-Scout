"""ROS 2 node: uncertainty-driven next view for the wrist camera (Phase 4).

Inputs
  field_topic          scout_piper_scene_repr/SemanticDistanceField  occupancy and unknown space
  target_topic         geometry_msgs/PointStamped   grasp target (Stage A memory's p_world)
  target_state_topic   std_msgs/String (JSON)       /piper_jepa/target_state → U_JEPA
  twin_state_topic     std_msgs/String (JSON)       /plant_twin/state → U_twin
  stem_cloud_topic     sensor_msgs/PointCloud2      target stem points (stem_grasp), cropped
                                                    around the target: the stem joint estimate
  camera_info_topic    sensor_msgs/CameraInfo       intrinsics for the view prediction
  /odom, /joint_states                              whole-body state (as the MPC reads it)
  TF                   world ← cloud frame, flange_frame → camera_frame (hand-eye)

Outputs
  /active_perception/status     std_msgs/String (JSON)  U terms (``uncertainty.py``), needs_view,
                                                        joint estimate, last choice
  /active_perception/next_view  geometry_msgs/PoseStamped  TCP pose of the chosen view in
                                                        world_frame, z = approach axis: the
                                                        MPC's goal_pose_topic convention
  /active_perception/next_view_camera  PoseStamped      camera pose of that view (RViz)
  ~/plan_view  std_srvs/Trigger  plan and publish a view now
  ~/reset      std_srvs/Trigger  forget the joint estimate (new target)

A stem cloud enters the estimate as a new view only when the camera moved
``view_min_baseline_m`` or turned ``view_min_angle_deg`` from every earlier
view (repeated frames from one pose share its pose error and would make the
estimate overconfident). With ``auto_publish`` a view is published whenever a
required term is unmet, at most every ``auto_period_s``; the MPC follows it
with ``goal_pose_topic:=/active_perception/next_view``.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from .candidates import camera_pose_fn, sample_views
from .grid import OccupancyGrid, blocked_distance_fn, corridor_mask
from .nbv import ViewPlanner
from .stem_joint import NoiseModel, StemJointEstimator
from .uncertainty import Weights, assess, u_jepa_from_state, u_twin_from_state
from .vggt import is_novel_view


def _tf_to_T(tr) -> np.ndarray:
    q, t = tr.rotation, tr.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _yaw(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def _pose_msg(T: np.ndarray, frame: str, stamp) -> PoseStamped:
    R = T[:3, :3]
    w = np.sqrt(max(1.0 + np.trace(R), 1e-12)) / 2.0
    if w > 1e-3:
        x, y, z = (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
    else:                                             # rotation by ~π: largest diagonal element
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        v = np.zeros(3)
        v[i] = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12)) / 2.0
        v[j] = (R[j, i] + R[i, j]) / (4 * v[i])
        v[k] = (R[k, i] + R[i, k]) / (4 * v[i])
        w = (R[k, j] - R[j, k]) / (4 * v[i])
        x, y, z = v
    m = PoseStamped()
    m.header.frame_id, m.header.stamp = frame, stamp
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, T[:3, 3])
    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, (x, y, z, w))
    return m


class ActivePerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("active_perception")
        self.declare_parameters("", [
            ("world_frame", "odom"),
            ("flange_frame", "piper_link6"),
            ("camera_frame", "camera_color_optical_frame"),
            ("joint_names", [f"piper_joint{i}" for i in range(1, 7)]),
            ("tcp_offset_m", 0.14),
            ("field_topic", "/scene_repr/distance_field"),
            ("target_topic", "/piper_jepa/target_point"),
            ("target_state_topic", "/piper_jepa/target_state"),
            ("twin_state_topic", "/plant_twin/state"),
            ("stem_cloud_topic", "/stem_grasp/filtered_cloud"),
            ("camera_info_topic", "/camera/color/camera_info"),
            ("policy", "fused"),
            ("alpha", 0.3), ("beta", 0.5), ("gamma", 0.2),
            ("lambda_t", 0.02),
            ("lambda_turn", 0.15),
            ("required", ["joint", "corridor", "jepa"]),
            ("sigma_req_mm", 1.0),
            ("corridor_req", 0.05),
            ("corridor_length_m", 0.15),
            ("corridor_radius_m", 0.03),
            ("stem_crop_m", 0.05),
            ("stem_radius_guess_m", 0.004),
            ("half_len_m", 0.03),
            ("view_min_baseline_m", 0.03),
            ("view_min_angle_deg", 8.0),
            ("depth_sigma0_m", 0.0005), ("depth_k2", 0.002),
            ("pose_sigma_m", 0.0005), ("pose_sigma_deg", 0.1),
            ("d_min_m", 0.2), ("d_max_m", 0.45),
            ("max_base_move_m", 0.5),
            ("reach_max_m", 0.36),
            ("clearance_m", 0.03),
            ("n_samples", 120000), ("n_candidates", 200),
            ("max_field_age_s", 5.0),
            ("max_state_age_s", 1.0),
            ("target_jump_m", 0.03),
            ("rate_hz", 1.0),
            ("auto_publish", False),
            ("auto_period_s", 10.0),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.p = p
        from scout_piper_whole_body_mpc.dynamics.piper import PiperKinematics  # noqa: PLC0415
        from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel  # noqa: PLC0415
        self.model = WholeBodyModel(kin=PiperKinematics(tcp_offset_m=float(p("tcp_offset_m"))))
        self.world, self.flange, self.cam_frame = p("world_frame"), p("flange_frame"), p("camera_frame")
        self.joint_names = list(p("joint_names"))
        self.weights = Weights(float(p("alpha")), float(p("beta")), float(p("gamma")))
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        self.snap = None
        self.target: Optional[np.ndarray] = None
        self.jepa_state: Optional[dict] = None
        self.twin_state: Optional[dict] = None
        self.K: Optional[np.ndarray] = None
        self.hw = None
        self.base = None
        self.q = None
        self.t_base = self.t_q = -np.inf
        self.T_flange_cam: Optional[np.ndarray] = None
        self.est: Optional[StemJointEstimator] = None
        self.view_poses = []
        self.last_choice = None
        self.last_auto = -np.inf

        from scout_piper_scene_repr.msg import SemanticDistanceField  # noqa: PLC0415
        from scout_piper_scene_repr_py.field import snapshot_from_msg  # noqa: PLC0415
        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        # State, field and the other plain inputs in their own group: a status tick or a
        # plan takes up to a few seconds on a loaded CPU, and the robot state must not go
        # stale behind it (plan_view then refused with "no fresh whole-body state"). The
        # stem cloud and the target change the estimator, so they stay with the tick and
        # the services in the default (mutually exclusive) group.
        inputs = MutuallyExclusiveCallbackGroup()
        self.create_subscription(SemanticDistanceField, p("field_topic"),
                                 lambda m: setattr(self, "snap", snapshot_from_msg(m)), 1,
                                 callback_group=inputs)
        self.create_subscription(PointStamped, p("target_topic"), self._target, 5)
        self.create_subscription(String, p("target_state_topic"),
                                 lambda m: setattr(self, "jepa_state", json.loads(m.data)), 5,
                                 callback_group=inputs)
        self.create_subscription(String, p("twin_state_topic"),
                                 lambda m: setattr(self, "twin_state", json.loads(m.data)), 5,
                                 callback_group=inputs)
        self.create_subscription(PointCloud2, p("stem_cloud_topic"), self._cloud, be)
        self.create_subscription(CameraInfo, p("camera_info_topic"), self._info, be, callback_group=inputs)
        self.create_subscription(Odometry, "/odom", self._odom, 10, callback_group=inputs)
        self.create_subscription(JointState, "/joint_states", self._js, 20, callback_group=inputs)
        self.pub_status = self.create_publisher(String, "/active_perception/status", 5)
        self.pub_view = self.create_publisher(PoseStamped, "/active_perception/next_view", 1)
        self.pub_cam = self.create_publisher(PoseStamped, "/active_perception/next_view_camera", 1)
        self.create_service(Trigger, "~/plan_view", self._srv_plan)
        self.create_service(Trigger, "~/reset", self._srv_reset)
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)

    # ---------------------------------------------------------------- inputs
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom(self, m: Odometry) -> None:
        pz = m.pose.pose
        self.base = np.array([pz.position.x, pz.position.y, _yaw(pz.orientation)])
        self.t_base = self._now()

    def _js(self, m: JointState) -> None:
        idx = {n: i for i, n in enumerate(m.name)}
        if all(n in idx for n in self.joint_names):
            self.q = np.array([m.position[idx[n]] for n in self.joint_names])
            self.t_q = self._now()

    def _info(self, m: CameraInfo) -> None:
        self.K = np.array(m.k, float).reshape(3, 3)
        self.hw = (int(m.height), int(m.width))

    def _lookup(self, target: str, source: str, stamp=None) -> Optional[np.ndarray]:
        try:
            tr = self.tf_buf.lookup_transform(target, source, stamp or rclpy.time.Time())
        except Exception:  # noqa: BLE001 — TF not ready is routine
            return None
        return _tf_to_T(tr.transform)

    def _target(self, m: PointStamped) -> None:
        T = self._lookup(self.world, m.header.frame_id) if m.header.frame_id != self.world else np.eye(4)
        if T is None:
            return
        t = T[:3, :3] @ np.array([m.point.x, m.point.y, m.point.z]) + T[:3, 3]
        if self.target is None or np.linalg.norm(t - self.target) > float(self.p("target_jump_m")):
            self._reset_estimate(t)
        self.target = t

    def _reset_estimate(self, t: np.ndarray) -> None:
        p = self.p
        noise = NoiseModel(sigma0=float(p("depth_sigma0_m")), k2=float(p("depth_k2")),
                           f_px=float(self.K[0, 0]) if self.K is not None else 430.0,
                           pose_m=float(p("pose_sigma_m")), pose_rad=float(np.radians(p("pose_sigma_deg"))))
        self.est = StemJointEstimator(t, half_len=float(p("half_len_m")), r0=float(p("stem_radius_guess_m")),
                                      noise=noise)
        self.view_poses = []

    def _cloud(self, m: PointCloud2) -> None:
        if self.est is None or self.target is None:
            return
        T_wc = self._lookup(self.world, m.header.frame_id, rclpy.time.Time.from_msg(m.header.stamp))
        T_cam = self._lookup(self.world, self.cam_frame, rclpy.time.Time.from_msg(m.header.stamp))
        if T_wc is None or T_cam is None:
            return
        if not is_novel_view(T_cam, self.view_poses, float(self.p("view_min_baseline_m")),
                             float(self.p("view_min_angle_deg"))):
            return
        xyz = np.asarray(point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True),
                         float).reshape(-1, 3)
        P = xyz @ T_wc[:3, :3].T + T_wc[:3, 3]
        P = P[np.linalg.norm(P - self.est.cyl.p, axis=1) < float(self.p("stem_crop_m"))]
        if len(P) < 8:
            return
        self.est.add_view(P, T_cam[:3, 3])
        self.est.fit()
        self.view_poses.append(T_cam)

    # ------------------------------------------------------------- planning
    def _state(self) -> Optional[np.ndarray]:
        now = self._now()
        age = float(self.p("max_state_age_s"))
        if self.base is None or self.q is None or now - self.t_base > age or now - self.t_q > age:
            return None
        return np.r_[self.base, self.q]

    def _grid(self) -> Optional[OccupancyGrid]:
        if self.snap is None:
            return None
        if self._now() - float(self.snap.stamp) > float(self.p("max_field_age_s")):
            return None
        return OccupancyGrid.from_snapshot(self.snap)

    def _corridor(self, x: Optional[np.ndarray]):
        if self.est is None:
            return None
        p = self.est.cyl.p
        base_xy = x[:2] if x is not None else np.zeros(2)
        a = np.r_[p[:2] - base_xy, 0.0]
        a /= max(np.linalg.norm(a), 1e-9)
        return p - float(self.p("corridor_length_m")) * a, p - 0.01 * a, a

    def _assess(self, grid, x):
        now = self._now()
        sigma = self.est.sigma() if self.est is not None and self.est.fitted else 1.0
        corr = self._corridor(x)
        cu = 1.0
        if grid is not None and corr is not None:
            cm = corridor_mask(grid, corr[0], corr[1], float(self.p("corridor_radius_m")),
                               exclude=self.est.target_exclusion(0.01))
            cu = grid.unknown_fraction(cm)
        return assess(sigma, cu, u_jepa_from_state(self.jepa_state, now), u_twin_from_state(self.twin_state, now),
                      sigma_req_m=float(self.p("sigma_req_mm")) / 1000.0, corridor_req=float(self.p("corridor_req")),
                      weights=self.weights, required=list(self.p("required")))

    def _plan(self):
        """(ok, message); publishes the view on success."""
        x = self._state()
        if self.est is None or self.target is None:
            return False, "no target"
        if x is None:
            return False, "no fresh whole-body state (/odom, /joint_states)"
        if self.K is None:
            return False, "no camera_info"
        if self.T_flange_cam is None:
            self.T_flange_cam = self._lookup(self.flange, self.cam_frame)
            if self.T_flange_cam is None:
                return False, f"no TF {self.flange} -> {self.cam_frame}"
        grid = self._grid()
        if grid is None:
            return False, "no fresh distance field"
        u = self._assess(grid, x)
        cam_fn = camera_pose_fn(self.model, self.T_flange_cam)
        p = self.p
        corr = self._corridor(x)
        cand = sample_views(self.model, cam_fn, self.est.cyl.p, x, n=int(p("n_samples")),
                            reach_max_m=float(p("reach_max_m")),
                            d_range=(float(p("d_min_m")), float(p("d_max_m"))),
                            max_base_move_m=float(p("max_base_move_m")), distance_fn=blocked_distance_fn(grid),
                            clearance_m=float(p("clearance_m")), max_candidates=int(p("n_candidates")))
        planner = ViewPlanner(self.model, self.K, self.hw, lambda_t=float(p("lambda_t")),
                              lambda_turn=float(p("lambda_turn")), corridor_radius_m=float(p("corridor_radius_m")))
        twin_pt = None
        if self.twin_state and u.twin > 0:
            twin_pt = self.est.cyl.p                       # the contact region is the target's
        ch = planner.choose(str(p("policy")), cand, x, self.est, grid, u, corr[:2], corr[2], twin_pt)
        if ch is None:
            return False, f"no reachable view among {len(cand.X)} candidates"
        X = cand.X[ch["index"]]
        stamp = self.get_clock().now().to_msg()
        self.pub_view.publish(_pose_msg(self.model.tcp_world(X), self.world, stamp))
        self.pub_cam.publish(_pose_msg(cam_fn(X), self.world, stamp))
        self.last_choice = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ch.items()}
        self.last_choice.update({"candidates": len(cand.X), "arm_only": bool(cand.arm_only[ch["index"]]),
                                 "state": np.round(X, 4).tolist()})
        return True, json.dumps(self.last_choice)

    def _tick(self) -> None:
        x = self._state()
        grid = self._grid()
        u = self._assess(grid, x) if self.est is not None else None
        body = {"stamp": self._now(), "target": None if self.target is None else np.round(self.target, 4).tolist(),
                "field": grid is not None, "state": x is not None, "last_choice": self.last_choice}
        if u is not None:
            body.update(u.as_dict())
        if self.est is not None:
            c = self.est.cyl
            body["joint"] = {"fitted": self.est.fitted, "p": np.round(c.p, 4).tolist(), "d": np.round(c.d, 4).tolist(),
                             "r_mm": round(1000 * c.r, 2), "views": len(self.view_poses), "points": len(self.est.pts)}
        self.pub_status.publish(String(data=json.dumps(body)))
        if (bool(self.p("auto_publish")) and u is not None and u.needs_view
                and self._now() - self.last_auto >= float(self.p("auto_period_s"))):
            self.last_auto = self._now()
            ok, msg = self._plan()
            if not ok:
                self.get_logger().warn(f"no view: {msg}", throttle_duration_sec=10.0)

    def _srv_plan(self, req, res):
        res.success, res.message = self._plan()
        return res

    def _srv_reset(self, req, res):
        if self.target is not None:
            self._reset_estimate(self.target)
        self.last_choice = None
        res.success, res.message = True, "joint estimate reset"
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActivePerceptionNode()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
