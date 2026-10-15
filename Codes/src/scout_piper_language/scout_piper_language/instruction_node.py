"""ROS 2 node: operator instructions -> confirmed targets for the existing stack (Phase 5).

Inputs
  /operator/instruction   std_msgs/String   text (typed in the GUI, or a transcript)
  /operator/confirm       std_msgs/Empty    commit the pending proposal now
  /operator/cancel        std_msgs/Empty    drop the pending proposal
  /operator/abort         std_msgs/Empty    software stop (``safety_stop.py``) + drop
  detections_topic        std_msgs/String   /stem_grasp/detections (JSON, camera frame)
  field_topic             SemanticDistanceField (optional) for the reachability filter
  camera_info_topic, /odom, /joint_states, TF

Outputs
  /operator/say       std_msgs/String   plain-language messages (GUI log, TTS)
  /operator/proposal  std_msgs/String   JSON event stream of the session (proposal, question,
                                        rejected, commit, cancelled, command)
  /operator/state     std_msgs/String   JSON session state at 5 Hz (countdown remaining)
  /operator/target    geometry_msgs/PointStamped  committed grasp target (world_frame): the
                                        stem_grasp pipeline's ``operator_target_topic``
  /piper_jepa/init_mask  sensor_msgs/Image mono8  the chosen detection's mask (re-grounds the
                                        Stage A target memory)
  services called on commit: pipeline ``~/scan`` (grasp, scan), ``~/release`` (release),
  /active_perception/plan_view (inspect, when running)

Language: ``parser.py`` + ``grounding.py`` (or ``vlm_url`` for a VLM behind HTTP,
validated against the detections), reachability: ``reachability.py``, the
countdown and clarification: ``session.py``.
"""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from .grounding import Detection
from .reachability import ReachabilityChecker, ReachConfig
from .safety_stop import SafetyStop
from .session import InstructionSession
from .vlm import HttpProposer, RuleProposer


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


def polygon_mask(polygon, hw) -> np.ndarray:
    import cv2  # noqa: PLC0415
    m = np.zeros(tuple(int(v) for v in hw), np.uint8)
    if polygon and len(polygon) >= 3:
        cv2.fillPoly(m, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 255)
    return m


class InstructionNode(Node):
    def __init__(self) -> None:
        super().__init__("operator_instruction")
        self.declare_parameters("", [
            ("world_frame", "odom"),
            ("detections_topic", "/stem_grasp/detections"),
            ("detections_max_age_s", 2.0),
            ("field_topic", "/scene_repr/distance_field"),
            ("camera_info_topic", "/camera/color/camera_info"),
            ("camera_frame", "camera_color_optical_frame"),
            ("joint_names", [f"piper_joint{i}" for i in range(1, 7)]),
            ("tcp_offset_m", 0.14),
            ("countdown_s", 2.0),
            ("vlm_url", ""),
            ("vlm_timeout_s", 3.0),
            ("pipeline_node", "/stem_grasp_pipeline"),
            ("active_perception_service", "/active_perception/plan_view"),
            ("gripper_open_m", 0.07),
            ("max_base_move_m", 2.0),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.p = p
        self.world = p("world_frame")
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        from scout_piper_whole_body_mpc.dynamics.piper import PiperKinematics  # noqa: PLC0415
        from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel  # noqa: PLC0415
        model = WholeBodyModel(kin=PiperKinematics(tcp_offset_m=float(p("tcp_offset_m"))))
        checker = ReachabilityChecker(model, ReachConfig(gripper_open_m=float(p("gripper_open_m")),
                                                         max_base_move_m=float(p("max_base_move_m"))))
        rules = RuleProposer()
        proposer = HttpProposer(p("vlm_url"), float(p("vlm_timeout_s")), rules) if p("vlm_url") else rules
        self.session = InstructionSession(proposer, checker, float(p("countdown_s")),
                                          width_fn=lambda d: d.width_m(self.fx))
        self.stop = SafetyStop(self)
        self.dets_msg: Optional[dict] = None
        self.dets_t = -np.inf
        self.snap = None
        self.fx: Optional[float] = None
        self.base = None
        self.q = None
        self.joint_names = list(p("joint_names"))

        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(String, "/operator/instruction", self._instruction, 10)
        self.create_subscription(Empty, "/operator/confirm", lambda m: self._emit(self.session.confirm()), 10)
        self.create_subscription(Empty, "/operator/cancel", lambda m: self._emit(self.session.cancel()), 10)
        self.create_subscription(Empty, "/operator/abort", lambda m: self._emit(self.session.abort("operator abort")), 10)
        self.create_subscription(String, p("detections_topic"), self._dets, be)
        self.create_subscription(CameraInfo, p("camera_info_topic"), lambda m: setattr(self, "fx", float(m.k[0])), be)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(JointState, "/joint_states", self._js, 20)
        if p("field_topic"):
            try:
                from scout_piper_scene_repr.msg import SemanticDistanceField  # noqa: PLC0415
                from scout_piper_scene_repr_py.field import snapshot_from_msg  # noqa: PLC0415
                self.create_subscription(SemanticDistanceField, p("field_topic"),
                                         lambda m: setattr(self, "snap", snapshot_from_msg(m)), 1)
            except ImportError:
                self.get_logger().warn("scout_piper_scene_repr not found: reachability without the scene map")
        self.pub_say = self.create_publisher(String, "/operator/say", 10)
        self.pub_events = self.create_publisher(String, "/operator/proposal", 10)
        self.pub_state = self.create_publisher(String, "/operator/state", 5)
        self.pub_target = self.create_publisher(PointStamped, "/operator/target", 5)
        self.pub_mask = self.create_publisher(Image, "/piper_jepa/init_mask", 2)
        pn = str(p("pipeline_node")).rstrip("/")
        self.cl_scan = self.create_client(Trigger, f"{pn}/scan")
        self.cl_release = self.create_client(Trigger, f"{pn}/release")
        self.cl_view = self.create_client(Trigger, str(p("active_perception_service")))
        self.create_timer(0.05, lambda: self._emit(self.session.tick()))
        self.create_timer(0.2, self._publish_state)
        self.get_logger().info("operator instruction node ready" + (f" (VLM at {p('vlm_url')})" if p("vlm_url") else ""))

    # ---------------------------------------------------------------- inputs
    def _odom(self, m: Odometry) -> None:
        pz = m.pose.pose
        self.base = np.array([pz.position.x, pz.position.y, _yaw(pz.orientation)])

    def _js(self, m: JointState) -> None:
        idx = {n: i for i, n in enumerate(m.name)}
        if all(n in idx for n in self.joint_names):
            self.q = np.array([m.position[idx[n]] for n in self.joint_names])

    def _dets(self, m: String) -> None:
        self.dets_msg = json.loads(m.data)
        self.dets_t = self.get_clock().now().nanoseconds * 1e-9

    def _lookup(self, target: str, source: str) -> Optional[np.ndarray]:
        try:
            return _tf_to_T(self.tf_buf.lookup_transform(target, source, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def _detections(self) -> List[Detection]:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.dets_msg is None or now - self.dets_t > float(self.p("detections_max_age_s")):
            return []
        T = self._lookup(self.world, self.dets_msg.get("frame_id", self.p("camera_frame")))
        out = []
        for d in self.dets_msg.get("detections", []):
            pw = None
            if d.get("p_camera") is not None and T is not None:
                pw = T[:3, :3] @ np.asarray(d["p_camera"], float) + T[:3, 3]
            out.append(Detection.from_dict(d, pw))
        return out

    def _state(self) -> Optional[np.ndarray]:
        return None if self.base is None or self.q is None else np.r_[self.base, self.q]

    def _instruction(self, m: String) -> None:
        dets = self._detections()
        x = self._state()
        kw = {}
        if self.snap is not None:
            kw["snapshot"] = self.snap
        Tc = self._lookup(self.world, self.p("camera_frame"))
        if Tc is not None:
            kw["camera_pos"] = Tc[:3, 3]
        self.get_logger().info(f"instruction: {m.data!r} ({len(dets)} detections)")
        self._emit(self.session.instruction(m.data, dets, x, kw))

    # --------------------------------------------------------------- outputs
    def _emit(self, events) -> None:
        for e in events:
            if e["type"] == "say":
                self.pub_say.publish(String(data=e["text"]))
                self.get_logger().info(f"say: {e['text']}")
            else:
                self.pub_events.publish(String(data=json.dumps(e)))
            if e["type"] == "command" and e["name"] == "stop":
                self.stop.trigger(e.get("reason", "operator"))
            elif e["type"] == "commit":
                self._commit(e)

    def _call(self, client, what: str) -> None:
        if not client.service_is_ready():
            self.pub_say.publish(String(data=f"({what} is not running.)"))
            return
        client.call_async(Trigger.Request())

    def _commit(self, e: dict) -> None:
        t = e.get("target")
        if t is not None:
            hw = (self.dets_msg or {}).get("image_hw")
            if hw and t.get("polygon"):
                img = Image()
                mask = polygon_mask(t["polygon"], hw)
                img.header.stamp = self.get_clock().now().to_msg()
                img.header.frame_id = (self.dets_msg or {}).get("frame_id", "")
                img.height, img.width, img.encoding, img.step = mask.shape[0], mask.shape[1], "mono8", mask.shape[1]
                img.data = mask.tobytes()
                self.pub_mask.publish(img)
            if t.get("p_world") is not None and e["action"] == "grasp":
                ps = PointStamped()
                ps.header.frame_id, ps.header.stamp = self.world, self.get_clock().now().to_msg()
                ps.point.x, ps.point.y, ps.point.z = map(float, t["p_world"])
                self.pub_target.publish(ps)
        if e["action"] in ("grasp", "scan"):
            self._call(self.cl_scan, "the grasp pipeline")
        elif e["action"] == "release":
            self._call(self.cl_release, "the grasp pipeline")
        elif e["action"] == "inspect":
            self._call(self.cl_view, "active perception")

    def _publish_state(self) -> None:
        st = self.session.status()
        st["detections"] = len(self._detections())
        st["stop_count"] = self.stop.count
        self.pub_state.publish(String(data=json.dumps(st)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InstructionNode()
    try:
        rclpy.spin(node)
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
