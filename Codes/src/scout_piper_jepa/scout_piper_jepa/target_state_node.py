"""ROS 2 node: dense target memory on the live RealSense stream.

Inputs
  rgb_topic          sensor_msgs/Image   colour frames (rgb8/bgr8)
  depth_topic        sensor_msgs/Image   depth aligned to colour (16UC1 mm / 32FC1 m)
  camera_info_topic  sensor_msgs/CameraInfo
  init_mask_topic    sensor_msgs/Image   mono8 target mask; every message
                                         (re)initialises the memory — this is
                                         the re-grounding entry point
  TF                 world_frame ← camera optical frame

Outputs
  /piper_jepa/target_state      std_msgs/String  JSON: status, u_mean, u_cov,
                                                 entropy, confidence, visible,
                                                 p_world, p_cov, stamp, max_age
  /piper_jepa/target_point      geometry_msgs/PointStamped  (only with valid depth)
  /piper_jepa/target_visible    std_msgs/Bool
  /piper_jepa/debug_similarity  sensor_msgs/Image mono8 (if publish_debug)

Dense features never leave this process (ROADMAP Phase 2A rule).
"""

from __future__ import annotations

import json
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from .encoder import make_encoder
from .image_codec import image_to_numpy, numpy_to_mono8
from .target_memory import TargetMemory, TargetMemoryConfig


def _quat_to_T(tr) -> np.ndarray:
    q, t = tr.rotation, tr.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class TargetStateNode(Node):
    def __init__(self) -> None:
        super().__init__("piper_jepa_target_state")
        self.declare_parameters("", [
            ("rgb_topic", "/camera/color/image_raw"),
            ("depth_topic", "/camera/aligned_depth_to_color/image_raw"),
            ("camera_info_topic", "/camera/color/camera_info"),
            ("init_mask_topic", "/piper_jepa/init_mask"),
            ("world_frame", "odom"),
            ("encoder", "color_patch"),          # 'color_patch' | 'vjepa'
            ("vjepa_hub_repo", "facebookresearch/vjepa2"),
            ("vjepa_hub_entry", ""),
            ("vjepa_image_size", 384),
            ("vjepa_device", "cuda"),
            ("clip_length", 2),
            ("temperature", 0.05),
            ("prior_sigma_px", 30.0),
            ("gate_px", 48.0),
            ("visible_similarity", 0.6),
            ("lost_after_frames", 30),
            ("max_valid_age_s", 0.5),
            ("publish_debug", False),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        kind = p("encoder")
        if kind == "vjepa":
            self.encoder = make_encoder("vjepa", hub_repo=p("vjepa_hub_repo"),
                                        hub_entry=p("vjepa_hub_entry"),
                                        image_size=int(p("vjepa_image_size")),
                                        device=p("vjepa_device"))
        else:
            self.encoder = make_encoder(kind)
        self.memory = TargetMemory(TargetMemoryConfig(
            temperature=float(p("temperature")),
            prior_sigma_px=float(p("prior_sigma_px")) if p("prior_sigma_px") > 0 else None,
            gate_px=float(p("gate_px")),
            visible_similarity=float(p("visible_similarity")),
            lost_after_frames=int(p("lost_after_frames")),
        ))
        self.world = p("world_frame")
        self.max_age = float(p("max_valid_age_s"))
        self.debug = bool(p("publish_debug"))
        self.clip = deque(maxlen=int(p("clip_length")))
        self.depth: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.cam_frame: Optional[str] = None
        self.pending_mask: Optional[np.ndarray] = None
        self.initialized = False

        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        be = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, p("rgb_topic"), self._rgb_cb, be)
        self.create_subscription(Image, p("depth_topic"), self._depth_cb, be)
        self.create_subscription(CameraInfo, p("camera_info_topic"), self._info_cb, be)
        self.create_subscription(Image, p("init_mask_topic"), self._mask_cb, 2)
        self.pub_state = self.create_publisher(String, "/piper_jepa/target_state", 5)
        self.pub_point = self.create_publisher(PointStamped, "/piper_jepa/target_point", 5)
        self.pub_vis = self.create_publisher(Bool, "/piper_jepa/target_visible", 5)
        self.pub_dbg = self.create_publisher(Image, "/piper_jepa/debug_similarity", 1)
        self.get_logger().info(f"target memory up (encoder={kind}); waiting for {p('init_mask_topic')}")

    # ------------------------------------------------------------- inputs
    def _info_cb(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.cam_frame = msg.header.frame_id

    def _depth_cb(self, msg: Image) -> None:
        d = image_to_numpy(msg).astype(np.float32)
        self.depth = d / 1000.0 if msg.encoding == "16UC1" else d

    def _mask_cb(self, msg: Image) -> None:
        self.pending_mask = image_to_numpy(msg) > 0

    def _T_world_cam(self, stamp) -> Optional[np.ndarray]:
        if self.cam_frame is None:
            return None
        try:
            tr = self.tf.lookup_transform(self.world, self.cam_frame, rclpy.time.Time.from_msg(stamp))
        except Exception:  # noqa: BLE001 — TF gaps are routine
            return None
        return _quat_to_T(tr.transform)

    def _rgb_cb(self, msg: Image) -> None:
        rgb = image_to_numpy(msg)
        if msg.encoding == "bgr8":
            rgb = rgb[..., ::-1]
        self.clip.append(np.ascontiguousarray(rgb))
        feats = self.encoder.encode(list(self.clip))
        if self.pending_mask is not None:
            try:
                self.memory.initialize(feats, self.pending_mask)
                self.initialized = True
                self.get_logger().info("target memory (re)initialised from mask")
            except ValueError as exc:
                self.get_logger().warn(f"init mask rejected: {exc}")
            self.pending_mask = None
        if not self.initialized:
            return
        stamp = msg.header.stamp
        t = stamp.sec + 1e-9 * stamp.nanosec
        s = self.memory.update(feats, stamp=t, depth=self.depth, K=self.K,
                               T_world_cam=self._T_world_cam(stamp),
                               image_hw=rgb.shape[:2], keep_prob=self.debug)
        self._publish(s, msg.header)

    # ------------------------------------------------------------ outputs
    def _publish(self, s, header) -> None:
        body = {
            "stamp": s.stamp, "max_age": self.max_age, "status": s.status,
            "visible": bool(s.visible), "confidence": s.confidence,
            "entropy": s.entropy, "entropy_norm": s.entropy_norm,
            "peak_similarity": s.peak_similarity,
            "u_mean": s.u_mean.tolist(), "u_cov": s.u_cov.tolist(),
            "p_world": None if s.p_world is None else s.p_world.tolist(),
            "p_cov": None if s.p_cov is None else s.p_cov.tolist(),
            "world_frame": self.world,
        }
        self.pub_state.publish(String(data=json.dumps(body)))
        self.pub_vis.publish(Bool(data=bool(s.visible)))
        if s.p_world is not None:
            pt = PointStamped()
            pt.header.stamp, pt.header.frame_id = header.stamp, self.world
            pt.point.x, pt.point.y, pt.point.z = map(float, s.p_world)
            self.pub_point.publish(pt)
        if self.debug and s.prob is not None:
            img = numpy_to_mono8(s.prob / max(s.prob.max(), 1e-12))
            img.header = header
            self.pub_dbg.publish(img)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetStateNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):    # Ctrl-C, or SIGINT from ros2 launch
        pass
    except Exception:
        if rclpy.ok():                                        # not a callback cut off by the shutdown
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
