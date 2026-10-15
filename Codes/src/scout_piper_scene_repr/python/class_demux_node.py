#!/usr/bin/env python3
"""class_demux_node — fans a multi-class semantic label image into per-class
binary masks and mask-gated depth streams suitable for nvblox integration.

Phase 1 v0 design (see ../docs/PHASE1_DESIGN.md §3, §4).

Inputs (configurable):
    /stem_grasp/semantic_label   sensor_msgs/Image (mono8 or 16UC1, label values)
    /camera/aligned_depth_to_color/image_raw  sensor_msgs/Image (16UC1 mm or 32FC1 m)

Outputs (one per class declared in semantic_classes.yaml):
    /scene_repr/mask/<class>     sensor_msgs/Image (mono8, 0/255)
    /scene_repr/depth/<class>    sensor_msgs/Image (same encoding as input,
                                                    depth zeroed where mask == 0)

A latched /scene_repr/policy String contains the YAML policy so the C++
collision plugin can pick it up without re-reading a file at runtime.

Phase 0 fallback mode: if `input_mode == "separate"`, the node subscribes
to `/stem_grasp/mask` (stem) and `/stem_grasp/target_mask` (target) instead
of the merged label image. This lets us run Phase 1 against the existing
ROS 1-style segmentation output before that node is fully ported.

`publish_other` (default true) adds the class `other`: depth outside every
class mask (pots, walls, supports), so that an nvblox mapper for it gives the
planner the non-plant geometry and the "observed" space (P1.3.1).

`self_filter` (default true) removes the robot's own gripper fingers from
every gated depth image (P1.7.8): boxes attached to TF frames
(`self_filter_boxes`, "frame:cx,cy,cz,sx,sy,sz" in metres) are projected with
the depth camera_info; pixels inside a box no farther than its far side are
zeroed. Boxes whose TF is missing are skipped.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from scout_piper_scene_repr_py.ros_integrator import transform_matrix
from scout_piper_scene_repr_py.self_filter import DEFAULT_PIPER_FINGER_BOXES, parse_boxes, robot_mask
from tf2_ros import Buffer, TransformListener
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String


CLASS_DEFAULT: Dict[str, int] = {
    "stem": 1,
    "branch": 2,
    "leaf": 3,
    "target": 4,
}


class ClassDemuxNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_repr_class_demux")

        self.declare_parameter("input_mode", "merged")  # "merged" | "separate"
        self.declare_parameter(
            "label_topic", "/stem_grasp/semantic_label"
        )
        # Must be aligned to the colour image the masks come from.
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter(
            "legacy_masks",
            ["/stem_grasp/mask:stem", "/stem_grasp/target_mask:target"],
        )
        self.declare_parameter("class_ids", [1, 2, 3, 4])
        self.declare_parameter("class_names", ["stem", "branch", "leaf", "target"])
        self.declare_parameter("policy_yaml_path", "")
        self.declare_parameter("publish_other", True)
        self.declare_parameter("self_filter", True)
        self.declare_parameter("self_filter_boxes", list(DEFAULT_PIPER_FINGER_BOXES))
        self.declare_parameter("depth_camera_info_topic", "/camera/aligned_depth_to_color/camera_info")

        self.input_mode = self.get_parameter("input_mode").value
        self.label_topic = self.get_parameter("label_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        class_ids: List[int] = list(self.get_parameter("class_ids").value)
        class_names: List[str] = list(self.get_parameter("class_names").value)
        if len(class_ids) != len(class_names):
            raise ValueError(
                "class_ids and class_names must have the same length"
            )
        self.label_map: Dict[int, str] = dict(zip(class_ids, class_names))
        self.publish_other = bool(self.get_parameter("publish_other").value)
        if self.publish_other and "other" not in class_names:
            class_names = list(class_names) + ["other"]
        self._latest_masks: Dict[str, np.ndarray] = {}
        self._boxes = parse_boxes(self.get_parameter("self_filter_boxes").value) \
            if bool(self.get_parameter("self_filter").value) else []
        self._K: Optional[np.ndarray] = None
        self._robot_mask_cache = (None, None)       # (depth stamp, mask)
        if self._boxes:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.last_depth: Optional[Image] = None

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Latched policy publisher
        self.pub_policy = self.create_publisher(
            String,
            "/scene_repr/policy",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._publish_policy()

        # Per-class publishers
        self.mask_pubs: Dict[str, "rclpy.publisher.Publisher"] = {}
        self.depth_pubs: Dict[str, "rclpy.publisher.Publisher"] = {}
        for name in class_names:
            self.mask_pubs[name] = self.create_publisher(
                Image, f"/scene_repr/mask/{name}", qos
            )
            self.depth_pubs[name] = self.create_publisher(
                Image, f"/scene_repr/depth/{name}", qos
            )

        # Depth subscriber (always)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos)
        if self._boxes:
            self.create_subscription(CameraInfo, self.get_parameter("depth_camera_info_topic").value,
                                     self._on_info, qos)

        if self.input_mode == "merged":
            self.create_subscription(
                Image, self.label_topic, self._on_label, qos
            )
            self.get_logger().info(
                f"class_demux merged mode: {self.label_topic} → "
                f"{list(class_names)}"
            )
        elif self.input_mode == "separate":
            legacy = self.get_parameter("legacy_masks").value
            for entry in legacy:
                topic, _, name = entry.partition(":")
                if name not in self.mask_pubs:
                    self.get_logger().warn(
                        f"legacy mask {topic} maps to unknown class {name!r}; "
                        "skipping."
                    )
                    continue
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, _n=name: self._on_legacy_mask(msg, _n),
                    qos,
                )
                self.get_logger().info(
                    f"class_demux separate mode: {topic} → class={name}"
                )
        else:
            raise ValueError(f"unknown input_mode: {self.input_mode!r}")

    def _publish_policy(self) -> None:
        path = self.get_parameter("policy_yaml_path").value
        msg = String()
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    msg.data = f.read()
                self.get_logger().info(f"published policy from {path}")
            except OSError as exc:
                self.get_logger().error(f"failed to read policy {path}: {exc}")
                msg.data = ""
        else:
            msg.data = ""
        self.pub_policy.publish(msg)

    def _on_depth(self, msg: Image) -> None:
        self.last_depth = msg

    def _on_info(self, msg: CameraInfo) -> None:
        self._K = np.array(msg.k, float).reshape(3, 3)

    def _robot_mask(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """Finger pixels of the current depth frame (cached per frame)."""
        if not self._boxes or self._K is None or self.last_depth is None:
            return None
        stamp = (self.last_depth.header.stamp.sec, self.last_depth.header.stamp.nanosec)
        if self._robot_mask_cache[0] == stamp:
            return self._robot_mask_cache[1]
        Ts = []
        for b in self._boxes:
            try:
                tr = self.tf_buffer.lookup_transform(self.last_depth.header.frame_id, b.frame,
                                                     rclpy.time.Time.from_msg(self.last_depth.header.stamp))
                Ts.append(transform_matrix(tr.transform))
            except Exception:  # noqa: BLE001 — no TF for this link (fake or missing): skip it
                Ts.append(None)
        d = depth.astype(np.float32)
        if self.last_depth.encoding == "16UC1":
            d = d / 1000.0
        m = robot_mask(np.where(d > 0, d, np.nan), self._K, self._boxes, Ts)
        self._robot_mask_cache = (stamp, m)
        return m

    def _on_label(self, msg: Image) -> None:
        try:
            label = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge label conversion failed: {exc}")
            return
        if label.dtype not in (np.uint8, np.uint16):
            self.get_logger().warn(
                f"label image has dtype {label.dtype}; expected uint8/uint16"
            )
            return
        for class_id, name in self.label_map.items():
            mask = (label == class_id).astype(np.uint8) * 255
            self._publish_pair(name, mask, msg.header)
        if self.publish_other:
            self._publish_pair("other", (label == 0).astype(np.uint8) * 255, msg.header)

    def _on_legacy_mask(self, msg: Image, class_name: str) -> None:
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge legacy mask conversion failed: {exc}")
            return
        self._latest_masks[class_name] = mask
        self._publish_pair(class_name, mask, msg.header)
        first = self.get_parameter("legacy_masks").value[0].partition(":")[2]
        if self.publish_other and class_name == first:
            union = np.zeros(mask.shape, bool)
            for m in self._latest_masks.values():
                if m.shape == mask.shape:
                    union |= m > 0
            self._publish_pair("other", (~union).astype(np.uint8) * 255, msg.header)

    def _publish_pair(self, name: str, mask: np.ndarray, header) -> None:
        # Publish the mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = header
        self.mask_pubs[name].publish(mask_msg)

        # Publish a mask-gated depth (depth zeroed where mask == 0)
        if self.last_depth is None:
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(
                self.last_depth, desired_encoding="passthrough"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge depth conversion failed: {exc}")
            return
        if depth.shape[:2] != mask.shape[:2]:
            return  # camera info / resolution mismatch
        gated = depth.copy()
        gated[mask == 0] = 0
        robot = self._robot_mask(depth)
        if robot is not None:
            gated[robot] = 0
        out = self.bridge.cv2_to_imgmsg(gated, encoding=self.last_depth.encoding)
        out.header = self.last_depth.header
        self.depth_pubs[name].publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClassDemuxNode()
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


if __name__ == "__main__":
    main()
