#!/usr/bin/env python3
"""class_demux_node — fans a multi-class semantic label image into per-class
binary masks and mask-gated depth streams suitable for nvblox integration.

Phase 1 v0 design (see ../docs/PHASE1_DESIGN.md §3, §4).

Inputs (configurable):
    /stem_grasp/semantic_label   sensor_msgs/Image (mono8 or 16UC1, label values)
    /camera/depth/image_rect_raw sensor_msgs/Image (16UC1 mm or 32FC1 m)

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
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
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
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter(
            "legacy_masks",
            ["/stem_grasp/mask:stem", "/stem_grasp/target_mask:target"],
        )
        self.declare_parameter("class_ids", [1, 2, 3, 4])
        self.declare_parameter("class_names", ["stem", "branch", "leaf", "target"])
        self.declare_parameter("policy_yaml_path", "")

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

        self.bridge = CvBridge()
        self.last_depth: Optional[Image] = None

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Latched policy publisher
        self.pub_policy = self.create_publisher(
            String,
            "/scene_repr/policy",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
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

    def _on_legacy_mask(self, msg: Image, class_name: str) -> None:
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge legacy mask conversion failed: {exc}")
            return
        self._publish_pair(class_name, mask, msg.header)

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
        out = self.bridge.cv2_to_imgmsg(gated, encoding=self.last_depth.encoding)
        out.header = self.last_depth.header
        self.depth_pubs[name].publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClassDemuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
