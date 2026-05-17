#!/usr/bin/env python3
"""Synthetic mask + depth publisher for Phase 1 plumbing tests.

Without a live camera or YOLO/Grounded-SAM model, the class_demux_node has
no input. This node publishes:

  /stem_grasp/semantic_label  (sensor_msgs/Image, mono8, label values 1-4)
  /camera/depth/image_rect_raw (sensor_msgs/Image, 16UC1, mm)

at ~10 Hz, with synthetic geometry: a vertical green "stem" pixel column,
a circular "leaf" blob, and a small "target" disc. Depth is a constant
plane at 0.5 m.

This lets us verify:
  1. class_demux_node correctly fans out the label image into 4 per-class
     binary masks on /scene_repr/mask/<class>.
  2. The gated depth streams /scene_repr/depth/<class> appear and contain
     non-zero pixels only where the class mask is non-zero.

Run:
  ros2 run scout_piper_scene_repr test_mask_publisher.py
  ros2 launch scout_piper_scene_repr test_class_demux.launch.py     # all-in-one
"""

from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


# Class label values used by the demux. Match config/semantic_classes.yaml.
LABEL_BG = 0
LABEL_STEM = 1
LABEL_BRANCH = 2
LABEL_LEAF = 3
LABEL_TARGET = 4


class TestMaskPublisher(Node):
    def __init__(self) -> None:
        super().__init__("test_mask_publisher")
        self.declare_parameter("image_w", 640)
        self.declare_parameter("image_h", 480)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("depth_mm", 500)
        self.declare_parameter("label_topic", "/stem_grasp/semantic_label")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")

        self.w = int(self.get_parameter("image_w").value)
        self.h = int(self.get_parameter("image_h").value)
        self.depth_mm = int(self.get_parameter("depth_mm").value)
        hz = float(self.get_parameter("publish_hz").value)

        self.bridge = CvBridge()
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.pub_label = self.create_publisher(
            Image, str(self.get_parameter("label_topic").value), qos
        )
        self.pub_depth = self.create_publisher(
            Image, str(self.get_parameter("depth_topic").value), qos
        )

        self.label_img = self._build_label_image()
        self.depth_img = self._build_depth_image()
        self.create_timer(1.0 / max(hz, 0.1), self._tick)
        self.get_logger().info(
            f"Publishing synthetic mask + depth at {hz} Hz "
            f"on {self.get_parameter('label_topic').value} / "
            f"{self.get_parameter('depth_topic').value}"
        )

    def _build_label_image(self) -> np.ndarray:
        """Compose a layered scene: stem column, branch nub, leaf blob, target disc."""
        img = np.zeros((self.h, self.w), dtype=np.uint8)

        cx, cy = self.w // 2, self.h // 2

        # Leaf: large soft circle on the left
        yy, xx = np.ogrid[: self.h, : self.w]
        leaf_r = 80
        leaf_cx, leaf_cy = cx - 120, cy
        leaf = (xx - leaf_cx) ** 2 + (yy - leaf_cy) ** 2 <= leaf_r ** 2
        img[leaf] = LABEL_LEAF

        # Stem: vertical column running through the centre
        stem_w = 6
        img[:, cx - stem_w // 2 : cx + stem_w // 2 + 1] = LABEL_STEM

        # Branch: short horizontal nub off the stem (will overwrite stem briefly)
        branch_y0 = cy - 50
        img[branch_y0 : branch_y0 + 8, cx : cx + 60] = LABEL_BRANCH

        # Target: small disc at the top end of the stem
        tgt_r = 16
        tgt_cx, tgt_cy = cx, cy - 130
        tgt = (xx - tgt_cx) ** 2 + (yy - tgt_cy) ** 2 <= tgt_r ** 2
        img[tgt] = LABEL_TARGET

        return img

    def _build_depth_image(self) -> np.ndarray:
        """Constant-depth plane at depth_mm millimetres."""
        return np.full((self.h, self.w), self.depth_mm, dtype=np.uint16)

    def _tick(self) -> None:
        stamp = self.get_clock().now().to_msg()
        m = self.bridge.cv2_to_imgmsg(self.label_img, encoding="mono8")
        m.header.stamp = stamp
        m.header.frame_id = "camera_color_optical_frame"
        self.pub_label.publish(m)

        d = self.bridge.cv2_to_imgmsg(self.depth_img, encoding="16UC1")
        d.header.stamp = stamp
        d.header.frame_id = "camera_color_optical_frame"
        self.pub_depth.publish(d)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TestMaskPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
