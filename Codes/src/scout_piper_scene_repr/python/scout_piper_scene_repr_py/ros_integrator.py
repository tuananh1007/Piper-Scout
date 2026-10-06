"""Attach a SemanticVoxelMap to any rclpy node (in-process geometry).

Used by ``scene_query_node`` and by controllers that query distances in their
own process (research/semantic_scene §11). ROS imports are local to this
module so the numpy core stays importable without ROS.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from .voxel_map import SemanticVoxelMap, grid_around


def image_array(msg: Image) -> np.ndarray:
    dt = {"16UC1": np.uint16, "32FC1": np.float32, "mono8": np.uint8, "8UC1": np.uint8}[msg.encoding]
    item = np.dtype(dt).itemsize
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    return a[:, : msg.width * item].copy().view(dt).reshape(msg.height, msg.width)


def transform_matrix(tr) -> np.ndarray:
    q, t = tr.rotation, tr.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class RosSceneIntegrator:
    def __init__(self, node, world_frame: str = "odom",
                 depth_topic: str = "/camera/aligned_depth_to_color/image_raw",
                 camera_info_topic: str = "/camera/aligned_depth_to_color/camera_info",
                 classes: Sequence[str] = ("stem", "branch", "leaf", "target"),
                 grid_center=(0.6, 0.0, 0.6), half_extent_m: float = 0.5,
                 voxel_size_m: float = 0.01, stride: int = 4, max_range_m: float = 1.0):
        self.node, self.world, self.stride = node, world_frame, stride
        self.map = SemanticVoxelMap(grid_around(grid_center, half_extent_m, voxel_size_m),
                                    classes=tuple(classes) + ("other",), max_range_m=max_range_m)
        self.depth: Optional[Image] = None
        self.masks: Dict[str, np.ndarray] = {}
        self.K: Optional[np.ndarray] = None
        self.cam_frame: Optional[str] = None
        self.last_integrate_ms = 0.0
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, node)
        be = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(Image, depth_topic, self._depth, be)
        node.create_subscription(CameraInfo, camera_info_topic, self._info, be)
        for c in classes:
            node.create_subscription(Image, f"/scene_repr/mask/{c}",
                                     lambda m, c=c: self.masks.__setitem__(c, image_array(m) > 0), be)

    def _info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.cam_frame = msg.header.frame_id

    def _depth(self, msg: Image) -> None:
        self.depth = msg

    def integrate_latest(self) -> bool:
        """Fuse the newest depth frame; returns False if inputs/TF are missing."""
        if self.depth is None or self.K is None or self.cam_frame is None:
            return False
        msg = self.depth
        try:
            tr = self.tf.lookup_transform(self.world, self.cam_frame,
                                          rclpy.time.Time.from_msg(msg.header.stamp))
        except Exception:  # noqa: BLE001 — TF gaps are routine
            return False
        d = image_array(msg).astype(np.float32)
        if msg.encoding == "16UC1":
            d /= 1000.0
        d[d <= 0] = np.nan
        masks = {c: m for c, m in self.masks.items() if m.shape == d.shape}
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        t0 = time.perf_counter()
        self.map.integrate(d, masks, self.K, transform_matrix(tr.transform), stamp, self.stride)
        self.last_integrate_ms = 1e3 * (time.perf_counter() - t0)
        self.depth = None
        return True
