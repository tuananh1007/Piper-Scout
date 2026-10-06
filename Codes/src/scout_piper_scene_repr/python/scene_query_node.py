#!/usr/bin/env python3
"""scene_query_node — CPU semantic voxel map + distance-query snapshot.

Fuses aligned depth with the per-class masks from ``class_demux_node`` into a
``SemanticVoxelMap`` in ``world_frame`` and publishes what it holds. Controllers
that need distances at high rate should embed ``SemanticVoxelMap`` /
``SemanticDistanceQuery`` in their own process (in-process queries, research
plan §11); this node is the reference integrator, visualiser and timing probe.

Inputs
  depth_topic                    aligned depth (16UC1 mm / 32FC1 m)
  camera_info_topic              intrinsics
  /scene_repr/mask/<class>       mono8 masks from class_demux_node
  TF                             world_frame ← camera optical frame

Outputs
  /scene_repr/voxels             visualization_msgs/MarkerArray (one CUBE_LIST per class)
  /scene_repr/map_status         std_msgs/String JSON: stamp, version, occupied
                                 voxel counts, integrate_ms
"""

from __future__ import annotations

import json
import time
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from scout_piper_scene_repr_py.voxel_map import SemanticVoxelMap, grid_around

COLORS = {"stem": (0.55, 0.35, 0.15), "branch": (0.4, 0.25, 0.1),
          "leaf": (0.2, 0.75, 0.25), "target": (0.95, 0.2, 0.6), "other": (0.6, 0.6, 0.6)}


def _img(msg: Image) -> np.ndarray:
    dt = {"16UC1": np.uint16, "32FC1": np.float32, "mono8": np.uint8, "8UC1": np.uint8}[msg.encoding]
    item = np.dtype(dt).itemsize
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    return a[:, : msg.width * item].copy().view(dt).reshape(msg.height, msg.width)


def _T(tr) -> np.ndarray:
    q, t = tr.rotation, tr.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class SceneQueryNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_repr_query")
        self.declare_parameters("", [
            ("depth_topic", "/camera/aligned_depth_to_color/image_raw"),
            ("camera_info_topic", "/camera/aligned_depth_to_color/camera_info"),
            ("classes", ["stem", "branch", "leaf", "target"]),
            ("world_frame", "odom"),
            ("grid_center", [0.6, 0.0, 0.6]),
            ("grid_half_extent_m", 0.5),
            ("voxel_size_m", 0.01),
            ("stride", 4),
            ("rate_hz", 5.0),
            ("max_range_m", 1.0),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.classes = list(p("classes"))
        self.world = p("world_frame")
        self.stride = int(p("stride"))
        self.map = SemanticVoxelMap(
            grid_around(p("grid_center"), float(p("grid_half_extent_m")), float(p("voxel_size_m"))),
            classes=tuple(self.classes) + ("other",), max_range_m=float(p("max_range_m")))
        self.depth: Optional[Image] = None
        self.masks: Dict[str, np.ndarray] = {}
        self.K: Optional[np.ndarray] = None
        self.cam_frame: Optional[str] = None
        self.last_ms = 0.0

        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        be = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, p("depth_topic"), self._depth_cb, be)
        self.create_subscription(CameraInfo, p("camera_info_topic"), self._info_cb, be)
        for c in self.classes:
            self.create_subscription(Image, f"/scene_repr/mask/{c}",
                                     lambda m, c=c: self.masks.__setitem__(c, _img(m) > 0), be)
        self.pub_vox = self.create_publisher(MarkerArray, "/scene_repr/voxels", 1)
        self.pub_status = self.create_publisher(String, "/scene_repr/map_status", 1)
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)

    def _info_cb(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.cam_frame = msg.header.frame_id

    def _depth_cb(self, msg: Image) -> None:
        self.depth = msg

    def _tick(self) -> None:
        if self.depth is None or self.K is None or self.cam_frame is None:
            return
        msg = self.depth
        try:
            tr = self.tf.lookup_transform(self.world, self.cam_frame,
                                          rclpy.time.Time.from_msg(msg.header.stamp))
        except Exception:  # noqa: BLE001
            return
        d = _img(msg).astype(np.float32)
        if msg.encoding == "16UC1":
            d /= 1000.0
        d[d <= 0] = np.nan
        masks = {c: m for c, m in self.masks.items() if m.shape == d.shape}
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        t0 = time.perf_counter()
        self.map.integrate(d, masks, self.K, _T(tr.transform), stamp, self.stride)
        self.last_ms = 1e3 * (time.perf_counter() - t0)
        self._publish(msg.header.stamp, stamp)

    def _publish(self, ros_stamp, stamp: float) -> None:
        ma = MarkerArray()
        counts = {}
        vs = self.map.spec.voxel_size
        for i, c in enumerate(self.map.classes):
            pts = self.map.occupied_points(c)
            counts[c] = int(len(pts))
            m = Marker()
            m.header.frame_id, m.header.stamp = self.world, ros_stamp
            m.ns, m.id, m.type, m.action = c, i, Marker.CUBE_LIST, Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = vs
            m.pose.orientation.w = 1.0
            r, g, b = COLORS.get(c, (1, 1, 1))
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.8)
            m.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in pts[:20000]]
            ma.markers.append(m)
        self.pub_vox.publish(ma)
        self.pub_status.publish(String(data=json.dumps({
            "stamp": stamp, "version": self.map.version, "frame": self.world,
            "voxel_size_m": vs, "occupied": counts, "integrate_ms": round(self.last_ms, 1)})))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SceneQueryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
