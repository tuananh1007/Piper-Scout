"""Masked point-cloud builder.

Port of ROS 1 stem_grasp_ros1/scripts/pointcloud_node.py to rclpy.

Subscribes to color, depth, mask, and (optionally) target mask. Publishes
two PointCloud2 streams:
    /stem_grasp/filtered_cloud       — stem-masked
    /stem_grasp/leaf_filtered_cloud  — target/leaf-masked (if topic active)

Phase 1 supersedes this node's role for the planning collision world (replaced
by nvblox semantic SDF via scout_piper_scene_repr), but the leaf cloud
remains useful for leaf-normal estimation in pipeline_node.
"""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

try:
    import open3d as o3d

    _O3D_OK = True
except Exception:  # pragma: no cover
    _O3D_OK = False


def _pack_rgb_float(r, g, b) -> float:
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", packed))[0]


class MaskedPointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__("stem_grasp_pointcloud")

        self.declare_parameter("publish_hz", 5.0)
        self.declare_parameter(
            "color_topic", "/camera/color/image_raw"
        )
        self.declare_parameter(
            "depth_topic", "/camera/depth/image_rect_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/color/camera_info"
        )
        self.declare_parameter("mask_topic", "/stem_grasp/mask")
        self.declare_parameter("target_mask_topic", "/stem_grasp/target_mask")
        self.declare_parameter(
            "output_frame", "camera_color_optical_frame"
        )
        self.declare_parameter("enable_open3d_postprocess", True)
        self.declare_parameter("voxel_size_m", 0.003)

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_rgb: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.latest_mask: Optional[Image] = None
        self.latest_target_mask: Optional[Image] = None

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(
            PointCloud2, "/stem_grasp/filtered_cloud", qos
        )
        self.leaf_pub = self.create_publisher(
            PointCloud2, "/stem_grasp/leaf_filtered_cloud", qos
        )

        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._info_cb,
            qos,
        )
        self.create_subscription(
            Image, self.get_parameter("color_topic").value, self._rgb_cb, qos
        )
        self.create_subscription(
            Image, self.get_parameter("depth_topic").value, self._depth_cb, qos
        )
        self.create_subscription(
            Image, self.get_parameter("mask_topic").value, self._mask_cb, qos
        )
        self.create_subscription(
            Image,
            self.get_parameter("target_mask_topic").value,
            self._target_mask_cb,
            qos,
        )

        hz = float(self.get_parameter("publish_hz").value)
        self.create_timer(1.0 / max(hz, 0.1), self._process)
        self.get_logger().info(f"masked pointcloud node ready ({hz} Hz)")

    # --------------------------------------------------------------- callbacks
    def _info_cb(self, msg: CameraInfo) -> None:
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def _rgb_cb(self, msg: Image) -> None:
        self.latest_rgb = msg

    def _depth_cb(self, msg: Image) -> None:
        self.latest_depth = msg

    def _mask_cb(self, msg: Image) -> None:
        self.latest_mask = msg

    def _target_mask_cb(self, msg: Image) -> None:
        self.latest_target_mask = msg

    # ---------------------------------------------------------------- process
    def _process(self) -> None:
        if any(
            x is None
            for x in [self.latest_rgb, self.latest_depth, self.latest_mask, self.fx]
        ):
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding="rgb8")
            depth = self.bridge.imgmsg_to_cv2(
                self.latest_depth, desired_encoding="passthrough"
            )
            mask = self.bridge.imgmsg_to_cv2(self.latest_mask, desired_encoding="mono8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        if np.nanmax(depth) > 10.0:
            depth = depth / 1000.0

        self._build_and_publish(rgb, depth, mask, self.pub, label="stem")

        if self.latest_target_mask is not None:
            try:
                target_mask = self.bridge.imgmsg_to_cv2(
                    self.latest_target_mask, desired_encoding="mono8"
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"cv_bridge conversion failed for target_mask: {exc}"
                )
                return
            self._build_and_publish(rgb, depth, target_mask, self.leaf_pub, label="leaf")

    # ------------------------------------------------------------- low-level
    def _build_and_publish(self, rgb, depth, mask, publisher, label: str) -> None:
        v_idx, u_idx = np.where(mask > 0)
        if len(v_idx) == 0:
            return
        z = depth[v_idx, u_idx]
        valid = np.isfinite(z) & (z > 0.01) & (z < 2.0)
        u_idx, v_idx, z = u_idx[valid], v_idx[valid], z[valid]
        if len(z) < 10:
            return

        x = (u_idx - self.cx) * z / self.fx
        y = (v_idx - self.cy) * z / self.fy
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        colors = rgb[v_idx, u_idx].astype(np.uint8)

        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.get_parameter("output_frame").value

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        use_o3d = _O3D_OK and bool(
            self.get_parameter("enable_open3d_postprocess").value
        )

        if use_o3d:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
                pcd = self._enhance(pcd)
                pts_f = np.asarray(pcd.points)
                clrs_f = np.asarray(pcd.colors)
                if pts_f is None or len(pts_f) == 0:
                    return
                clrs_u8 = (clrs_f * 255.0).astype(np.uint8)
                points = [
                    (float(px), float(py), float(pz),
                     _pack_rgb_float(c[0], c[1], c[2]))
                    for (px, py, pz), c in zip(pts_f, clrs_u8)
                ]
                publisher.publish(pc2.create_cloud(hdr, fields, points))
                return
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"Open3D processing failed for {label}, falling back: {exc}"
                )

        points = [
            (float(px), float(py), float(pz), _pack_rgb_float(c[0], c[1], c[2]))
            for (px, py, pz), c in zip(pts, colors)
        ]
        publisher.publish(pc2.create_cloud(hdr, fields, points))

    def _enhance(self, pcd):
        voxel_size = float(self.get_parameter("voxel_size_m").value)
        if len(pcd.points) == 0:
            return pcd
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        try:
            pcd_stat, _ = pcd_down.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            pcd_rad, _ = pcd_stat.remove_radius_outlier(nb_points=10, radius=0.01)
        except Exception:
            pcd_rad = pcd_down
        if len(pcd_rad.points) > 0:
            pcd_rad.estimate_normals(
                o3d.geometry.KDTreeSearchParamRadius(0.005)
            )
        return pcd_rad


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MaskedPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
