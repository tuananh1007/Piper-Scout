#!/usr/bin/env python3
"""scene_query_node — CPU semantic voxel map, visualisation and timing probe.

Fuses aligned depth with the per-class masks from ``class_demux_node`` into a
``SemanticVoxelMap`` (via ``RosSceneIntegrator``) and publishes what it holds.
Controllers that need distances at high rate embed the same integrator in
their own process (research plan §11).

Outputs
  /scene_repr/voxels      visualization_msgs/MarkerArray (one CUBE_LIST per class)
  /scene_repr/map_status  std_msgs/String JSON: stamp, version, occupied voxel
                          counts, integrate_ms
"""

from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from scout_piper_scene_repr_py.ros_integrator import RosSceneIntegrator

COLORS = {"stem": (0.55, 0.35, 0.15), "branch": (0.4, 0.25, 0.1),
          "leaf": (0.2, 0.75, 0.25), "target": (0.95, 0.2, 0.6), "other": (0.6, 0.6, 0.6)}


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
        self.scene = RosSceneIntegrator(
            self, world_frame=p("world_frame"), depth_topic=p("depth_topic"),
            camera_info_topic=p("camera_info_topic"), classes=list(p("classes")),
            grid_center=list(p("grid_center")), half_extent_m=float(p("grid_half_extent_m")),
            voxel_size_m=float(p("voxel_size_m")), stride=int(p("stride")),
            max_range_m=float(p("max_range_m")))
        self.pub_vox = self.create_publisher(MarkerArray, "/scene_repr/voxels", 1)
        self.pub_status = self.create_publisher(String, "/scene_repr/map_status", 1)
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)

    def _tick(self) -> None:
        if self.scene.integrate_latest():
            self._publish()

    def _publish(self) -> None:
        vmap = self.scene.map
        stamp = self.get_clock().now().to_msg()
        ma, counts = MarkerArray(), {}
        vs = vmap.spec.voxel_size
        for i, c in enumerate(vmap.classes):
            pts = vmap.occupied_points(c)
            counts[c] = int(len(pts))
            m = Marker()
            m.header.frame_id, m.header.stamp = self.scene.world, stamp
            m.ns, m.id, m.type, m.action = c, i, Marker.CUBE_LIST, Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = vs
            m.pose.orientation.w = 1.0
            r, g, b = COLORS.get(c, (1, 1, 1))
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.8)
            m.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in pts[:20000]]
            ma.markers.append(m)
        self.pub_vox.publish(ma)
        self.pub_status.publish(String(data=json.dumps({
            "stamp": vmap.stamp, "version": vmap.version, "frame": self.scene.world,
            "voxel_size_m": vs, "occupied": counts,
            "integrate_ms": round(self.scene.last_integrate_ms, 1)})))


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
