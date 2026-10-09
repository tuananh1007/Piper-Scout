#!/usr/bin/env python3
"""scene_query_node — CPU semantic voxel map, visualisation and timing probe.

Fuses aligned depth with the per-class masks from ``class_demux_node`` into a
``SemanticVoxelMap`` (via ``RosSceneIntegrator``) and publishes what it holds.
Controllers that need distances at high rate embed the same integrator in
their own process (research plan §11).

Outputs
  /scene_repr/voxels          visualization_msgs/MarkerArray (one CUBE_LIST per class)
  /scene_repr/map_status      std_msgs/String JSON: stamp, version, occupied voxel
                              counts, integrate_ms, export_ms
  /scene_repr/distance_field  scout_piper_scene_repr/SemanticDistanceField, at most
                              field_rate_hz, reliable + transient local; consumed by
                              the MoveIt semantic collision plugin and the whole-body MPC
  /scene_repr/target_goal     geometry_msgs/PoseStamped pre-grasp pose from the target
                              class (attractor.py; z axis = approach direction), for
                              /whole_body_mpc/goal_pose (P1.3.3)

The robot's gripper fingers are removed from each depth frame before
integration (self_filter_boxes, P1.7.8).
"""

from __future__ import annotations

import json
import os
import time

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from scout_piper_scene_repr.msg import SemanticDistanceField
from scout_piper_scene_repr_py.attractor import axis_quaternion, target_goal
from scout_piper_scene_repr_py.distance_query import SemanticDistanceQuery
from scout_piper_scene_repr_py.field import export_field, fill_msg
from scout_piper_scene_repr_py.policy import DEFAULT_POLICIES, load_policies
from scout_piper_scene_repr_py.ros_integrator import RosSceneIntegrator
from scout_piper_scene_repr_py.self_filter import DEFAULT_PIPER_FINGER_BOXES

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
            ("policy_yaml_path", ""),          # "" = share/scout_piper_scene_repr/config/semantic_classes.yaml
            ("min_hits", 2),
            ("field_topic", "/scene_repr/distance_field"),
            ("field_rate_hz", 1.0),            # 0 disables the field (≈11 B per voxel per message)
            ("self_filter_boxes", list(DEFAULT_PIPER_FINGER_BOXES)),   # [] disables the self-filter
            ("target_goal_topic", "/scene_repr/target_goal"),
            ("target_goal_offset_m", 0.12),    # pre-grasp distance (stem_grasp target_position_offset_m)
            ("base_frame", "base_link"),       # approach axis: from this frame's origin to the target
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.scene = RosSceneIntegrator(
            self, world_frame=p("world_frame"), depth_topic=p("depth_topic"),
            camera_info_topic=p("camera_info_topic"), classes=list(p("classes")),
            grid_center=list(p("grid_center")), half_extent_m=float(p("grid_half_extent_m")),
            voxel_size_m=float(p("voxel_size_m")), stride=int(p("stride")),
            max_range_m=float(p("max_range_m")),
            self_filter_boxes=[b for b in p("self_filter_boxes") if b])
        self.query = SemanticDistanceQuery(self.scene.map, self._policies(p("policy_yaml_path")),
                                           min_hits=int(p("min_hits")))
        self.pub_vox = self.create_publisher(MarkerArray, "/scene_repr/voxels", 1)
        self.pub_status = self.create_publisher(String, "/scene_repr/map_status", 1)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_field = self.create_publisher(SemanticDistanceField, p("field_topic"), latched)
        self.pub_goal = self.create_publisher(PoseStamped, p("target_goal_topic"), 1)
        self.goal_offset = float(p("target_goal_offset_m"))
        self.base_frame = str(p("base_frame"))
        self.last_export_ms = 0.0
        self._field_version = -1
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)
        if float(p("field_rate_hz")) > 0:
            self.create_timer(1.0 / float(p("field_rate_hz")), self._publish_field)

    def _policies(self, path: str):
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415
                path = os.path.join(get_package_share_directory("scout_piper_scene_repr"),
                                    "config", "semantic_classes.yaml")
            except Exception:  # noqa: BLE001 — not installed: fall back to the built-in policy
                path = ""
        if path and os.path.exists(path):
            return load_policies(path)
        self.get_logger().warn("semantic_classes.yaml not found; using the built-in class policy")
        return DEFAULT_POLICIES

    def _tick(self) -> None:
        if self.scene.integrate_latest():
            self._publish()
            self._publish_target_goal()

    def _publish_target_goal(self) -> None:
        vmap = self.scene.map
        if "target" not in vmap.hits:
            return
        try:
            tr = self.scene.tf.lookup_transform(self.scene.world, self.base_frame, rclpy.time.Time())
            robot_xy = [tr.transform.translation.x, tr.transform.translation.y]
        except Exception:  # noqa: BLE001
            return
        g = target_goal(vmap.occupied_points("target"), vmap.spec.voxel_size, robot_xy, self.goal_offset)
        if g is None:
            return
        ps = PoseStamped()
        ps.header.frame_id = self.scene.world
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in g.pre_grasp)
        q = axis_quaternion(g.axis)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = (
            float(v) for v in q)
        self.pub_goal.publish(ps)

    def _publish_field(self) -> None:
        vmap = self.scene.map
        if vmap.version == self._field_version:
            return                                     # nothing new since the last snapshot
        t0 = time.perf_counter()
        snap = export_field(self.query)
        if snap is None:
            return
        self.pub_field.publish(fill_msg(SemanticDistanceField(), snap, self.scene.world))
        self.last_export_ms = 1e3 * (time.perf_counter() - t0)
        self._field_version = vmap.version

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
            "integrate_ms": round(self.scene.last_integrate_ms, 1),
            "self_filtered_px": self.scene.filtered_pixels,
            "export_ms": round(self.last_export_ms, 1)})))


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
