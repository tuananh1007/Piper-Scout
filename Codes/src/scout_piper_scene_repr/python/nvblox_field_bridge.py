#!/usr/bin/env python3
"""nvblox_field_bridge — per-class nvblox ESDFs → /scene_repr/distance_field (P1.3.1, P3.2.2).

Every 1 / rate_hz s it asks each class mapper of ``nvblox_semantic.launch.py``
for its ESDF inside the plant box (``~/get_esdf_and_gradient``,
nvblox_msgs/srv/EsdfAndGradients; the mappers run with ``esdf_mode: 3d``),
merges them with the class policies (``nvblox_field.merge_class_grids``) and
publishes the same ``SemanticDistanceField`` as the CPU ``scene_query_node``.
The MoveIt semantic collision plugin and the whole-body MPC
(``field_topic``) consume it unchanged, now backed by the GPU maps.

Outputs
  /scene_repr/distance_field  SemanticDistanceField (reliable, transient local)
  /scene_repr/target_goal     geometry_msgs/PoseStamped pre-grasp pose from the target class
  /scene_repr/bridge_status   std_msgs/String JSON: per-class result, ms, known voxels

Needs nvblox_msgs (built with isaac_ros_nvblox). Run either this node or
scene_query_node on /scene_repr/distance_field, not both.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from scout_piper_scene_repr.msg import SemanticDistanceField
from scout_piper_scene_repr_py.attractor import axis_quaternion, target_goal
from scout_piper_scene_repr_py.field import AGE_UNKNOWN, fill_msg
from scout_piper_scene_repr_py.nvblox_field import grid_from_response, merge_class_grids, target_points
from scout_piper_scene_repr_py.policy import DEFAULT_POLICIES, load_policies


class NvbloxFieldBridge(Node):
    def __init__(self) -> None:
        super().__init__("scene_repr_nvblox_field_bridge")
        self.declare_parameters("", [
            ("classes", ["stem", "branch", "leaf", "target", "other"]),
            ("service_pattern", "/scene_repr/{c}/nvblox_{c}/get_esdf_and_gradient"),
            ("world_frame", "odom"),
            ("grid_center", [0.6, 0.0, 0.6]),
            ("grid_half_extent_m", 0.5),
            ("voxel_size_m", 0.01),
            ("rate_hz", 2.0),
            ("request_timeout_s", 1.0),
            ("update_esdf", True),
            ("conservative_resample", True),
            ("policy_yaml_path", ""),
            ("field_topic", "/scene_repr/distance_field"),
            ("target_goal_topic", "/scene_repr/target_goal"),
            ("target_goal_offset_m", 0.12),
            ("base_frame", "base_link"),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        try:
            from nvblox_msgs.srv import EsdfAndGradients  # noqa: PLC0415
        except ImportError as exc:
            raise SystemExit("nvblox_msgs not found: build isaac_ros_nvblox (INSTALL.md 10.10)") from exc
        self.srv_type = EsdfAndGradients
        self.world = str(p("world_frame"))
        self.classes = list(p("classes"))
        self.clients = {c: self.create_client(EsdfAndGradients, str(p("service_pattern")).format(c=c))
                        for c in self.classes}
        half = float(p("grid_half_extent_m"))
        self.vs = float(p("voxel_size_m"))
        n = int(np.ceil(2 * half / self.vs))
        self.shape = (n, n, n)
        self.origin = np.asarray(p("grid_center"), float) - half
        self.timeout = float(p("request_timeout_s"))
        self.update_esdf = bool(p("update_esdf"))
        self.conservative = bool(p("conservative_resample"))
        self.policies = self._policies(str(p("policy_yaml_path")))
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_field = self.create_publisher(SemanticDistanceField, str(p("field_topic")), latched)
        self.pub_goal = self.create_publisher(PoseStamped, str(p("target_goal_topic")), 1)
        self.pub_status = self.create_publisher(String, "/scene_repr/bridge_status", 1)
        self.goal_offset = float(p("target_goal_offset_m"))
        self.base_frame = str(p("base_frame"))
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.pending = {}
        self.t_request = 0.0
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)
        self.get_logger().info(f"bridging {self.classes} → {p('field_topic')} on a {n}³ grid of {self.vs} m")

    def _policies(self, path: str):
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415
                path = os.path.join(get_package_share_directory("scout_piper_scene_repr"),
                                    "config", "semantic_classes.yaml")
            except Exception:  # noqa: BLE001
                path = ""
        return load_policies(path) if path and os.path.exists(path) else DEFAULT_POLICIES

    def _request(self):
        req = self.srv_type.Request()
        req.update_esdf = self.update_esdf
        req.visualize_esdf = False
        req.use_aabb = True
        req.frame_id = self.world
        req.aabb_min_m = Point(x=float(self.origin[0]), y=float(self.origin[1]), z=float(self.origin[2]))
        size = self.vs * np.array(self.shape, float)
        req.aabb_size_m = Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2]))
        return req

    def _tick(self) -> None:
        now = time.monotonic()
        if self.pending:
            done = all(f.done() for f in self.pending.values())
            if not done and now - self.t_request < self.timeout:
                return
            self._merge(now)
            self.pending = {}
        self.t_request = now
        for c, cl in self.clients.items():
            if cl.service_is_ready():
                self.pending[c] = cl.call_async(self._request())
        if not self.pending:
            self.get_logger().warn("no nvblox ESDF service ready (is nvblox_semantic.launch.py running?)",
                                   throttle_duration_sec=10.0)

    def _merge(self, now: float) -> None:
        grids, status = {}, {}
        for c in self.classes:
            f = self.pending.get(c)
            if f is None:
                grids[c], status[c] = None, "no service"
                continue
            if not f.done():
                grids[c], status[c] = None, "timeout"
                continue
            r = f.result()
            if r is None or not r.success:
                grids[c], status[c] = None, "failed"
                continue
            dims = [d.size for d in r.esdf_and_gradients.layout.dim]
            try:
                grids[c] = grid_from_response(r.esdf_and_gradients.data, dims,
                                              [r.origin_m.x, r.origin_m.y, r.origin_m.z], r.voxel_size_m)
                status[c] = "ok"
            except ValueError as exc:
                grids[c], status[c] = None, str(exc)
        t0 = time.perf_counter()
        stamp = self.get_clock().now().nanoseconds * 1e-9
        snap = merge_class_grids({c: g for c, g in grids.items()}, self.policies, self.origin, self.shape,
                                 self.vs, stamp, conservative=self.conservative)
        self.pub_field.publish(fill_msg(SemanticDistanceField(), snap, self.world))
        merge_ms = 1e3 * (time.perf_counter() - t0)
        self._publish_goal(grids.get("target"))
        self.pub_status.publish(String(data=json.dumps({
            "classes": status, "merge_ms": round(merge_ms, 1),
            "known_voxels": int((snap.age_ds != AGE_UNKNOWN).sum()),
            "request_s": round(now - self.t_request, 3)})))

    def _publish_goal(self, target_grid) -> None:
        pts = target_points(target_grid)
        if not len(pts):
            return
        try:
            tr = self.tf.lookup_transform(self.world, self.base_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return
        g = target_goal(pts, target_grid.voxel_size, [tr.transform.translation.x, tr.transform.translation.y],
                        self.goal_offset)
        if g is None:
            return
        ps = PoseStamped()
        ps.header.frame_id = self.world
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in g.pre_grasp)
        q = axis_quaternion(g.axis)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = (
            float(v) for v in q)
        self.pub_goal.publish(ps)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NvbloxFieldBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
