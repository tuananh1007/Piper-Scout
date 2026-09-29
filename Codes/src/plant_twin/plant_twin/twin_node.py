"""ROS 2 node: fits the leaf + stem twin every frame and publishes markers.

Inputs (all in the planning frame after TF lookup):
  /stem_grasp/leaf_cloud      PointCloud2   leaf points (class_demux / pointcloud_node)
  /stem_grasp/stem_cloud      PointCloud2   stem points
  /ft_sensor/raw              WrenchStamped contact detection (|F| > threshold)
  /joint_states               JointState    gripper joint7 → closed → "pulling"
  TF: planning_frame -> piper_link7 (fingertip) for the contact point.

Outputs:
  /plant_twin/markers         MarkerArray   TRIANGLE_LIST leaf + LINE_STRIP stem
  /plant_twin/leaf_tip        PointStamped  stem/leaf attachment point
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .fitting import FrameObservation, PlantTwinFitter
from .leaf import LeafModel
from .stem import StemModel


def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.asarray(pts, dtype=float).reshape(-1, 3)


class PlantTwinNode(Node):
    def __init__(self) -> None:
        super().__init__("plant_twin")
        self.declare_parameters("", [
            ("planning_frame", "piper_base_link"),
            ("fingertip_frame", "piper_link7"),
            ("contact_threshold_n", 0.15),
            ("gripper_closed_m", 0.01),
            ("rate_hz", 10.0),
            ("leaf_outline_radius_m", 0.06),
            ("stem_ctrl_points", 6),
            ("stem_radius_m", 0.003),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.frame = p("planning_frame")
        self.fingertip = p("fingertip_frame")
        self.thr = float(p("contact_threshold_n"))
        self.closed = float(p("gripper_closed_m"))

        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        self.leaf_pts = np.zeros((0, 3))
        self.stem_pts = np.zeros((0, 3))
        self.force = 0.0
        self.gripper_gap = 0.035
        self.fitter = None

        self.create_subscription(PointCloud2, "/stem_grasp/leaf_cloud", self._leaf_cb, 5)
        self.create_subscription(PointCloud2, "/stem_grasp/stem_cloud", self._stem_cb, 5)
        self.create_subscription(WrenchStamped, "/ft_sensor/raw", self._wrench_cb, 20)
        self.create_subscription(JointState, "/joint_states", self._js_cb, 20)
        self.pub_markers = self.create_publisher(MarkerArray, "/plant_twin/markers", 1)
        self.pub_tip = self.create_publisher(PointStamped, "/plant_twin/leaf_tip", 1)
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)

    # ---------------------------------------------------------------- inputs
    def _leaf_cb(self, msg): self.leaf_pts = _cloud_to_xyz(msg)
    def _stem_cb(self, msg): self.stem_pts = _cloud_to_xyz(msg)

    def _wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        self.force = float(np.hypot(np.hypot(f.x, f.y), f.z))

    def _js_cb(self, msg: JointState):
        if "piper_joint7" in msg.name:
            self.gripper_gap = float(msg.position[msg.name.index("piper_joint7")])

    def _fingertip(self):
        try:
            tr = self.tf_buf.lookup_transform(self.frame, self.fingertip, rclpy.time.Time())
        except Exception:  # noqa: BLE001 — TF not ready is routine
            return None
        t = tr.transform.translation
        return np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------- initialise
    def _init_models(self) -> bool:
        """First frame with both clouds: PCA plane for the leaf, vertical
        polyline from the lowest stem point up to the leaf for the stem."""
        if len(self.leaf_pts) < 30 or len(self.stem_pts) < 5:
            return False
        c = self.leaf_pts.mean(0)
        _, _, vt = np.linalg.svd(self.leaf_pts - c)
        R = vt.T
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        r = float(self.get_parameter("leaf_outline_radius_m").value)
        th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        outline = np.column_stack([r * np.cos(th), r * np.sin(th)])
        leaf = LeafModel(outline)
        # rotation matrix -> rotvec
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        rv = axis / max(np.linalg.norm(axis), 1e-9) * angle
        leaf_params = leaf.initial_params(rv, c)

        k = int(self.get_parameter("stem_ctrl_points").value)
        base = self.stem_pts[np.argmin(self.stem_pts[:, 2])]
        tip = leaf.tip_point(leaf_params)
        ctrl = np.linspace(base, tip, k)
        stem = StemModel(ctrl, radius_m=float(self.get_parameter("stem_radius_m").value))

        self.fitter = PlantTwinFitter(leaf, stem)
        self.fitter.leaf_params = leaf_params
        self.get_logger().info("plant_twin initialised from first frame")
        return True

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        if self.fitter is None and not self._init_models():
            return
        touching = self.force > self.thr
        contact = self._fingertip() if touching else None
        pulling = touching and self.gripper_gap < self.closed
        res = self.fitter.step(FrameObservation(self.leaf_pts, self.stem_pts, contact, pulling))
        self._publish(res)

    def _publish(self, res) -> None:
        out = self.fitter.export()
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        leaf = Marker()
        leaf.header.frame_id, leaf.header.stamp = self.frame, now
        leaf.ns, leaf.id, leaf.type, leaf.action = "leaf", 0, Marker.TRIANGLE_LIST, Marker.ADD
        leaf.scale.x = leaf.scale.y = leaf.scale.z = 1.0
        leaf.color = ColorRGBA(r=0.3, g=0.7, b=0.3, a=0.9)
        leaf.pose.orientation.w = 1.0
        V, F = out["leaf_vertices"], out["leaf_faces"]
        for tri in F:
            for i in tri:
                leaf.points.append(Point(x=float(V[i, 0]), y=float(V[i, 1]), z=float(V[i, 2])))
        ma.markers.append(leaf)

        stem = Marker()
        stem.header.frame_id, stem.header.stamp = self.frame, now
        stem.ns, stem.id, stem.type, stem.action = "stem", 1, Marker.LINE_STRIP, Marker.ADD
        stem.scale.x = 2.0 * float(out["stem_radius"])
        stem.color = ColorRGBA(r=0.4, g=0.5, b=0.2, a=1.0)
        stem.pose.orientation.w = 1.0
        for q in out["stem_curve"]:
            stem.points.append(Point(x=float(q[0]), y=float(q[1]), z=float(q[2])))
        ma.markers.append(stem)
        self.pub_markers.publish(ma)

        tip = self.fitter.leaf.tip_point(res.leaf_params)
        ps = PointStamped()
        ps.header.frame_id, ps.header.stamp = self.frame, now
        ps.point.x, ps.point.y, ps.point.z = map(float, tip)
        self.pub_tip.publish(ps)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlantTwinNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
