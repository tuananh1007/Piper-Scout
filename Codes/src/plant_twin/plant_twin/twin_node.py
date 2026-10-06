"""ROS 2 node: fits the leaf + stem twin every frame and publishes markers.

Inputs:
  leaf_cloud_topic   PointCloud2 (xyz + packed rgb)  leaf points  — default
                     /stem_grasp/leaf_filtered_cloud from stem_grasp/pointcloud_node
  stem_cloud_topic   PointCloud2                     stem points  — default
                     /stem_grasp/filtered_cloud
  mask_topic         Image mono8                     leaf mask, for the outline
  depth_topic        Image                           depth aligned to the mask
  camera_info_topic  CameraInfo                      intrinsics
  /ft_sensor/raw     WrenchStamped                   |F| > threshold ⇒ touching
  /joint_states      JointState                      piper_joint7 closed ⇒ grasped
  TF                 planning_frame ← cloud frame, camera frame, piper_link7

Outputs:
  /plant_twin/markers   MarkerArray   TRIANGLE_LIST leaf (per-vertex colour)
                                      + LINE_STRIP stem
  /plant_twin/leaf_tip  PointStamped  stem/leaf attachment point
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .fitting import FrameObservation, PlantTwinFitter
from .leaf import LeafModel
from .outline import outline_from_mask
from .stem import StemModel


def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
    """Packed float32 rgb (as pointcloud_node writes it) → (N, 3) in [0, 1]."""
    u = np.asarray(packed, dtype=np.float32).view(np.uint32)
    return np.column_stack([(u >> 16) & 255, (u >> 8) & 255, u & 255]) / 255.0


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class PlantTwinNode(Node):
    def __init__(self) -> None:
        super().__init__("plant_twin")
        self.declare_parameters("", [
            ("planning_frame", "piper_base_link"),
            ("fingertip_frame", "piper_link7"),
            ("leaf_cloud_topic", "/stem_grasp/leaf_filtered_cloud"),
            ("stem_cloud_topic", "/stem_grasp/filtered_cloud"),
            ("mask_topic", "/stem_grasp/target_mask"),
            ("depth_topic", "/camera/aligned_depth_to_color/image_raw"),
            ("camera_info_topic", "/camera/color/camera_info"),
            ("contact_threshold_n", 0.15),
            ("gripper_closed_m", 0.01),
            ("rate_hz", 10.0),
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
        self.bridge = CvBridge()

        self.leaf_pts = np.zeros((0, 3))
        self.leaf_rgb = np.zeros((0, 3))
        self.stem_pts = np.zeros((0, 3))
        self.mask: Optional[Image] = None
        self.depth: Optional[Image] = None
        self.K: Optional[np.ndarray] = None
        self.force = 0.0
        self.gripper_gap = 0.035
        self.fitter: Optional[PlantTwinFitter] = None

        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PointCloud2, p("leaf_cloud_topic"), self._leaf_cb, be)
        self.create_subscription(PointCloud2, p("stem_cloud_topic"), self._stem_cb, be)
        self.create_subscription(Image, p("mask_topic"), self._mask_cb, be)
        self.create_subscription(Image, p("depth_topic"), self._depth_cb, be)
        self.create_subscription(CameraInfo, p("camera_info_topic"), self._info_cb, be)
        self.create_subscription(WrenchStamped, "/ft_sensor/raw", self._wrench_cb, 20)
        self.create_subscription(JointState, "/joint_states", self._js_cb, 20)
        self.pub_markers = self.create_publisher(MarkerArray, "/plant_twin/markers", 1)
        self.pub_tip = self.create_publisher(PointStamped, "/plant_twin/leaf_tip", 1)
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)

    # ---------------------------------------------------------------- inputs
    def _lookup(self, source: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            tr = self.tf_buf.lookup_transform(self.frame, source, rclpy.time.Time())
        except Exception:  # noqa: BLE001 — TF not ready is routine
            return None
        t = tr.transform.translation
        return _quat_to_matrix(tr.transform.rotation), np.array([t.x, t.y, t.z])

    def _cloud(self, msg: PointCloud2, want_rgb: bool):
        names = [f.name for f in msg.fields]
        fields = ("x", "y", "z", "rgb") if want_rgb and "rgb" in names else ("x", "y", "z")
        arr = point_cloud2.read_points_numpy(msg, field_names=fields, skip_nans=True)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, len(fields))
        xyz = arr[:, :3].astype(float)
        tf = self._lookup(msg.header.frame_id)
        if tf is None:
            return None, None
        R, t = tf
        rgb = _unpack_rgb(arr[:, 3]) if len(fields) == 4 else np.zeros((len(xyz), 3))
        return xyz @ R.T + t, rgb

    def _leaf_cb(self, msg):
        xyz, rgb = self._cloud(msg, want_rgb=True)
        if xyz is not None:
            self.leaf_pts, self.leaf_rgb = xyz, rgb

    def _stem_cb(self, msg):
        xyz, _ = self._cloud(msg, want_rgb=False)
        if xyz is not None:
            self.stem_pts = xyz

    def _mask_cb(self, msg): self.mask = msg
    def _depth_cb(self, msg): self.depth = msg

    def _info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.camera_frame = msg.header.frame_id

    def _wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        self.force = float(np.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))

    def _js_cb(self, msg: JointState):
        if "piper_joint7" in msg.name:
            self.gripper_gap = float(msg.position[msg.name.index("piper_joint7")])

    def _fingertip(self):
        tf = self._lookup(self.fingertip)
        return None if tf is None else tf[1]

    # ------------------------------------------------------------- initialise
    def _init_models(self) -> bool:
        """First frame with clouds + mask + depth + intrinsics: outline and
        holes from the mask in the PCA leaf plane, stem as a straight line
        from the lowest stem point to the petiole, texture from the cloud."""
        if len(self.leaf_pts) < 30 or len(self.stem_pts) < 5:
            return False
        if self.mask is None or self.depth is None or self.K is None:
            return False
        cam = self._lookup(self.camera_frame)
        if cam is None:
            return False
        try:
            mask = self.bridge.imgmsg_to_cv2(self.mask, desired_encoding="mono8")
            depth = self.bridge.imgmsg_to_cv2(self.depth, desired_encoding="passthrough")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge failed: {exc}")
            return False
        depth = depth.astype(np.float32)
        if np.nanmax(depth) > 10.0:
            depth = depth / 1000.0
        try:
            outline, holes, plane = outline_from_mask(
                mask, depth, self.K, cam[0], cam[1], self.leaf_pts)
        except ValueError as exc:
            self.get_logger().warn(f"outline init failed: {exc}")
            return False

        leaf = LeafModel(outline, holes=holes)
        leaf_params = leaf.initial_params(plane.rotvec, plane.origin)
        base = self.stem_pts[np.argmin(self.stem_pts[:, 2])]
        leaf.set_petiole(plane.to_plane(base[None])[0])
        leaf.texture_from_cloud(leaf_params, self.leaf_pts, self.leaf_rgb)

        k = int(self.get_parameter("stem_ctrl_points").value)
        ctrl = np.linspace(base, leaf.tip_point(leaf_params), k)
        stem = StemModel(ctrl, radius_m=float(self.get_parameter("stem_radius_m").value))

        self.fitter = PlantTwinFitter(leaf, stem)
        self.fitter.leaf_params = leaf_params
        self.get_logger().info(
            f"plant_twin initialised: {len(leaf.faces)} faces, {len(holes)} holes")
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
        leaf.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        leaf.pose.orientation.w = 1.0
        V, F, C = out["leaf_vertices"], out["leaf_faces"], out["leaf_colors"]
        for tri in F:
            for i in tri:
                leaf.points.append(Point(x=float(V[i, 0]), y=float(V[i, 1]), z=float(V[i, 2])))
                if C is not None:
                    leaf.colors.append(ColorRGBA(r=float(C[i, 0]), g=float(C[i, 1]),
                                                 b=float(C[i, 2]), a=1.0))
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
