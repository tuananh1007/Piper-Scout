"""Publish the hand-eye (camera mount) transform from a calibration file.

The RealSense driver already publishes its own internal frames
(camera_link -> camera_{color,depth}_frame -> *_optical_frame) from the
factory calibration. This node attaches that tree to the arm with one static
transform, ``link6 -> camera_link``.

The calibration file may give the mount relative to any frame of the driver's
tree, e.g. MoveIt Calibration run with sensor frame = camera_color_optical_frame
gives ``link6 -> camera_color_optical_frame``. In that case this node waits
for the driver's ``camera_link -> <child>`` transform and composes:

    T_link6_cameralink = T_link6_child * inv(T_cameralink_child)

so the RealSense tree keeps a single parent.

It also latches the calibration and its provenance on /camera_mount/info
(std_msgs/String, JSON) so recorded episodes carry it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from piper_quickstart.transforms import invert, load_camera_mount, make_transform, split


class CameraMountPublisher(Node):
    def __init__(self) -> None:
        super().__init__("camera_mount_publisher")
        self.exit_code = None
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("camera_root_frame", "camera_link")
        self.declare_parameter("driver_tf_timeout_s", 30.0)

        path = Path(self.get_parameter("calibration_file").value)
        self._root = self.get_parameter("camera_root_frame").value
        self._mount = load_camera_mount(path)
        self._path = path
        m = self._mount

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._info_pub = self.create_publisher(String, "/camera_mount/info", latched)
        self._broadcaster = StaticTransformBroadcaster(self)

        if m.calibrated:
            self.get_logger().info(f"Camera mount from {path} ({m.source or 'calibrated'}): "
                                   f"{m.parent_frame} -> {m.child_frame}")
        else:
            self.get_logger().warn(
                f"\n{'!' * 78}\nCamera extrinsics are NOMINAL, not calibrated ({path}).\n"
                "Recorded camera poses will be wrong by up to a few cm / degrees.\n"
                "Calibrate: ros2 launch piper_quickstart quickstart.launch.py calibrate:=true\n"
                f"{'!' * 78}")

        if m.child_frame == self._root:
            self._publish(m.matrix())
        else:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._deadline = self.get_clock().now() + Duration(seconds=self.get_parameter("driver_tf_timeout_s").value)
            self.get_logger().info(f"Waiting for driver TF {self._root} -> {m.child_frame} to compose the mount")
            self._timer = self.create_timer(0.2, self._try_compose)

    def _try_compose(self) -> None:
        m = self._mount
        try:
            tf = self._tf_buffer.lookup_transform(self._root, m.child_frame, Time())
        except Exception:
            if self.get_clock().now() > self._deadline:
                self.get_logger().fatal(f"No TF {self._root} -> {m.child_frame} from the RealSense driver; "
                                        "cannot place the camera. Is the camera running?")
                self.exit_code = 4
            return
        self._timer.cancel()
        t, q = tf.transform.translation, tf.transform.rotation
        root_to_child = make_transform([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
        self._publish(m.matrix() @ invert(root_to_child))

    def _publish(self, parent_to_root) -> None:
        m = self._mount
        trans, quat = split(parent_to_root)
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = m.parent_frame
        msg.child_frame_id = self._root
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, trans)
        (msg.transform.rotation.x, msg.transform.rotation.y,
         msg.transform.rotation.z, msg.transform.rotation.w) = map(float, quat)
        self._broadcaster.sendTransform(msg)
        info = {
            "calibration_file": str(self._path),
            "calibration": m.to_dict(),
            "published": {
                "parent_frame": m.parent_frame,
                "child_frame": self._root,
                "translation": [float(v) for v in trans],
                "rotation_xyzw": [float(v) for v in quat],
            },
        }
        self._info_pub.publish(String(data=json.dumps(info)))
        self.get_logger().info(f"Published {m.parent_frame} -> {self._root}: t={trans.round(4).tolist()} "
                               f"q={quat.round(4).tolist()}")



def main(args=None) -> int:
    rclpy.init(args=args)
    node = CameraMountPublisher()
    try:
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    code = node.exit_code or 0
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
