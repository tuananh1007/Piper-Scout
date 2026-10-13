"""ROS 2 node: depth with the D405's thin-structure holes filled by VGGT (P4.1).

Inputs   rgb_topic, depth_topic (aligned to colour), camera_info_topic, TF world ← camera
Outputs  fused_topic   sensor_msgs/Image 32FC1 metres, the depth frame's header: the sensor
                       depth, holes filled from the last reconstruction while the camera is
                       still at its newest keyframe (``vggt.DepthFiller``), otherwise unchanged
         ~/status      std_msgs/String JSON: keyframes, last reconstruction (seconds, scale),
                       pixels filled in the last frame, agreement with the sensor
         ~/reconstruct std_srvs/Trigger: run now (otherwise after each new keyframe)

Keyframes enter the batch when the camera moved 3 cm or turned 8 degrees
(``SparseViewBatcher``); the reconstruction runs in a worker thread, one at a
time. ``model: vggt`` loads facebook/VGGT-1B on ``device`` (GPU environment
only); ``model: stand_in`` inpaints the sensor depth instead (tests and
hardware-free checks; not a reconstruction). Feed the scene stack from
``fused_topic`` (scene_query_node / class_demux ``depth_topic``) once
MODULE_TASKS.md A14 shows that VGGT agrees with the D405 on this plant.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from .vggt import DepthFiller, SparseViewBatcher, View, VGGTReconstructor, inpaint_stand_in


def _tf_to_T(tr) -> np.ndarray:
    q, t = tr.rotation, tr.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def image_to_array(m: Image) -> np.ndarray:
    dt = {"16UC1": np.uint16, "32FC1": np.float32, "rgb8": np.uint8, "bgr8": np.uint8, "mono8": np.uint8}[m.encoding]
    ch = 3 if m.encoding in ("rgb8", "bgr8") else 1
    a = np.frombuffer(bytes(m.data), dt)
    row = m.step // np.dtype(dt).itemsize
    a = a.reshape(m.height, row)[:, :m.width * ch]
    return a.reshape(m.height, m.width, ch) if ch == 3 else a.reshape(m.height, m.width)


class FusedDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("fused_depth")
        self.declare_parameters("", [
            ("rgb_topic", "/camera/color/image_raw"),
            ("depth_topic", "/camera/aligned_depth_to_color/image_raw"),
            ("camera_info_topic", "/camera/color/camera_info"),
            ("world_frame", "odom"),
            ("fused_topic", "/active_perception/fused_depth"),
            ("model", "vggt"), ("weights", "facebook/VGGT-1B"), ("device", "cuda"),
            ("max_views", 6), ("min_views", 2),
            ("min_baseline_m", 0.03), ("min_angle_deg", 8.0),
            ("still_m", 0.01), ("still_deg", 2.0), ("min_conf", 3.0),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.world = p("world_frame")
        self.batcher = SparseViewBatcher(int(p("max_views")), float(p("min_baseline_m")), float(p("min_angle_deg")))
        model_fn = inpaint_stand_in(self.batcher) if p("model") == "stand_in" else None
        self.filler = DepthFiller(VGGTReconstructor(model_fn, p("weights"), p("device")), self.batcher,
                                  int(p("min_views")), float(p("still_m")), float(p("still_deg")), float(p("min_conf")))
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.rgb: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.busy = False
        self.last_info = {}
        self.error = ""
        self.lock = threading.Lock()
        be = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, p("rgb_topic"), self._rgb, be)
        self.create_subscription(Image, p("depth_topic"), self._depth, be)
        self.create_subscription(CameraInfo, p("camera_info_topic"),
                                 lambda m: setattr(self, "K", np.array(m.k, float).reshape(3, 3)), be)
        self.pub = self.create_publisher(Image, p("fused_topic"), 2)
        self.pub_status = self.create_publisher(String, "~/status", 5)
        self.create_service(Trigger, "~/reconstruct", self._srv)
        self.create_timer(1.0, self._status)
        self.get_logger().info(f"fused depth up (model {p('model')})")

    def _rgb(self, m: Image) -> None:
        a = image_to_array(m)
        self.rgb = a[..., ::-1].copy() if m.encoding == "bgr8" else a

    def _depth(self, m: Image) -> None:
        d = image_to_array(m).astype(np.float32)
        if m.encoding == "16UC1":
            d = d / 1000.0
        d[d <= 0] = np.nan
        try:
            T = _tf_to_T(self.tf_buf.lookup_transform(self.world, m.header.frame_id,
                                                      rclpy.time.Time.from_msg(m.header.stamp)).transform)
        except Exception:  # noqa: BLE001 — TF gaps are routine: pass through
            T = None
        out, info = d, {"filled": 0, "reason": "no TF"}
        if T is not None:
            if self.rgb is not None and self.K is not None and self.rgb.shape[:2] == d.shape:
                stamp = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
                with self.lock:
                    self.filler.add_frame(View(self.rgb.copy(), d.copy(), T, self.K, stamp))
                if self.filler.ready() and not self.busy:
                    self._start()
            with self.lock:
                out, info = self.filler.fill(d, T)
        self.last_info = info
        msg = Image()
        msg.header = m.header
        msg.height, msg.width, msg.encoding, msg.step = out.shape[0], out.shape[1], "32FC1", out.shape[1] * 4
        msg.data = np.ascontiguousarray(out, np.float32).tobytes()
        self.pub.publish(msg)

    def _start(self) -> None:
        self.busy = True
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            r = self.filler.reconstruct()
            self.error = "" if r is not None else "no metric scale (no sensor depth overlap, no baseline)"
        except Exception as exc:  # noqa: BLE001 — model missing / CUDA errors: report, keep passing through
            self.error = f"{type(exc).__name__}: {exc}"[:200]
            self.get_logger().error(f"reconstruction failed: {self.error}", throttle_duration_sec=30.0)
        finally:
            self.busy = False

    def _srv(self, req, res):
        if self.busy:
            res.success, res.message = False, "reconstruction running"
        elif not self.batcher.ready(1):
            res.success, res.message = False, "no keyframes"
        else:
            self.filler.dirty = True
            self._start()
            res.success, res.message = True, "started"
        return res

    def _status(self) -> None:
        r = self.filler.result
        body = {"keyframes": len(self.batcher.views), "busy": self.busy, "error": self.error, **self.last_info,
                "reconstruction": None if r is None else {"views": r["views"], "seconds": round(r["seconds"], 3),
                                                          "scale": round(r["scale"], 4)}}
        self.pub_status.publish(String(data=json.dumps(body, default=float)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusedDepthNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
