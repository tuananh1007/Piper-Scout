"""ROS 2 node: the operator GUI's robot side (P5.3).

Serves ``gui_server.py`` with a backend bridged to ROS:

  page -> robot   /operator/instruction, /operator/confirm, /operator/cancel (the
                  instruction node does the language work); STOP runs the software
                  stop here as well (``safety_stop.py``), so it works without the
                  instruction node; manual drive: /cmd_vel (base) and
                  /servo_node/delta_twist_cmds (TCP jog in the camera frame), each
                  command held ``hold_s`` then zero, only while drive is enabled and
                  the pipeline is not moving on its own
  robot -> page   /operator/say, /operator/proposal (events), /operator/state,
                  /stem_grasp/pipeline_state, /whole_body_mpc/status,
                  /active_perception/status (uncertainty), /plant_twin/state,
                  /ft_sensor/raw (force), /stem_grasp/detections (overlay), camera image

Voice (optional): ``whisper_cmd`` transcribes the page's 16 kHz WAV, e.g.
"whisper-cli -m /models/ggml-base.en.bin -nt -f {wav}" (the transcript is read
from stdout); ``tts_cmd`` speaks, e.g. "piper --model /models/en_US-lessac-medium.onnx
--output_file {wav}" (the text on stdin). Without them the page types and uses
the browser's own speech synthesis.

    ros2 run scout_piper_language operator_gui_node --ros-args -p port:=8080 [-p host:=0.0.0.0]
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped, WrenchStamped
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from std_srvs.srv import SetBool, Trigger

from .gui_server import EventHub, GuiServer
from .safety_stop import SafetyStop

MOVING_STATES = ("REACHING", "SERVOING", "APPROACHING", "GRASPING", "RELEASING")


class RosBackend:
    def __init__(self, node: "OperatorGuiNode", hub: EventHub):
        self.n, self.hub = node, hub

    def snapshot(self) -> Dict:
        return self.n.snapshot()

    def instruction(self, text: str) -> Dict:
        self.n.pub_instr.publish(String(data=text))
        return {"ok": True}

    def confirm(self) -> Dict:
        self.n.pub_confirm.publish(Empty())
        return {"ok": True}

    def cancel(self) -> Dict:
        self.n.pub_cancel.publish(Empty())
        return {"ok": True}

    def abort(self) -> Dict:
        return self.n.abort()

    def drive(self, cmd: Dict) -> Dict:
        return self.n.drive(cmd)

    def drive_enable(self, enable: bool) -> Dict:
        return self.n.drive_enable(enable)

    def jpeg(self) -> Optional[bytes]:
        return self.n.jpeg

    def transcribe(self, wav: bytes) -> Optional[str]:
        return self.n.run_whisper(wav)

    def tts(self, text: str) -> Optional[bytes]:
        return self.n.run_tts(text)


class OperatorGuiNode(Node):
    def __init__(self) -> None:
        super().__init__("operator_gui")
        self.declare_parameters("", [
            ("host", "127.0.0.1"), ("port", 8080), ("fps", 10.0),
            ("camera_topic", "/camera/color/image_raw"),
            ("jpeg_quality", 70), ("max_width", 640),
            ("hold_s", 0.3),
            ("base_v_max", 0.2), ("base_w_max", 0.4), ("arm_v_max", 0.05),
            ("camera_frame", "camera_color_optical_frame"),
            ("whisper_cmd", ""), ("tts_cmd", ""),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.p = p
        self.hub = EventHub()
        self.stop = SafetyStop(self)
        self.lock = threading.Lock()
        self.data: Dict = {"pipeline": None, "mpc": {}, "uncertainty": {}, "twin": {}, "force_n": None,
                           "session": {}, "detections": [], "image_hw": None}
        self.jpeg: Optional[bytes] = None
        self.last_img = 0.0
        self.drive_enabled = False
        self.cmd: Dict = {}
        self.cmd_t = -np.inf
        self.t_seen = time.time()

        self.pub_instr = self.create_publisher(String, "/operator/instruction", 10)
        self.pub_confirm = self.create_publisher(Empty, "/operator/confirm", 10)
        self.pub_cancel = self.create_publisher(Empty, "/operator/cancel", 10)
        self.pub_abort = self.create_publisher(Empty, "/operator/abort", 10)
        self.pub_base = self.create_publisher(Twist, "/cmd_vel", 5)
        self.pub_twist = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 5)
        self.cl_servo = self.create_client(Trigger, "/servo_node/start_servo")
        self.cl_bridge = self.create_client(SetBool, "/piper_servo_bridge/enable")
        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        js = lambda key: (lambda m: self._set(key, json.loads(m.data)))  # noqa: E731
        self.create_subscription(String, "/operator/say", lambda m: self.hub.publish("say", {"text": m.data}), 10)
        self.create_subscription(String, "/operator/proposal",
                                 lambda m: self.hub.publish("proposal", json.loads(m.data)), 10)
        self.create_subscription(String, "/operator/state", js("session"), 5)
        self.create_subscription(String, "/stem_grasp/pipeline_state", lambda m: self._set("pipeline", m.data), 10)
        self.create_subscription(String, "/whole_body_mpc/status", js("mpc"), 5)
        self.create_subscription(String, "/active_perception/status", js("uncertainty"), 5)
        self.create_subscription(String, "/plant_twin/state", js("twin"), 5)
        self.create_subscription(String, "/stem_grasp/detections", self._dets, be)
        self.create_subscription(WrenchStamped, "/ft_sensor/raw", self._force, be)
        self.create_subscription(Image, p("camera_topic"), self._image, be)
        self.create_timer(0.1, self._drive_tick)
        self.create_timer(0.5, lambda: self.hub.publish("state", self.snapshot()))

        self.server = GuiServer((p("host"), int(p("port"))), RosBackend(self, self.hub), self.hub, float(p("fps")))
        self.server.start()
        self.get_logger().info(f"operator GUI on http://{p('host')}:{self.server.server_address[1]}/")

    # ------------------------------------------------------------- inputs
    def _set(self, key, val) -> None:
        with self.lock:
            self.data[key] = val
        self.t_seen = time.time()

    def _dets(self, m: String) -> None:
        d = json.loads(m.data)
        with self.lock:
            self.data["detections"] = [{k: x.get(k) for k in ("index", "class", "colour", "bbox", "conf")}
                                       for x in d.get("detections", [])]
            self.data["image_hw"] = d.get("image_hw")

    def _force(self, m: WrenchStamped) -> None:
        f = m.wrench.force
        self._set("force_n", float(np.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)))

    def _image(self, m: Image) -> None:
        now = time.time()
        if now - self.last_img < 1.0 / max(float(self.p("fps")), 1.0):
            return
        self.last_img = now
        try:
            import cv2  # noqa: PLC0415
            ch = 3 if m.encoding in ("rgb8", "bgr8") else 1
            a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * ch]
            a = a.reshape(m.height, m.width, ch) if ch == 3 else a.reshape(m.height, m.width)
            if m.encoding == "rgb8":
                a = a[..., ::-1]
            mw = int(self.p("max_width"))
            if a.shape[1] > mw:                      # overlay boxes are scaled by image_hw on the page
                a = cv2.resize(a, (mw, int(a.shape[0] * mw / a.shape[1])), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, int(self.p("jpeg_quality"))])
            if ok:
                self.jpeg = buf.tobytes()
                with self.lock:
                    if self.data.get("image_hw") is None:
                        self.data["image_hw"] = [m.height, m.width]
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"camera image not encoded: {exc}", throttle_duration_sec=10.0)

    def snapshot(self) -> Dict:
        with self.lock:
            s = json.loads(json.dumps(self.data))
        s["drive_enabled"] = self.drive_enabled
        s["stops"] = self.stop.count
        s["stop_bridge_confirmed"] = self.stop.bridge_confirmed
        s["connected"] = time.time() - self.t_seen < 3.0
        return s

    # ------------------------------------------------------------- actions
    def abort(self) -> Dict:
        self.drive_enabled = False
        self.cmd, self.cmd_t = {}, -np.inf
        self.pub_abort.publish(Empty())
        res = self.stop.trigger("GUI")
        self.hub.publish("stop", res)
        self.hub.publish("say", {"text": "Stopped."})
        return res

    def drive_enable(self, enable: bool) -> Dict:
        if not enable:
            self.drive_enabled = False
            self.cmd = {}
            return {"ok": True, "enabled": False}
        pipe = self.data.get("pipeline")
        if pipe in MOVING_STATES:
            return {"ok": False, "enabled": False, "reason": f"the pipeline is moving ({pipe})"}
        if self.cl_servo.service_is_ready():
            self.cl_servo.call_async(Trigger.Request())
        if self.cl_bridge.service_is_ready():
            self.cl_bridge.call_async(SetBool.Request(data=True))
        self.drive_enabled = True
        return {"ok": True, "enabled": True}

    def drive(self, cmd: Dict) -> Dict:
        if not self.drive_enabled:
            return {"ok": False, "reason": "drive not enabled"}
        if self.data.get("pipeline") in MOVING_STATES or (self.data.get("mpc") or {}).get("mode") == "whole_body":
            return {"ok": False, "reason": "the robot is moving on its own"}
        self.cmd, self.cmd_t = dict(cmd), time.time()
        return {"ok": True}

    def _drive_tick(self) -> None:
        if not self.drive_enabled:
            return
        fresh = time.time() - self.cmd_t < float(self.p("hold_s"))
        c = self.cmd if fresh else {}
        vmax, wmax, amax = float(self.p("base_v_max")), float(self.p("base_w_max")), float(self.p("arm_v_max"))
        tw = Twist()
        tw.linear.x = float(np.clip(c.get("v", 0.0), -vmax, vmax))
        tw.angular.z = float(np.clip(c.get("w", 0.0), -wmax, wmax))
        self.pub_base.publish(tw)
        arm = np.clip(np.asarray(c.get("arm", [0.0] * 6), float)[:6], -amax, amax)
        if fresh or self._arm_was_moving:
            ts = TwistStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id = str(self.p("camera_frame"))
            ts.twist.linear.x, ts.twist.linear.y, ts.twist.linear.z = map(float, arm[:3])
            ts.twist.angular.x, ts.twist.angular.y, ts.twist.angular.z = map(float, arm[3:])
            self.pub_twist.publish(ts)
        self._arm_was_moving = bool(fresh and np.any(arm != 0))
        if not fresh:
            self.cmd = {}

    _arm_was_moving = False

    # ------------------------------------------------------------- voice
    def run_whisper(self, wav: bytes) -> Optional[str]:
        cmd = str(self.p("whisper_cmd")).strip()
        if not cmd:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            path = f.name
        try:
            out = subprocess.run(shlex.split(cmd.replace("{wav}", path)), capture_output=True, text=True, timeout=30)
            return " ".join(ln.strip() for ln in out.stdout.splitlines() if ln.strip() and not ln.startswith("["))
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.get_logger().warn(f"whisper failed: {exc}")
            return ""
        finally:
            os.unlink(path)

    def run_tts(self, text: str) -> Optional[bytes]:
        cmd = str(self.p("tts_cmd")).strip()
        if not cmd:
            return None
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(shlex.split(cmd.replace("{wav}", path)), input=text, text=True, capture_output=True,
                           timeout=30, check=True)
            with open(path, "rb") as f:
                return f.read()
        except (OSError, subprocess.SubprocessError) as exc:
            self.get_logger().warn(f"tts failed: {exc}")
            return None
        finally:
            os.unlink(path)

    def destroy_node(self):
        self.server.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OperatorGuiNode()
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
