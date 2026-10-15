"""Runtime watchdog: kills the whole launch if the arm or camera goes quiet.

A stream counts as alive only while its header.stamp keeps advancing, not
merely while messages arrive: the Piper driver republishes cached joint values
at 200 Hz even after CAN feedback stops.

Preflight catches hardware that is missing at startup; this catches hardware
that never starts streaming (driver failed to enable the arm, RealSense
failed to open) or that drops out mid-session (cable pulled, e-stop, USB
reset). It exits non-zero with a FATAL message, and quickstart.launch.py
shuts everything down when it exits, so you never collect data from a
half-dead system.

Also logs Piper error flags from /arm_status as they change.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState

from piper_quickstart.freshness import StampWatch, stamp_to_ns

try:
    from piper_msgs.msg import PiperStatusMsg
except ImportError:  # pragma: no cover
    PiperStatusMsg = None


@dataclass
class _Stream:
    label: str
    topic: str
    startup_grace_s: float
    stale_s: float
    hint: str
    watch: StampWatch = field(default_factory=StampWatch)
    announced: bool = False


class HealthMonitor(Node):
    def __init__(self) -> None:
        super().__init__("piper_health_monitor")
        self.declare_parameter("monitor_arm", True)
        self.declare_parameter("monitor_camera", True)
        self.declare_parameter("arm_topic", "/joint_states_single")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("arm_startup_grace_s", 12.0)
        self.declare_parameter("camera_startup_grace_s", 25.0)
        self.declare_parameter("arm_stale_s", 0.5)
        self.declare_parameter("camera_stale_s", 2.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        self._t0 = time.monotonic()
        self._streams: Dict[str, _Stream] = {}
        self.exit_code: Optional[int] = None

        if p("monitor_arm"):
            self._watch(
                JointState, _Stream(
                    "Piper joint states", p("arm_topic"), p("arm_startup_grace_s"), p("arm_stale_s"),
                    "the driver could not enable the arm, or CAN dropped: check the e-stop, power and CAN cable; "
                    "see the piper_ctrl_single_node log above"),
                10)
            if PiperStatusMsg is not None:
                self._last_status: Optional[tuple] = None
                self.create_subscription(PiperStatusMsg, "/arm_status", self._on_status, 10)
        if p("monitor_camera"):
            for label, topic in (("RealSense color", p("color_topic")), ("RealSense aligned depth", p("depth_topic"))):
                self._watch(
                    Image, _Stream(
                        label, topic, p("camera_startup_grace_s"), p("camera_stale_s"),
                        "the RealSense driver is not streaming: check the camera_node log above, the USB 3 "
                        "cable, and that no other process holds the camera"),
                    qos_profile_sensor_data)

        self.create_timer(0.25, self._check)

    def _watch(self, msg_type, s: _Stream, qos) -> None:
        self._streams[s.topic] = s

        def cb(msg, s=s):
            s.watch.update(stamp_to_ns(msg.header.stamp), time.monotonic())

        self.create_subscription(msg_type, s.topic, cb, qos)

    def _on_status(self, msg) -> None:
        comm = [getattr(msg, f"communication_status_joint_{i}") for i in range(1, 7)]
        lim = [getattr(msg, f"joint_{i}_angle_limit") for i in range(1, 7)]
        key = (msg.err_code, tuple(comm), tuple(lim))
        if key == self._last_status:
            return
        self._last_status = key
        bad_comm = [i + 1 for i, v in enumerate(comm) if v]
        bad_lim = [i + 1 for i, v in enumerate(lim) if v]
        if msg.err_code or bad_comm or bad_lim:
            self.get_logger().warn(
                f"Piper status: err_code={msg.err_code} joint comm errors={bad_comm} joint limit hits={bad_lim}")
        else:
            self.get_logger().info("Piper status: no errors")

    def _fatal(self, msg: str, code: int) -> None:
        self.get_logger().fatal("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)
        self.exit_code = code

    def _check(self) -> None:
        if self.exit_code is not None:
            return
        now = time.monotonic()
        for s in self._streams.values():
            w = s.watch
            if w.last_arrival_mono is None:
                if now - self._t0 > s.startup_grace_s:
                    self._fatal(f"{s.label}: nothing on {s.topic} after {s.startup_grace_s:.0f}s.\n"
                                f"Likely cause: {s.hint}.", 2)
                    return
                continue
            if not s.announced:
                s.announced = True
                self.get_logger().info(f"{s.label}: streaming on {s.topic}")
            arrival_age, data_age = w.arrival_age(now), w.data_age(now)
            if arrival_age > s.stale_s:
                self._fatal(f"{s.label}: {s.topic} stopped ({arrival_age:.1f}s since the last message).\n"
                            f"Likely cause: {s.hint}.", 3)
                return
            if data_age > s.stale_s:
                # Still publishing, but header.stamp hasn't advanced: the publisher is
                # repeating cached data (the Piper driver does this when CAN feedback stops).
                self._fatal(f"{s.label}: {s.topic} is still publishing but its data is frozen "
                            f"(header.stamp unchanged for {data_age:.1f}s).\n"
                            f"Likely cause: {s.hint}.", 4)
                return


def main(args=None) -> int:
    rclpy.init(args=args)
    node = HealthMonitor()
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
