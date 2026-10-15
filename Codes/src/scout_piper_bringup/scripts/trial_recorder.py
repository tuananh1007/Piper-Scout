#!/usr/bin/env python3
"""Record a robot session as JSON lines for trial_report.py.

    ros2 run scout_piper_bringup trial_recorder.py --ros-args -p out:=$LOG/R11_trials.jsonl
    ros2 topic pub --once /trial/note std_msgs/msg/String "data: 'slipped off'"   # annotate
    ros2 run scout_piper_bringup trial_report.py $LOG/R11_trials.jsonl

One line per event, ``{"t": <ROS time s>, "kind": ..., "data": ...}``:

  state        /stem_grasp/pipeline_state        every change
  mpc          /whole_body_mpc/status            mode, TCP / orientation error, solve_ms,
                                                 safety, clearance; mode changes and 5 Hz
  servo        /stem_grasp/servo_status          pixel error, distance, axis miss; 5 Hz
  safety       /safety/state                     on ok / tripped changes and 1 Hz
  ap           /active_perception/status         U terms, needs_view, unmet, joint views; 2 Hz
  force        force_topic (/ft_sensor/raw)      |F| N, the peak of each 0.1 s
  gripper      /joint_states (finger joints)     opening m, on a 0.5 mm change
  odom         /odom                             [x, y, yaw, v, w], 5 Hz
  instruction, proposal, say                     /operator/instruction, /operator/proposal, /operator/say
  note         /trial/note                       free text from the operator

A small rosbag stand-in: what trial_report.py needs, light enough for the
Orin through a whole session (about 30 lines a second while moving). Lines
are flushed as written, so a crash loses at most the last line.
"""

import json
import math
import signal
import time

THROTTLE_S = {"mpc": 0.2, "servo": 0.2, "ap": 0.5, "odom": 0.2, "safety": 1.0}
MPC_KEYS = ("mode", "tcp_error_m", "approach_error_deg", "solve_ms", "safety", "min_clearance_m", "scale")
SERVO_KEYS = ("controller", "error_px", "depth_m", "distance_m", "axis_miss_m", "mode", "safety",
              "min_clearance_m", "solve_ms")
AP_KEYS = ("u_total", "u_joint", "u_corridor", "u_jepa", "u_geometry", "u_twin", "needs_view", "unmet",
           "corridor_unknown", "axis_sigma_mm")


def pick(d, keys):
    return {k: d[k] for k in keys if k in d} if isinstance(d, dict) else d


class Throttle:
    """Pass an event when ``key`` changed or ``period`` s went by since the last one."""

    def __init__(self, period: float):
        self.period, self.t, self.key = period, -math.inf, object()

    def __call__(self, now: float, key=None) -> bool:
        if key != self.key or now - self.t >= self.period:
            self.t, self.key = now, key
            return True
        return False


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    import rclpy  # noqa: PLC0415
    from geometry_msgs.msg import WrenchStamped  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415
    from rclpy.executors import ExternalShutdownException  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: PLC0415
    from rclpy.signals import SignalHandlerOptions  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415
    from std_msgs.msg import String  # noqa: PLC0415

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("trial_recorder")
            p = self.declare_parameter
            out = p("out", "").value or time.strftime("trials_%Y%m%d_%H%M%S.jsonl")
            self.fingers = [s.strip() for s in p("gripper_finger_joints", "piper_joint7,piper_joint8").value.split(",")]
            self.f = open(out, "a", buffering=1)
            self.thr = {k: Throttle(v) for k, v in THROTTLE_S.items()}
            self.lines = 0
            self.force_peak, self.force_t = None, -math.inf
            self.grip = None
            be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

            def js_topic(topic, kind, keys, key_fn):
                def cb(m):
                    try:
                        d = json.loads(m.data)
                    except ValueError:
                        return
                    if self.thr[kind](self._now(), key_fn(d)):
                        self._write(kind, pick(d, keys))
                self.create_subscription(String, topic, cb, 10)

            self.create_subscription(String, "/stem_grasp/pipeline_state", self._state, 10)
            js_topic("/whole_body_mpc/status", "mpc", MPC_KEYS, lambda d: d.get("mode"))
            js_topic("/stem_grasp/servo_status", "servo", SERVO_KEYS, lambda d: None)
            js_topic("/safety/state", "safety", ("ok", "tripped", "moving", "trips", "cmd_timeouts"),
                     lambda d: (d.get("ok"), d.get("tripped")))
            js_topic("/active_perception/status", "ap", AP_KEYS, lambda d: (d.get("needs_view"),
                                                                            str(d.get("unmet"))))
            for topic, kind in (("/operator/instruction", "instruction"), ("/operator/say", "say"),
                                ("/trial/note", "note")):
                self.create_subscription(String, topic, lambda m, k=kind: self._write(k, m.data), 10)
            self.create_subscription(String, "/operator/proposal", self._proposal, 10)
            self.create_subscription(WrenchStamped, p("force_topic", "/ft_sensor/raw").value, self._force, be)
            self.create_subscription(JointState, "/joint_states", self._joints, be)
            self.create_subscription(Odometry, "/odom", self._odom, be)
            self.state = None
            self.last_state_pub = -math.inf
            self.get_logger().info(f"recording to {out}")

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _write(self, kind, data) -> None:
            self.f.write(json.dumps({"t": round(self._now(), 3), "kind": kind, "data": data}) + "\n")
            self.lines += 1

        def _state(self, m) -> None:
            if m.data != self.state:                 # the pipeline republishes its state at 2 Hz
                self.state = m.data
                self._write("state", m.data)

        def _proposal(self, m) -> None:
            try:
                d = json.loads(m.data)
            except ValueError:
                return
            self._write("proposal", {k: d.get(k) for k in ("type", "action", "message", "text") if k in d})

        def _force(self, m) -> None:
            f = m.wrench.force
            mag = math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
            now = self._now()
            self.force_peak = mag if self.force_peak is None else max(self.force_peak, mag)
            if now - self.force_t >= 0.1:
                self._write("force", round(self.force_peak, 3))
                self.force_peak, self.force_t = None, now

        def _joints(self, m) -> None:
            pos = dict(zip(m.name, m.position))
            if len(self.fingers) == 2 and all(n in pos for n in self.fingers):
                w = float(pos[self.fingers[0]] - pos[self.fingers[1]])
                if self.grip is None or abs(w - self.grip) > 5e-4:
                    self.grip = w
                    self._write("gripper", round(w, 5))

        def _odom(self, m) -> None:
            if not self.thr["odom"](self._now()):
                return
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            pz, tw = m.pose.pose.position, m.twist.twist
            self._write("odom", [round(pz.x, 4), round(pz.y, 4), round(yaw, 4),
                                 round(tw.linear.x, 3), round(tw.angular.z, 3)])

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = Recorder()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.f.close()
        node.get_logger().info(f"{node.lines} lines written")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
