#!/usr/bin/env python3
"""Safety supervisor: base command gate, operator heartbeat and feedback watchdogs.

A software layer above the planners and below the drivers. It does not cut
power and does not replace the physical stops (INSTALL.md Step 9); it stops
what software can stop, in cases no single planner sees:

* **Base command gate.** full_system.launch.py (``bringup_safety:=true``, the
  default) remaps the Scout driver's ``/cmd_vel`` to ``/cmd_vel_safe``; this
  node relays ``/cmd_vel`` there, clamped to ``max_cmd_v_mps`` /
  ``max_cmd_w_rps``. scout_ros2 and ugv_sdk have no command timeout (the last
  command stays in force), so a sender that dies mid-motion leaves the base
  driving; the gate sends one zero when no command arrived for
  ``cmd_timeout_s`` after a moving one. Senders must stream (the MPC at 10 Hz,
  the operator page at 10 Hz while a button is held).
* **Operator heartbeat** (``require_heartbeat``). The operator page posts a
  heartbeat four times a second; operator_gui_node republishes it on
  ``/operator/heartbeat`` while the page's are fresh. With no heartbeat for
  ``heartbeat_timeout_s`` while anything moves (base commanded or moving, an
  arm joint moving), or a base command arriving without one: trip. A closed
  tab, a frozen browser, a dropped Wi-Fi link or a dead GUI node all stop the
  robot.
* **Watchdogs.** Measured base speed above ``max_base_speed_mps`` /
  ``max_base_omega_rps`` (odom), an arm joint faster than
  ``max_joint_speed_rps`` in two consecutive samples (finite differences of
  ``/joint_states``), odom older than ``max_feedback_age_s`` while the base is
  commanded, joint states stopping while the arm moved, a contact force above
  ``max_force_n`` (``force_topic``: the effort estimate or an F/T sensor),
  the base farther than ``geofence_radius_m`` from where the supervisor first
  saw it (0 = off).

A trip latches: the MPC is cancelled, zero servo twist and joint jog are sent,
the base gets a zero and no further commands, piper_servo_bridge is disabled
(and refuses ~/enable while the supervisor reports a trip, when started with
``safety_state_topic``), and the stem_grasp pipeline is aborted. ``~/reset``
clears it once its cause has gone (heartbeat back, speeds and feedback
normal, force below the limit; a geofence trip re-arms 10 cm farther out so
the base can be driven back); the servo bridge then still has to be enabled
again. ``~/trip`` trips by hand. State on ``/safety/state`` (JSON, 5 Hz).

    ros2 run scout_piper_bringup safety_supervisor.py --ros-args -p require_heartbeat:=true
    ros2 service call /safety_supervisor/reset std_srvs/srv/Trigger
"""

import json
import math
import signal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

ARM_JOINTS = tuple(f"piper_joint{i}" for i in range(1, 7))


@dataclass
class Limits:
    require_heartbeat: bool = False
    heartbeat_timeout_s: float = 1.0
    cmd_timeout_s: float = 0.5
    max_cmd_v_mps: float = 0.5           # relayed commands are clamped to these
    max_cmd_w_rps: float = 1.0
    max_base_speed_mps: float = 0.7      # measured (odom): trip above
    max_base_omega_rps: float = 1.4
    max_joint_speed_rps: float = 1.5     # measured (finite differences): trip above, twice in a row
    max_feedback_age_s: float = 0.5
    max_force_n: float = 0.0             # <= 0: off
    geofence_radius_m: float = 0.0       # <= 0: off
    geofence_rearm_m: float = 0.1
    moving_v_mps: float = 0.02           # "moving" thresholds
    moving_w_rps: float = 0.05
    moving_qd_rps: float = 0.03


@dataclass
class Supervisor:
    """Decisions only; the node feeds it (with explicit times) and acts on them."""

    limits: Limits = field(default_factory=Limits)
    expect_odom: bool = True
    expect_joints: bool = True
    tripped: str = ""
    trip_t: Optional[float] = None
    trips: int = 0
    cmd_timeouts: int = 0
    heartbeat_t: Optional[float] = None
    odom_t: Optional[float] = None
    base_xy: Optional[Tuple[float, float]] = None
    base_vw: Tuple[float, float] = (0.0, 0.0)
    fence_centre: Optional[Tuple[float, float]] = None
    fence_radius: float = 0.0
    joints_t: Optional[float] = None
    joints_stamp: Optional[float] = None
    q: Optional[List[float]] = None
    qd_max: float = 0.0
    overspeed_samples: int = 0
    force_n: Optional[float] = None
    force_t: Optional[float] = None
    cmd_t: Optional[float] = None          # last command received
    cmd_moving: bool = False               # ... and whether it asked for motion
    timeout_sent: bool = True

    # ------------------------------------------------------------ inputs
    def heartbeat(self, now: float) -> None:
        self.heartbeat_t = now

    def odom(self, now: float, x: float, y: float, v: float, w: float) -> None:
        self.odom_t, self.base_xy, self.base_vw = now, (x, y), (v, w)
        if self.fence_centre is None:
            self.fence_centre, self.fence_radius = (x, y), self.limits.geofence_radius_m

    def joints(self, now: float, stamp: float, q: Sequence[float]) -> None:
        """Arm joint positions with their header stamp (receive time if unset)."""
        q = [float(v) for v in q]
        if self.q is not None and self.joints_stamp is not None:
            dt = stamp - self.joints_stamp
            if dt < 0.005:                 # same or out-of-order sample: no speed from it
                self.joints_t = now
                return
            self.qd_max = max(abs(a - b) for a, b in zip(q, self.q)) / dt
            fast = self.qd_max > self.limits.max_joint_speed_rps
            self.overspeed_samples = self.overspeed_samples + 1 if fast else 0
        self.q, self.joints_stamp, self.joints_t = q, stamp, now

    def force(self, now: float, f: float) -> None:
        self.force_n, self.force_t = f, now

    def command(self, now: float, v: float, w: float) -> Tuple[float, float]:
        """The base command to relay for an incoming one: clamped, zero while tripped."""
        moving = abs(v) > 1e-6 or abs(w) > 1e-6
        if moving and not self.tripped and self.limits.require_heartbeat and not self._heartbeat_ok(now):
            self.trip(now, f"base command without operator heartbeat ({self._heartbeat_age(now)})")
        if self.tripped:
            self.cmd_t, self.cmd_moving = now, False
            return 0.0, 0.0
        L = self.limits
        v = max(-L.max_cmd_v_mps, min(L.max_cmd_v_mps, v))
        w = max(-L.max_cmd_w_rps, min(L.max_cmd_w_rps, w))
        self.cmd_t, self.cmd_moving = now, moving
        self.timeout_sent = not moving
        return v, w

    # ------------------------------------------------------------ checks
    def _heartbeat_ok(self, now: float) -> bool:
        return self.heartbeat_t is not None and now - self.heartbeat_t <= self.limits.heartbeat_timeout_s

    def _heartbeat_age(self, now: float) -> str:
        return "none yet" if self.heartbeat_t is None else f"last {now - self.heartbeat_t:.1f} s ago"

    def base_commanded(self, now: float) -> bool:
        return self.cmd_moving and self.cmd_t is not None and now - self.cmd_t <= self.limits.cmd_timeout_s

    def base_moving(self) -> bool:
        v, w = self.base_vw
        return abs(v) > self.limits.moving_v_mps or abs(w) > self.limits.moving_w_rps

    def arm_moving(self) -> bool:
        return self.qd_max > self.limits.moving_qd_rps

    def moving(self, now: float) -> bool:
        return self.base_commanded(now) or self.base_moving() or self.arm_moving()

    def fault(self, now: float) -> str:
        """The first unsafe condition now, or ""."""
        L = self.limits
        moving = self.moving(now)
        if L.require_heartbeat and moving and not self._heartbeat_ok(now):
            return f"operator heartbeat lost ({self._heartbeat_age(now)}) while moving"
        v, w = self.base_vw
        if self.odom_t is not None and now - self.odom_t <= L.max_feedback_age_s:
            if abs(v) > L.max_base_speed_mps or abs(w) > L.max_base_omega_rps:
                return f"base overspeed: {v:.2f} m/s, {w:.2f} rad/s"
        if self.overspeed_samples >= 2:
            return f"arm joint overspeed: {self.qd_max:.2f} rad/s"
        if self.expect_odom and self.base_commanded(now) and (
                self.odom_t is None or now - self.odom_t > L.max_feedback_age_s):
            return "odom stale while the base is commanded"
        if (self.expect_joints and self.arm_moving() and self.joints_t is not None
                and now - self.joints_t > L.max_feedback_age_s):
            return "joint states stopped while the arm moved"
        if L.max_force_n > 0 and self.force_n is not None and self.force_t is not None \
                and now - self.force_t <= L.max_feedback_age_s and self.force_n > L.max_force_n:
            return f"contact force {self.force_n:.1f} N above {L.max_force_n:.1f} N"
        if self.fence_radius > 0 and self.base_xy is not None and self.fence_centre is not None:
            d = math.dist(self.base_xy, self.fence_centre)
            if d > self.fence_radius:
                return f"base {d:.2f} m from the start, geofence {self.fence_radius:.2f} m"
        return ""

    def trip(self, now: float, reason: str) -> bool:
        """Latch a trip; True when this call tripped (not already tripped)."""
        if self.tripped:
            return False
        self.tripped, self.trip_t = reason, now
        self.trips += 1
        self.cmd_moving = False
        return True

    def tick(self, now: float) -> List[str]:
        """Periodic check. Actions: "trip" (just tripped), "zero_base" (command timeout)."""
        actions = []
        reason = "" if self.tripped else self.fault(now)
        if reason and self.trip(now, reason):
            actions.append("trip")
        if (not self.tripped and not self.timeout_sent and self.cmd_t is not None
                and now - self.cmd_t > self.limits.cmd_timeout_s):
            self.timeout_sent, self.cmd_moving = True, False
            self.cmd_timeouts += 1
            actions.append("zero_base")
        return actions

    def reset(self, now: float) -> Tuple[bool, str]:
        if not self.tripped:
            return True, "not tripped"
        geofence = self.tripped.startswith("base ") and "geofence" in self.tripped
        if geofence and self.base_xy is not None and self.fence_centre is not None:
            # re-arm farther out so the base can be driven back toward the start
            self.fence_radius = math.dist(self.base_xy, self.fence_centre) + self.limits.geofence_rearm_m
        self.overspeed_samples = 0
        reason = self.fault(now)
        if reason:
            return False, f"still unsafe: {reason}"
        was, self.tripped, self.trip_t = self.tripped, "", None
        return True, f"reset (was: {was}); enable the servo bridge again to move the arm"

    def state(self, now: float) -> Dict:
        def age(t):
            return None if t is None else round(now - t, 2)
        return {"ok": not self.tripped, "tripped": self.tripped, "trip_age_s": age(self.trip_t),
                "trips": self.trips, "moving": self.moving(now), "base_commanded": self.base_commanded(now),
                "require_heartbeat": self.limits.require_heartbeat, "heartbeat_age_s": age(self.heartbeat_t),
                "odom_age_s": age(self.odom_t), "joints_age_s": age(self.joints_t),
                "base_v": round(self.base_vw[0], 3), "base_w": round(self.base_vw[1], 3),
                "joint_speed_max": round(self.qd_max, 3), "force_n": self.force_n,
                "geofence_m": self.fence_radius if self.fence_radius > 0 else None,
                "cmd_timeouts": self.cmd_timeouts}


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    import rclpy  # noqa: PLC0415 — keep the logic importable without ROS
    from geometry_msgs.msg import Twist, TwistStamped, WrenchStamped  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415
    from rclpy.executors import ExternalShutdownException  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from rclpy.signals import SignalHandlerOptions  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415
    from std_msgs.msg import Empty, String  # noqa: PLC0415
    from std_srvs.srv import SetBool, Trigger  # noqa: PLC0415
    try:
        from control_msgs.msg import JointJog  # noqa: PLC0415
    except ImportError:
        JointJog = None

    class SupervisorNode(Node):
        def __init__(self) -> None:
            super().__init__("safety_supervisor")
            d = Limits()
            p = self.declare_parameter
            lim = Limits(**{k: type(getattr(d, k))(p(k, getattr(d, k)).value) for k in d.__dataclass_fields__})
            self.sup = Supervisor(limits=lim, expect_odom=bool(p("expect_odom", True).value),
                                  expect_joints=bool(p("expect_joints", True).value))
            self.joint_names = list(p("joint_names", list(ARM_JOINTS)).value)
            in_topic, out_topic = p("input_cmd_topic", "/cmd_vel").value, p("output_cmd_topic", "/cmd_vel_safe").value
            self.pub_base = self.create_publisher(Twist, out_topic, 10)
            self.pub_cancel = self.create_publisher(Empty, p("mpc_cancel_topic", "/whole_body_mpc/cancel").value, 5)
            self.pub_twist = self.create_publisher(
                TwistStamped, p("servo_twist_topic", "/servo_node/delta_twist_cmds").value, 5)
            self.pub_jog = (self.create_publisher(JointJog, p("servo_jog_topic", "/servo_node/delta_joint_cmds").value, 5)
                            if JointJog is not None else None)
            self.pub_state = self.create_publisher(String, p("state_topic", "/safety/state").value, 5)
            self.bridge = self.create_client(SetBool, p("bridge_enable_service", "/piper_servo_bridge/enable").value)
            self.pipeline = self.create_client(Trigger, p("pipeline_abort_service", "/stem_grasp_pipeline/abort").value)
            self.create_subscription(Twist, in_topic, self._cmd, 20)
            self.create_subscription(Odometry, p("odom_topic", "/odom").value, self._odom, 20)
            self.create_subscription(JointState, p("joint_states_topic", "/joint_states").value, self._joints, 50)
            self.create_subscription(Empty, p("heartbeat_topic", "/operator/heartbeat").value,
                                     lambda m: self.sup.heartbeat(self._now()), 10)
            force_topic = p("force_topic", "").value
            if force_topic:
                self.create_subscription(WrenchStamped, force_topic, self._force, 20)
            self.create_service(Trigger, "~/reset", self._reset)
            self.create_service(Trigger, "~/trip", self._trip_srv)
            self.create_timer(0.05, self._tick)
            self.create_timer(0.2, self._publish_state)
            self.get_logger().info(
                f"safety supervisor: {in_topic} -> {out_topic}; heartbeat "
                f"{'required' if lim.require_heartbeat else 'not required'}")

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _cmd(self, msg: Twist) -> None:
            trips = self.sup.trips
            v, w = self.sup.command(self._now(), msg.linear.x, msg.angular.z)
            if self.sup.trips > trips:
                self._stop(self.sup.tripped)           # this command tripped it
            out = Twist()
            out.linear.x, out.angular.z = v, w
            self.pub_base.publish(out)

        def _odom(self, msg: Odometry) -> None:
            t = msg.twist.twist
            self.sup.odom(self._now(), msg.pose.pose.position.x, msg.pose.pose.position.y, t.linear.x, t.angular.z)

        def _joints(self, msg: JointState) -> None:
            idx = {n: i for i, n in enumerate(msg.name)}
            if not all(n in idx for n in self.joint_names):
                return
            now = self._now()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.sup.joints(now, stamp if stamp > 0 else now, [msg.position[idx[n]] for n in self.joint_names])

        def _force(self, msg: WrenchStamped) -> None:
            f = msg.wrench.force
            self.sup.force(self._now(), math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))

        def _tick(self) -> None:
            for a in self.sup.tick(self._now()):
                if a == "trip":
                    self._stop(self.sup.tripped)
                elif a == "zero_base":
                    self.pub_base.publish(Twist())
                    self.get_logger().info("no base command for "
                                           f"{self.sup.limits.cmd_timeout_s:.1f} s after a moving one: zero sent")

        def _stop(self, reason: str) -> None:
            now = self.get_clock().now().to_msg()
            self.pub_cancel.publish(Empty())
            for _ in range(3):
                self.pub_base.publish(Twist())
                t = TwistStamped()
                t.header.stamp = now
                self.pub_twist.publish(t)
                if self.pub_jog is not None:
                    j = JointJog()
                    j.header.stamp = now
                    self.pub_jog.publish(j)
            if self.bridge.service_is_ready():
                self.bridge.call_async(SetBool.Request(data=False))
            if self.pipeline.service_is_ready():
                self.pipeline.call_async(Trigger.Request())
            self.get_logger().error(f"SAFETY TRIP: {reason}; MPC cancelled, zero commands, servo bridge "
                                    "disabled, pipeline aborted. ~/reset when the cause has gone")
            self._publish_state()

        def _reset(self, req, res):
            res.success, res.message = self.sup.reset(self._now())
            (self.get_logger().warn if res.success else self.get_logger().error)(f"reset: {res.message}")
            self._publish_state()
            return res

        def _trip_srv(self, req, res):
            if self.sup.trip(self._now(), "manual trip (~/trip)"):
                self._stop(self.sup.tripped)
            res.success, res.message = True, self.sup.tripped
            return res

        def _publish_state(self) -> None:
            self.pub_state.publish(String(data=json.dumps(self.sup.state(self._now()))))

        def stop_base(self) -> None:
            self.pub_base.publish(Twist())

    # Ctrl-C / SIGTERM: one zero to the base on the way out (the gate's output is
    # the only command the driver sees, and the driver keeps the last one)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = SupervisorNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.stop_base()
        except Exception:  # noqa: BLE001 — the context may already be gone
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
