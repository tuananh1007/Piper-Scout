#!/usr/bin/env python3
"""Hardware-free stand-in for the Scout 2.0 base driver (scout_ros2 ``scout_base``).

full_system.launch.py starts it instead of the real driver with
``bringup_base:=true fake_base:=true``. Like the real driver it subscribes to
``/cmd_vel`` (linear.x, angular.z) and publishes ``odom`` (nav_msgs/Odometry,
``odom`` -> ``base_link``) and the matching TF.

Command handling mirrors what is known about the real stack: ``scout_ros2``
and ``ugv_sdk`` have no command timeout, so the last ``/cmd_vel`` stays in
force until a new one arrives (``cmd_timeout_s: 0``, the default). Whether
the Scout firmware stops on its own is not documented (INSTALL.md 10.5); set
``cmd_timeout_s`` > 0 to model such a timeout.

The base tracks the commanded velocities with a first-order lag
(``time_constant_s``) within the Scout's limits and integrates a unicycle; it
is a kinematic stand-in, not a model of skid-steer slip.
"""

import math
import signal
from typing import Tuple


def unicycle_step(x: float, y: float, yaw: float, v: float, w: float, dt: float
                  ) -> Tuple[float, float, float]:
    """Exact unicycle integration over dt at constant (v, w)."""
    if abs(w) < 1e-9:
        return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw
    yaw2 = yaw + w * dt
    return (x + v / w * (math.sin(yaw2) - math.sin(yaw)),
            y - v / w * (math.cos(yaw2) - math.cos(yaw)),
            math.atan2(math.sin(yaw2), math.cos(yaw2)))


def track(current: float, command: float, dt: float, tau: float, limit: float) -> float:
    """First-order tracking of a velocity command, clamped to +-limit."""
    command = max(-limit, min(limit, command))
    if tau <= 0:
        return command
    return current + (command - current) * (1.0 - math.exp(-dt / tau))


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    import rclpy  # noqa: PLC0415
    from geometry_msgs.msg import TransformStamped, Twist  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415
    from rclpy.executors import ExternalShutdownException  # noqa: PLC0415
    from rclpy.signals import SignalHandlerOptions  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from tf2_ros import TransformBroadcaster  # noqa: PLC0415

    class FakeBase(Node):
        def __init__(self) -> None:
            super().__init__("scout_base_node")
            p = self.declare_parameter
            self.odom_frame = p("odom_frame", "odom").value
            self.base_frame = p("base_frame", "base_link").value
            self.dt = 1.0 / float(p("rate_hz", 50.0).value)
            self.tau = float(p("time_constant_s", 0.15).value)
            self.v_lim = float(p("max_linear_speed", 1.5).value)      # Scout 2.0 spec
            self.w_lim = float(p("max_angular_speed", 1.0).value)
            self.timeout = float(p("cmd_timeout_s", 0.0).value)       # 0: no timeout (scout_ros2)
            self.x = self.y = self.yaw = 0.0
            self.v = self.w = 0.0
            self.cmd = (0.0, 0.0)
            self.t_cmd = None
            self.pub = self.create_publisher(Odometry, "odom", 50)
            self.tf = TransformBroadcaster(self)
            self.create_subscription(Twist, "/cmd_vel", self._cmd, 10)
            self.create_timer(self.dt, self._tick)
            self.get_logger().warn("FAKE Scout base: no hardware is commanded")

        def _cmd(self, msg: Twist) -> None:
            self.cmd = (msg.linear.x, msg.angular.z)
            self.t_cmd = self.get_clock().now()

        def _tick(self) -> None:
            now = self.get_clock().now()
            v_cmd, w_cmd = self.cmd
            if self.timeout > 0 and (self.t_cmd is None
                                     or (now - self.t_cmd).nanoseconds * 1e-9 > self.timeout):
                v_cmd = w_cmd = 0.0
            self.v = track(self.v, v_cmd, self.dt, self.tau, self.v_lim)
            self.w = track(self.w, w_cmd, self.dt, self.tau, self.w_lim)
            self.x, self.y, self.yaw = unicycle_step(self.x, self.y, self.yaw, self.v, self.w, self.dt)
            stamp = now.to_msg()
            qz, qw = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id, tf.child_frame_id = self.odom_frame, self.base_frame
            tf.transform.translation.x, tf.transform.translation.y = self.x, self.y
            tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
            self.tf.sendTransform(tf)
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id, odom.child_frame_id = self.odom_frame, self.base_frame
            odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
            odom.twist.twist.linear.x, odom.twist.twist.angular.z = self.v, self.w
            self.pub.publish(odom)

    # Ctrl-C / SIGTERM end spin with KeyboardInterrupt; rclpy's own handler
    # can invalidate the context while spin builds its wait set (RCLError).
    # spin_once with a timeout lets the handler run when no message arrives.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = FakeBase()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
