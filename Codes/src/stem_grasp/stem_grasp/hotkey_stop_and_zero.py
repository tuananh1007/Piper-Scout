"""Hotkey stop for servo-driven arm motion — ROS 2 Humble port.

Ports stem_grasp_ros1/scripts/hotkey_stop_and_zero.py.
- 'x' : publish a zero twist 5x to /servo_node/delta_twist_cmds, then disable
        piper_servo_bridge (/piper_servo_bridge/enable false), which makes the
        Piper hold its measured pose and stops forwarding servo commands.
        Re-enable the bridge to continue.
- 'q' : graceful quit.

This is a software stop for motion that goes through moveit_servo only. It does
not stop the Scout base (/cmd_vel), does not cut power, and the ROS 1
"zero arm" service has no ROS 2 counterpart yet. Keep the physical stops in
reach (INSTALL.md, Step 9).
"""

import sys
import termios
import tty
from typing import Optional

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_srvs.srv import SetBool


class HotkeyStop(Node):
    def __init__(self) -> None:
        super().__init__("stem_grasp_hotkey")
        self.declare_parameter("servo_cmd_topic", "/servo_node/delta_twist_cmds")
        topic = self.get_parameter("servo_cmd_topic").value
        self.pub = self.create_publisher(TwistStamped, topic, 5)
        self.declare_parameter("servo_bridge_enable_service", "/piper_servo_bridge/enable")
        self.bridge = self.create_client(
            SetBool, self.get_parameter("servo_bridge_enable_service").value)
        self.get_logger().info(
            f"Hotkey ready. Publishing to {topic}. Press 'x' to stop servo motion, 'q' to quit."
        )

    def emit_zero_twist(self, n: int = 5) -> None:
        for _ in range(n):
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)

    def disable_servo_bridge(self, timeout_sec: float = 1.0) -> bool:
        """Disable piper_servo_bridge; True once it confirmed."""
        if not self.bridge.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("piper_servo_bridge not running: nothing to disable")
            return False
        future = self.bridge.call_async(SetBool.Request(data=False))
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        ok = future.done() and future.result() is not None and future.result().success
        if ok:
            self.get_logger().warn(f"servo bridge: {future.result().message}")
        else:
            self.get_logger().error("servo bridge did not confirm the disable")
        return ok


def _read_key() -> Optional[str]:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HotkeyStop()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            # Non-blocking key read is tricky in a portable way; here we use
            # a simple blocking read because this helper is interactive.
            if sys.stdin.isatty():
                key = _read_key()
                if key == "x":
                    node.get_logger().warn("STOP: zero twist x5, disabling the servo bridge.")
                    node.emit_zero_twist()
                    node.disable_servo_bridge()
                elif key in ("q", "\x03"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
