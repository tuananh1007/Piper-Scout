"""Hotkey e-stop + zero-arm helper — ROS 2 Humble port skeleton.

Ports stem_grasp_ros1/scripts/hotkey_stop_and_zero.py.
- 'x' : publish zero-twist 5x to /servo_node/delta_twist_cmds, then call
        /piper/zero_arm service.
- 'q' : graceful quit.
"""

import sys
import termios
import tty
from typing import Optional

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node


class HotkeyStop(Node):
    def __init__(self) -> None:
        super().__init__("stem_grasp_hotkey")
        self.declare_parameter("servo_cmd_topic", "/servo_node/delta_twist_cmds")
        topic = self.get_parameter("servo_cmd_topic").value
        self.pub = self.create_publisher(TwistStamped, topic, 5)
        self.get_logger().info(
            f"Hotkey ready. Publishing to {topic}. Press 'x' to e-stop, 'q' to quit."
        )

    def emit_zero_twist(self, n: int = 5) -> None:
        for _ in range(n):
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)


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
                    node.get_logger().warn("E-STOP: publishing zero twist x5.")
                    node.emit_zero_twist()
                elif key in ("q", "\x03"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
