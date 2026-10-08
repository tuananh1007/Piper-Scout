#!/usr/bin/env python3
"""Hardware-free stand-in for the Piper driver (piper_ros ``piper_single_ctrl``).

For testing the command chain (moveit_servo -> piper_servo_bridge -> driver ->
piper_joint_state_relay) without the arm: full_system.launch.py starts it instead
of the real driver with ``bringup_arm:=true fake_arm:=true``. It mimics the
real driver's command handling, hazards included, so tests see what the arm
would do:

- commands arrive as sensor_msgs/JointState on ``joint_ctrl_single`` (remapped
  like the real driver);
- a joint missing from the command is commanded to 0;
- with fewer than seven positions the gripper is commanded to 0 (closed);
- the speed is ``velocity[6]`` percent when seven velocities are given and not
  all zero, otherwise 100 %.

The simulated arm moves each joint toward its target at ``speed % x
max_joint_speed`` and publishes ``joint_states_single`` (joint1..6, gripper).
It is a kinematic stand-in, not a model of the Piper's dynamics.
"""

import signal
from typing import List, Sequence, Tuple

JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def interpret_command(names: Sequence[str], position: Sequence[float], velocity: Sequence[float]
                      ) -> Tuple[List[float], float, float]:
    """(joint targets, gripper target, speed percent) as the real driver reads them."""
    pos = dict(zip(names, position))
    targets = [float(pos.get(j, 0.0)) for j in JOINTS]
    gripper = float(position[6]) if len(position) >= 7 else 0.0
    if velocity and not all(v == 0 for v in velocity):
        speed = min(max(round(velocity[6]), 1), 100) if len(velocity) == 7 else 100
    else:
        speed = 100
    return targets, gripper, float(speed)


def step_toward(current: Sequence[float], target: Sequence[float], max_delta: float) -> List[float]:
    return [c + min(max(t - c, -max_delta), max_delta) for c, t in zip(current, target)]


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    import rclpy  # noqa: PLC0415
    from rclpy.executors import ExternalShutdownException  # noqa: PLC0415
    from rclpy.signals import SignalHandlerOptions  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415

    class FakeDriver(Node):
        def __init__(self) -> None:
            super().__init__("piper_ctrl_single_node")
            p = self.declare_parameter
            # a working pose with a well-conditioned Jacobian (servo.yaml)
            self.q = [float(v) for v in p("initial_positions", [0.0, 1.5, -1.0, 0.0, 0.6, 0.0]).value]
            self.gripper = float(p("initial_gripper", 0.03).value)
            self.vmax = float(p("max_joint_speed", 3.0).value)        # rad/s at 100 %
            self.gripper_vmax = float(p("max_gripper_speed", 0.05).value)  # m/s at 100 %
            self.dt = 1.0 / float(p("rate_hz", 100.0).value)
            self.target, self.gripper_target, self.speed = list(self.q), self.gripper, 100.0
            self.pub = self.create_publisher(JointState, "joint_states_single", 10)
            self.create_subscription(JointState, "joint_ctrl_single", self._cmd, 10)
            self.create_timer(self.dt, self._tick)
            self.get_logger().warn("FAKE Piper driver: no hardware is commanded")

        def _cmd(self, msg: JointState) -> None:
            self.target, self.gripper_target, self.speed = interpret_command(
                msg.name, msg.position, msg.velocity)

        def _tick(self) -> None:
            f = self.speed / 100.0
            q_old = self.q
            self.q = step_toward(self.q, self.target, f * self.vmax * self.dt)
            self.gripper = step_toward([self.gripper], [self.gripper_target],
                                       f * self.gripper_vmax * self.dt)[0]
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name = list(JOINTS) + ["gripper"]
            out.position = self.q + [self.gripper]
            out.velocity = [(a - b) / self.dt for a, b in zip(self.q, q_old)]
            self.pub.publish(out)

    # Ctrl-C / SIGTERM end spin with KeyboardInterrupt; rclpy's own handler
    # can invalidate the context while spin builds its wait set (RCLError).
    # spin_once with a timeout lets the handler run when no message arrives.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = FakeDriver()
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
