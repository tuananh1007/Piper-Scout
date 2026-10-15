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
It is a kinematic stand-in, not a model of the Piper's dynamics. An object
between the fingers can be simulated with the ``object_width_m`` parameter
(settable at run time): a closing gripper stops at that width.

Joint efforts (N·m) are simulated as the torque that holds the arm against
gravity (URDF masses, scout_piper_whole_body_mpc.dynamics.effort), plus
friction, ``effort_noise_nm`` of noise and the torque of a force at the TCP,
``external_force_n`` [fx, fy, fz] in base_link (settable at run time), so the
joint-effort force estimate (effort_force_node) can be tested.
``effort_sign`` -1 mimics a driver with the opposite sign convention.
"""

import random
import signal
from typing import List, Optional, Sequence, Tuple

JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# friction of the simulated joints (offset, Coulomb, viscous), N·m and N·m·s/rad
FAKE_FRICTION = {"b": [0.05, -0.1, 0.05, 0.0, 0.02, 0.0], "c": [0.3, 0.4, 0.2, 0.05, 0.05, 0.02],
                 "d": [0.3, 0.5, 0.3, 0.1, 0.1, 0.05]}
_EFFORT = {}


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


def gripper_step(width: float, target: float, max_delta: float, object_width: float = 0.0) -> float:
    """Next gripper opening; a closing gripper stops at an object of ``object_width``."""
    new = step_toward([width], [target], max_delta)[0]
    if object_width > 0.0 and width >= object_width > new:
        return object_width
    return new


def simulated_effort(q: Sequence[float], qd: Sequence[float], force: Sequence[float] = (0.0, 0.0, 0.0),
                     sign: float = 1.0, tcp_offset_m: float = 0.14) -> Optional[List[float]]:
    """Noise-free efforts of joint1..6, or None without scout_piper_whole_body_mpc."""
    key = (float(sign), float(tcp_offset_m))
    if key not in _EFFORT:
        try:
            from scout_piper_whole_body_mpc.dynamics.effort import EffortModel  # noqa: PLC0415
            from scout_piper_whole_body_mpc.dynamics.piper import PiperKinematics  # noqa: PLC0415
        except ImportError:
            return None
        _EFFORT[key] = (PiperKinematics(tcp_offset_m=tcp_offset_m),
                        EffortModel(a=[float(sign)] * 6, **{k: [float(sign) * x for x in v]
                                                            for k, v in FAKE_FRICTION.items()}))
    kin, model = _EFFORT[key]
    return [float(v) for v in model.torque(kin, q, qd, force)]


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
            p("object_width_m", 0.0)                 # read every tick: settable at run time
            p("external_force_n", [0.0, 0.0, 0.0])   # read every tick: settable at run time
            self.effort_on = bool(p("simulate_effort", True).value)
            self.effort_noise = float(p("effort_noise_nm", 0.02).value)
            self.effort_sign = float(p("effort_sign", 1.0).value)
            self.tcp_offset = float(p("tcp_offset_m", 0.14).value)
            self.rng = random.Random(0)
            if self.effort_on and simulated_effort(self.q, [0.0] * 6) is None:
                self.get_logger().warn("scout_piper_whole_body_mpc not found: no joint efforts published")
                self.effort_on = False
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
            self.gripper = gripper_step(self.gripper, self.gripper_target,
                                        f * self.gripper_vmax * self.dt,
                                        float(self.get_parameter("object_width_m").value))
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name = list(JOINTS) + ["gripper"]
            out.position = self.q + [self.gripper]
            out.velocity = [(a - b) / self.dt for a, b in zip(self.q, q_old)]
            if self.effort_on:
                force = list(self.get_parameter("external_force_n").value)
                force = force if len(force) == 3 else [0.0, 0.0, 0.0]
                tau = simulated_effort(self.q, out.velocity, force, self.effort_sign, self.tcp_offset)
                out.effort = [t + self.rng.gauss(0.0, self.effort_noise) for t in tau] + [0.0]
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
