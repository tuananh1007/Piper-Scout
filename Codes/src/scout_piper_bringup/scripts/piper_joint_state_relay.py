#!/usr/bin/env python3
"""Republish the Piper driver's joint feedback under the unified URDF's names.

The upstream driver (piper_ros, ``piper_single_ctrl``) publishes its feedback
on ``/joint_states_single`` with names ``joint1``..``joint6`` plus ``gripper``
(total opening in metres). The unified URDF, robot_state_publisher, the
whole-body MPC and plant_twin read ``/joint_states`` with ``piper_joint1``..
``piper_joint6`` and the two mirrored finger joints ``piper_joint7`` (0..0.035 m)
and ``piper_joint8`` (-0.035..0 m). This node bridges the two.

Safety: the upstream launch file makes the driver execute ``/joint_states`` as
joint commands. full_system.launch.py starts the driver with its command input
remapped to ``arm_command_topic`` instead; this relay refuses to start if its
output topic equals that command topic, so it can never feed the arm's own
state back as a command.

    in   /joint_states_single   sensor_msgs/JointState  joint1..joint6, gripper
    out  /joint_states          sensor_msgs/JointState  piper_joint1..piper_joint8
"""

from typing import List, Optional, Sequence, Tuple

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER = "gripper"


def relay_joint_state(names: Sequence[str], position: Sequence[float], velocity: Sequence[float],
                      prefix: str = "piper_",
                      finger_joints: Tuple[str, str] = ("piper_joint7", "piper_joint8"),
                      finger_max_m: float = 0.035
                      ) -> Optional[Tuple[List[str], List[float], List[float]]]:
    """Driver feedback -> (names, positions, velocities) for /joint_states.

    Returns None unless all six arm joints are present. The gripper opening is
    split evenly between the two fingers; the driver reports no finger
    velocity, so finger velocities are 0. Effort is not relayed.
    """
    pos = dict(zip(names, position))
    if len(velocity) == len(names):
        vel = dict(zip(names, velocity))
    elif len(velocity) == len(ARM_JOINTS):          # the driver sends 6 velocities for its 7 names
        vel = dict(zip(ARM_JOINTS, velocity))
    else:
        vel = {}
    if not all(j in pos for j in ARM_JOINTS):
        return None
    out_names = [prefix + j for j in ARM_JOINTS]
    out_pos = [float(pos[j]) for j in ARM_JOINTS]
    out_vel = [float(vel.get(j, 0.0)) for j in ARM_JOINTS]
    if GRIPPER in pos and finger_joints:
        half = min(max(float(pos[GRIPPER]) / 2.0, 0.0), finger_max_m)
        out_names += list(finger_joints)
        out_pos += [half, -half]
        out_vel += [0.0, 0.0]
    return out_names, out_pos, out_vel


def _same_topic(a: str, b: str) -> bool:
    return a.strip().lstrip("/") == b.strip().lstrip("/")


def main() -> None:
    import rclpy  # noqa: PLC0415 — keep the conversion importable without ROS
    from rclpy.executors import ExternalShutdownException  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415

    class Relay(Node):
        def __init__(self) -> None:
            super().__init__("piper_joint_state_relay")
            self.input_topic = self.declare_parameter("input_topic", "/joint_states_single").value
            self.output_topic = self.declare_parameter("output_topic", "/joint_states").value
            command_topic = self.declare_parameter("arm_command_topic", "/piper/joint_cmd").value
            self.prefix = self.declare_parameter("prefix", "piper_").value
            fingers = list(self.declare_parameter("finger_joints", ["piper_joint7", "piper_joint8"]).value)
            self.fingers = tuple(fingers) if len(fingers) == 2 else ()
            self.finger_max = float(self.declare_parameter("finger_max_m", 0.035).value)
            if _same_topic(self.output_topic, command_topic):
                raise RuntimeError(f"output_topic {self.output_topic} is the arm command topic: "
                                   "relaying feedback there would command the arm")
            self.pub = self.create_publisher(JointState, self.output_topic, 10)
            self.create_subscription(JointState, self.input_topic, self._cb, 10)
            self._warned = False
            self.get_logger().info(f"{self.input_topic} -> {self.output_topic} (prefix '{self.prefix}')")

        def _cb(self, msg: JointState) -> None:
            r = relay_joint_state(msg.name, msg.position, msg.velocity, self.prefix, self.fingers,
                                  self.finger_max)
            if r is None:
                if not self._warned:
                    self.get_logger().warn(f"{self.input_topic} lacks joint1..joint6: {list(msg.name)}")
                    self._warned = True
                return
            out = JointState()
            out.header = msg.header
            out.name, out.position, out.velocity = r
            self.pub.publish(out)

    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # Ctrl-C / SIGTERM from ros2 launch
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
