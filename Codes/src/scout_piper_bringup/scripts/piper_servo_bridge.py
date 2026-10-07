#!/usr/bin/env python3
"""Forward moveit_servo's joint targets to the Piper driver, with safety latches.

moveit_servo publishes ``trajectory_msgs/JointTrajectory`` (``piper_joint1..6``,
one point, absolute positions). The Piper driver (piper_ros
``piper_single_ctrl``) takes ``sensor_msgs/JointState`` with names
``joint1..joint6`` + ``gripper`` on its command input, which full_system.launch.py
maps to ``/piper/joint_cmd``. The driver executes every message immediately:
missing joints are commanded to 0, a message with fewer than seven positions
closes the gripper, and the speed is ``velocity[6]`` percent (100 % otherwise).

This bridge therefore always sends all six joints plus the *measured* gripper
opening, sets the speed percent explicitly, and only forwards while:

- it has been enabled (``~/enable``, std_srvs/SetBool; starts disabled unless
  ``start_enabled``);
- the driver feedback (``/joint_states_single``) is fresher than
  ``max_feedback_age_s`` (servo itself stops publishing when its input times
  out, and the driver holds the last target);
- each target is clamped into the joint limits and to within ``max_step_rad`` of
  the measured position, so the arm never jumps even if servo's internal state
  drifts from the real arm.

Disabling sends one command holding the measured position.

    in   /piper/servo/joint_trajectory  trajectory_msgs/JointTrajectory  (servo)
    in   /joint_states_single           sensor_msgs/JointState           (driver feedback)
    out  /piper/joint_cmd               sensor_msgs/JointState           (driver command input)
    srv  ~/enable                       std_srvs/SetBool
"""

from typing import Dict, List, Optional, Sequence, Tuple

DRIVER_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER = "gripper"
# piper_joint1..6 limits of the unified URDF (_piper_arm.xacro)
URDF_LOWER = (-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944)
URDF_UPPER = (2.618, 3.14, 0.0, 1.745, 1.22, 2.0944)
JOINT_STATE_TOPICS = ("/joint_states", "/joint_states_single", "/joint_states_feedback")


def driver_command(target_names: Sequence[str], target_positions: Sequence[float],
                   measured: Dict[str, float], gripper_opening: float,
                   lower: Sequence[float] = URDF_LOWER, upper: Sequence[float] = URDF_UPPER,
                   max_step_rad: float = 0.1, speed_percent: int = 30, prefix: str = "piper_"
                   ) -> Optional[Tuple[List[str], List[float], List[float]]]:
    """Servo target -> (names, positions, velocities) for the driver, or None.

    ``measured`` maps driver joint names to measured positions. Returns None
    unless the target names exactly the six arm joints (in any order) and all
    six are measured. Joints are clamped into [lower, upper] and to within
    ``max_step_rad`` of the measured position.
    """
    tgt = {}
    for n, p in zip(target_names, target_positions):
        tgt[n[len(prefix):] if n.startswith(prefix) else n] = float(p)
    if set(tgt) != set(DRIVER_JOINTS) or not all(j in measured for j in DRIVER_JOINTS):
        return None
    positions = []
    for i, j in enumerate(DRIVER_JOINTS):
        q = min(max(tgt[j], lower[i]), upper[i])
        m = float(measured[j])
        positions.append(min(max(q, m - max_step_rad), m + max_step_rad))
    speed = float(min(max(int(speed_percent), 1), 100))
    return (list(DRIVER_JOINTS) + [GRIPPER], positions + [float(gripper_opening)],
            [0.0] * 6 + [speed])


def hold_command(measured: Dict[str, float], gripper_opening: float, speed_percent: int = 30
                 ) -> Optional[Tuple[List[str], List[float], List[float]]]:
    """Command that holds the measured pose (sent when the bridge is disabled)."""
    if not all(j in measured for j in DRIVER_JOINTS):
        return None
    speed = float(min(max(int(speed_percent), 1), 100))
    return (list(DRIVER_JOINTS) + [GRIPPER], [float(measured[j]) for j in DRIVER_JOINTS]
            + [float(gripper_opening)], [0.0] * 6 + [speed])


def check_command_topic(topic: str) -> str:
    t = "/" + topic.strip().lstrip("/")
    if t in JOINT_STATE_TOPICS:
        raise RuntimeError(f"command_topic {topic} is a joint-state topic")
    return t


def main() -> None:
    import rclpy  # noqa: PLC0415 — keep the conversion importable without ROS
    from rclpy.executors import ExternalShutdownException  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415
    from std_srvs.srv import SetBool  # noqa: PLC0415
    from trajectory_msgs.msg import JointTrajectory  # noqa: PLC0415

    class Bridge(Node):
        def __init__(self) -> None:
            super().__init__("piper_servo_bridge")
            p = self.declare_parameter
            traj_topic = p("trajectory_topic", "/piper/servo/joint_trajectory").value
            fb_topic = p("feedback_topic", "/joint_states_single").value
            cmd_topic = check_command_topic(p("command_topic", "/piper/joint_cmd").value)
            self.enabled = bool(p("start_enabled", False).value)
            self.speed = int(p("speed_percent", 30).value)
            self.max_step = float(p("max_step_rad", 0.1).value)
            self.max_fb_age = float(p("max_feedback_age_s", 0.2).value)
            self.lower = list(p("joint_lower", list(URDF_LOWER)).value)
            self.upper = list(p("joint_upper", list(URDF_UPPER)).value)
            self.prefix = p("prefix", "piper_").value
            self.measured: Dict[str, float] = {}
            self.gripper = 0.0
            self.fb_time = None
            self.pub = self.create_publisher(JointState, cmd_topic, 10)
            self.create_subscription(JointState, fb_topic, self._feedback, 10)
            self.create_subscription(JointTrajectory, traj_topic, self._target, 10)
            self.create_service(SetBool, "~/enable", self._enable)
            self.get_logger().info(f"{traj_topic} -> {cmd_topic}; "
                                   f"{'ENABLED' if self.enabled else 'disabled (call ~/enable)'}; "
                                   f"speed {self.speed} %, max step {self.max_step} rad")

        def _age(self, t) -> float:
            return float("inf") if t is None else (self.get_clock().now() - t).nanoseconds * 1e-9

        def _feedback(self, msg: JointState) -> None:
            pos = dict(zip(msg.name, msg.position))
            self.measured = {j: pos[j] for j in DRIVER_JOINTS if j in pos}
            self.gripper = float(pos.get(GRIPPER, self.gripper))
            self.fb_time = self.get_clock().now()

        def _send(self, cmd) -> None:
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name, out.position, out.velocity = cmd
            self.pub.publish(out)

        def _target(self, msg: JointTrajectory) -> None:
            if not self.enabled or not msg.points:
                return
            if self._age(self.fb_time) > self.max_fb_age:
                self.get_logger().warn("driver feedback stale; not forwarding",
                                       throttle_duration_sec=2.0)
                return
            cmd = driver_command(msg.joint_names, msg.points[0].positions, self.measured,
                                 self.gripper, self.lower, self.upper, self.max_step,
                                 self.speed, self.prefix)
            if cmd is None:
                self.get_logger().warn(f"unexpected servo joints {list(msg.joint_names)}",
                                       throttle_duration_sec=2.0)
                return
            self._send(cmd)

        def _enable(self, req, res):
            self.enabled = bool(req.data)
            if not self.enabled and self._age(self.fb_time) <= self.max_fb_age:
                hold = hold_command(self.measured, self.gripper, self.speed)
                if hold is not None:
                    self._send(hold)
            res.success = True
            res.message = "enabled" if self.enabled else "disabled (holding measured pose)"
            self.get_logger().warn(f"servo bridge {res.message}")
            return res

    rclpy.init()
    node = Bridge()
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
