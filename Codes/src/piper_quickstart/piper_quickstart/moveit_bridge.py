"""Bridge between MoveIt 2 and the piper_ros driver node, for the real arm.

Upstream piper_ros drives the arm by running MoveIt against *mock* ros2_control
hardware and forwarding the mock /joint_states to the arm. MoveIt then shows
what was commanded rather than where the arm actually is, and the arm is
commanded to the mock's initial state (all zeros) the moment both start.

This node replaces that loop:

  piper driver  --joint_states_single (measured, 200 Hz)-->  bridge  --/joint_states-->  RSP, MoveIt
                 (only messages whose stamp advanced are forwarded: the driver
                  repeats cached values when no new CAN feedback has arrived)
  MoveIt  --FollowJointTrajectory (arm_controller, gripper_controller)-->  bridge
  bridge  --joint_ctrl_single (streamed setpoints)-->  piper driver  --CAN-->  arm

so /joint_states is always the measured state, and the arm only moves while a
trajectory MoveIt planned from that measured state is executing. Nothing is
sent to the arm until the first goal arrives.

Also published, for recording:
  piper_bridge/executed_trajectory (trajectory_msgs/JointTrajectory): every
      accepted goal, header.stamp = the time execution started.

Freeze (std_srvs/Trigger):
  /piper/freeze    abort whatever is executing and command the arm to hold the
                   position it was measured at; reject every new goal.
  /piper/unfreeze  accept goals again (the arm keeps holding until one arrives).
  /piper/frozen    std_msgs/Bool, latched: current state.
Freeze is a software hold through the driver: the motors stay powered and
holding. If the driver, CAN or this node is the problem, use the arm's power
switch.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from piper_quickstart.freshness import StampWatch, stamp_to_ns
from piper_quickstart.trajectory import (
    ARM_JOINTS,
    GRIPPER_JOINT,
    MIRROR_GRIPPER_JOINT,
    driver_to_urdf,
    max_abs_error,
    sample_trajectory,
)


@dataclass
class _ActiveGoal:
    controller: str
    joints: List[str]
    times: np.ndarray
    positions: np.ndarray
    start_mono: float

    @property
    def duration(self) -> float:
        return float(self.times[-1])


class PiperMoveItBridge(Node):
    def __init__(self) -> None:
        super().__init__("piper_moveit_bridge")

        self.declare_parameter("rate_hz", 100.0)
        # Driver "gripper" = total opening [m]; URDF joint7 = one finger.
        self.declare_parameter("gripper_scale", 2.0)
        self.declare_parameter("state_timeout_s", 0.5)
        self.declare_parameter("start_tolerance_rad", 0.15)
        self.declare_parameter("arm_goal_tolerance_rad", 0.03)
        self.declare_parameter("settle_timeout_s", 1.5)
        # Sent in JointState.effort[6]; the driver clips to [0.5, 3.0] N·m.
        self.declare_parameter("gripper_effort", 1.0)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._rate_hz = float(p("rate_hz"))
        self._gripper_scale = float(p("gripper_scale"))
        self._state_timeout = float(p("state_timeout_s"))
        self._start_tol = float(p("start_tolerance_rad"))
        self._goal_tol = float(p("arm_goal_tolerance_rad"))
        self._settle_timeout = float(p("settle_timeout_s"))
        self._gripper_effort = float(p("gripper_effort"))

        self._lock = threading.Lock()
        self._measured: Dict[str, float] = {}
        # Freshness = time since the driver's header stamp last advanced. The
        # driver republishes cached values at 200 Hz even if CAN goes quiet.
        self._watch = StampWatch()
        self._target: Dict[str, float] = {}  # last commanded setpoint, URDF names
        self._streaming = False
        self._active: Dict[str, _ActiveGoal] = {}
        self._frozen = False

        cb = ReentrantCallbackGroup()
        self._js_pub = self.create_publisher(JointState, "joint_states", 10)
        self._cmd_pub = self.create_publisher(JointState, "joint_ctrl_single", 10)
        self._traj_pub = self.create_publisher(
            JointTrajectory,
            "piper_bridge/executed_trajectory",
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_subscription(JointState, "joint_states_single", self._on_driver_state, 10, callback_group=cb)
        self.create_timer(1.0 / self._rate_hz, self._stream, callback_group=cb)
        self._frozen_pub = self.create_publisher(
            Bool, "/piper/frozen",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        self._frozen_pub.publish(Bool(data=False))
        self.create_service(Trigger, "/piper/freeze", self._on_freeze, callback_group=cb)
        self.create_service(Trigger, "/piper/unfreeze", self._on_unfreeze, callback_group=cb)

        self._servers = [
            self._make_server("arm_controller", ARM_JOINTS, cb),
            self._make_server("gripper_controller", [GRIPPER_JOINT], cb),
        ]
        self.get_logger().info(
            "Bridge up: /joint_states <- measured driver state; "
            "MoveIt goals on /arm_controller and /gripper_controller -> /joint_ctrl_single"
        )

    # ------------------------------------------------------------------ state
    def _on_driver_state(self, msg: JointState) -> None:
        mapped = driver_to_urdf(msg.name, msg.position, self._gripper_scale)
        if not mapped:
            return
        with self._lock:
            if not self._watch.update(stamp_to_ns(msg.header.stamp), time.monotonic()):
                return  # cached repeat of data we already have: not new information
            self._measured.update(mapped)

        names = ARM_JOINTS + [GRIPPER_JOINT, MIRROR_GRIPPER_JOINT]
        if any(n not in mapped for n in names):
            return
        out = JointState()
        out.header.stamp = msg.header.stamp  # SDK receive time (wall clock)
        out.name = names
        out.position = [mapped[n] for n in names]
        vel = list(msg.velocity[:6]) if len(msg.velocity) >= 6 else []
        out.velocity = vel + [0.0, 0.0] if vel else []
        eff = list(msg.effort[:7]) if len(msg.effort) >= 7 else []
        out.effort = eff[:6] + [eff[6], 0.0] if eff else []
        self._js_pub.publish(out)

    def _state_fresh(self) -> bool:
        return self._watch.data_age(time.monotonic()) < self._state_timeout

    # -------------------------------------------------------------- streaming
    def _stream(self) -> None:
        now = time.monotonic()
        with self._lock:
            for g in self._active.values():
                pos = sample_trajectory(g.times, g.positions, now - g.start_mono)
                self._target.update(zip(g.joints, pos.tolist()))
            if not self._streaming:
                return
            target = dict(self._target)
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        # The driver reads joint1..6 by name and the gripper from position[6]
        # (opening = position[6] * gripper_val_mutiple), so joint7 goes last.
        cmd.name = ARM_JOINTS + [GRIPPER_JOINT]
        cmd.position = [target[j] for j in ARM_JOINTS] + [target[GRIPPER_JOINT]]
        cmd.effort = [0.0] * 6 + [self._gripper_effort]
        self._cmd_pub.publish(cmd)

    # ----------------------------------------------------------------- freeze
    def _on_freeze(self, _req, resp):
        with self._lock:
            have_state = all(j in self._measured for j in ARM_JOINTS + [GRIPPER_JOINT])
            stopped = sorted(self._active)
            self._frozen = True
            # Executing goals see _frozen and abort; clearing _active stops the
            # stream from advancing them on its next tick.
            self._active.clear()
            if have_state:
                for j in ARM_JOINTS + [GRIPPER_JOINT]:
                    self._target[j] = self._measured[j]
                self._streaming = True
        self._frozen_pub.publish(Bool(data=True))
        resp.success = True
        resp.message = ("FROZEN: holding measured position" if have_state
                        else "FROZEN: no arm state yet, nothing commanded")
        resp.message += f"; stopped {stopped}" if stopped else "; nothing was executing"
        self.get_logger().warn(resp.message)
        return resp

    def _on_unfreeze(self, _req, resp):
        with self._lock:
            was = self._frozen
            self._frozen = False
        self._frozen_pub.publish(Bool(data=False))
        resp.success = True
        resp.message = "unfrozen: accepting goals" if was else "was not frozen"
        self.get_logger().info(resp.message)
        return resp

    # ----------------------------------------------------------------- action
    def _make_server(self, controller: str, joints: List[str], cb) -> ActionServer:
        return ActionServer(
            self,
            FollowJointTrajectory,
            f"{controller}/follow_joint_trajectory",
            execute_callback=lambda gh: self._execute(gh, controller, joints),
            goal_callback=lambda req: self._accept(req, controller, joints),
            cancel_callback=lambda _gh: CancelResponse.ACCEPT,
            callback_group=cb,
        )

    def _accept(self, req: FollowJointTrajectory.Goal, controller: str, joints: List[str]) -> GoalResponse:
        traj = req.trajectory
        log = self.get_logger()
        if not traj.points:
            log.error(f"[{controller}] rejected: empty trajectory")
            return GoalResponse.REJECT
        if sorted(traj.joint_names) != sorted(joints):
            log.error(f"[{controller}] rejected: joints {list(traj.joint_names)} != {joints}")
            return GoalResponse.REJECT
        with self._lock:
            fresh = self._state_fresh()
            measured = dict(self._measured)
            busy = controller in self._active
            frozen = self._frozen
        if frozen:
            log.error(f"[{controller}] rejected: arm is FROZEN (call /piper/unfreeze first)")
            return GoalResponse.REJECT
        if not fresh or any(j not in measured for j in ARM_JOINTS + [GRIPPER_JOINT]):
            log.error(f"[{controller}] rejected: no new joint feedback from the Piper driver in "
                      f"{self._state_timeout}s (arm off, or CAN feedback stopped?)")
            return GoalResponse.REJECT
        if busy:
            log.error(f"[{controller}] rejected: a trajectory is already executing on this controller")
            return GoalResponse.REJECT
        first = dict(zip(traj.joint_names, traj.points[0].positions))
        err = max_abs_error(first, measured, joints)
        if err > self._start_tol:
            log.error(
                f"[{controller}] rejected: trajectory starts {err:.3f} rad from the measured state "
                f"(limit {self._start_tol}). Re-plan from the current state."
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, gh, controller: str, joints: List[str]):
        traj = gh.request.trajectory
        order = [list(traj.joint_names).index(j) for j in joints]
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points])
        positions = np.array([[p.positions[i] for i in order] for p in traj.points], dtype=float)
        result = FollowJointTrajectory.Result()

        stamped = JointTrajectory()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = controller
        stamped.joint_names = list(traj.joint_names)
        stamped.points = list(traj.points)
        self._traj_pub.publish(stamped)

        with self._lock:
            if self._frozen:  # froze between accept and execute
                gh.abort()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "aborted: arm is FROZEN"
                return result
            for j in ARM_JOINTS + [GRIPPER_JOINT]:
                # Hold every joint not in this goal at its current setpoint, or at the
                # measured position if nothing has been commanded yet.
                self._target.setdefault(j, self._measured[j])
            goal = _ActiveGoal(controller, joints, times, positions, time.monotonic())
            self._active[controller] = goal
            self._streaming = True
        self.get_logger().info(f"[{controller}] executing {len(traj.points)} points over {goal.duration:.2f}s")

        def finish(state: str, code: int, msg: str, hold_measured: bool = False):
            with self._lock:
                self._active.pop(controller, None)
                if hold_measured:
                    for j in joints:
                        if j in self._measured:
                            self._target[j] = self._measured[j]
            result.error_code = code
            result.error_string = msg
            getattr(gh, state)()
            # Separate call sites: rclpy refuses one call site logging at two severities.
            if state == "succeed":
                self.get_logger().info(f"[{controller}] {msg}")
            else:
                self.get_logger().error(f"[{controller}] {msg}")
            return result

        fb = FollowJointTrajectory.Feedback()
        fb.joint_names = joints
        deadline = goal.start_mono + goal.duration + self._settle_timeout
        while True:
            time.sleep(1.0 / self._rate_hz)
            now = time.monotonic()
            if gh.is_cancel_requested:
                return finish("canceled", FollowJointTrajectory.Result.SUCCESSFUL, "canceled; holding measured position",
                              hold_measured=True)
            with self._lock:
                fresh = self._state_fresh()
                measured = dict(self._measured)
                target = {j: self._target[j] for j in joints}
                frozen = self._frozen
            if frozen:
                # The freeze already set the hold target; don't overwrite it.
                return finish("abort", FollowJointTrajectory.Result.INVALID_GOAL, "aborted: arm FROZEN")
            if not fresh:
                return finish("abort", FollowJointTrajectory.Result.INVALID_GOAL,
                              "aborted: Piper joint feedback stopped (stamps frozen) mid-trajectory",
                              hold_measured=True)
            fb.desired.positions = [target[j] for j in joints]
            fb.actual.positions = [measured.get(j, float("nan")) for j in joints]
            gh.publish_feedback(fb)
            if now < goal.start_mono + goal.duration:
                continue
            final = dict(zip(joints, positions[-1].tolist()))
            err = max_abs_error(final, measured, joints)
            if controller == "gripper_controller":
                # A gripper closing on an object never reaches its target; finishing
                # the trajectory is success.
                if err < 2e-3 or now >= deadline:
                    return finish("succeed", FollowJointTrajectory.Result.SUCCESSFUL,
                                  f"done (final opening error {err * 1000:.1f} mm)")
            elif err <= self._goal_tol:
                return finish("succeed", FollowJointTrajectory.Result.SUCCESSFUL, f"done (final error {err:.4f} rad)")
            elif now >= deadline:
                return finish("abort", FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                              f"aborted: arm settled {err:.3f} rad from the goal (tolerance {self._goal_tol})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PiperMoveItBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
