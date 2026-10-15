"""/piper/zero (std_srvs/Trigger): move the arm to its zero pose, slowly, through MoveIt.

Plans to all six arm joints = 0 (the SRDF's "zero" state) from the measured
state, with collision checking against the robot itself and the planning
scene (scene_objects boxes, octomap), at reduced speed. The service returns
when the motion finishes or fails. The gripper is not moved.

Refuses while the arm is frozen (/piper/frozen); /piper/freeze stops a zero
motion like any other.
"""

from __future__ import annotations

import sys

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from piper_quickstart.trajectory import ARM_JOINTS

_CODES = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}


def zero_goal(velocity_scaling: float, planning_time: float) -> MoveGroup.Goal:
    goal = MoveGroup.Goal()
    req = goal.request
    req.group_name = "arm"
    req.num_planning_attempts = 3
    req.allowed_planning_time = planning_time
    req.max_velocity_scaling_factor = velocity_scaling
    req.max_acceleration_scaling_factor = velocity_scaling
    req.start_state.is_diff = True  # plan from the current (measured) state
    req.goal_constraints = [Constraints(joint_constraints=[
        JointConstraint(joint_name=j, position=0.0, tolerance_above=0.005, tolerance_below=0.005, weight=1.0)
        for j in ARM_JOINTS])]
    goal.planning_options.plan_only = False
    return goal


class ArmSafety(Node):
    def __init__(self) -> None:
        super().__init__("arm_safety")
        self.declare_parameter("zero_velocity_scaling", 0.2)
        self.declare_parameter("planning_time_s", 5.0)
        self._vel = float(self.get_parameter("zero_velocity_scaling").value)
        self._ptime = float(self.get_parameter("planning_time_s").value)
        cb = ReentrantCallbackGroup()
        self._frozen = False
        self.create_subscription(
            Bool, "/piper/frozen", lambda m: setattr(self, "_frozen", m.data),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=cb)
        self._move = ActionClient(self, MoveGroup, "/move_action", callback_group=cb)
        self.create_service(Trigger, "/piper/zero", self._on_zero, callback_group=cb)
        self._busy = False

    async def _on_zero(self, _req, resp):
        log = self.get_logger()
        resp.success = False
        if self._frozen:
            resp.message = "refused: arm is FROZEN (call /piper/unfreeze first)"
        elif self._busy:
            resp.message = "refused: already moving to zero"
        elif not self._move.wait_for_server(timeout_sec=2.0):
            resp.message = "refused: move_group is not running"
        else:
            self._busy = True
            try:
                log.info(f"zeroing arm at {self._vel:.0%} speed")
                gh = await self._move.send_goal_async(zero_goal(self._vel, self._ptime))
                if not gh.accepted:
                    resp.message = "move_group rejected the zero goal"
                else:
                    code = (await gh.get_result_async()).result.error_code.val
                    resp.success = code == MoveItErrorCodes.SUCCESS
                    resp.message = "at zero" if resp.success else f"zero failed: {_CODES.get(code, code)}"
            finally:
                self._busy = False
        if resp.success:
            log.info(resp.message)
        else:
            log.error(resp.message)
        return resp


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ArmSafety()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
