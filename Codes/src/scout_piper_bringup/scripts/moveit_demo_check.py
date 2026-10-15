#!/usr/bin/env python3
"""Check of the upstream MoveIt 2 demo for the Piper (P0.3.9), headless.

    ros2 launch piper_with_gripper_moveit demo.launch.py use_rviz:=false   # piper_ros, mock hardware
    ros2 run scout_piper_bringup moveit_demo_check.py

Through move_group's MoveGroup action (/move_action) for the group "arm":
plans home -> a canned joint pose without executing, executes it on the
demo's mock hardware (ros2_control GenericSystem) and checks /joint_states
arrives there; asks /compute_fk for the flange pose of a second
configuration and plans and executes to that pose (IK through the
kinematics plugin); returns to the SRDF state "zero". Planning only on the
real arm: the demo's controllers are mock components, and full_system.launch.py
keeps the real driver's command input away from MoveIt (INSTALL.md 10.4b).
Exit code 0 when all pass.
"""

import sys
import time

JOINTS = [f"joint{i}" for i in range(1, 7)]
CANNED = [0.2, 0.8, -0.6, 0.0, 0.4, 0.0]
SECOND = [-0.3, 1.0, -0.9, 0.2, 0.5, 0.3]


def main() -> int:
    import rclpy  # noqa: PLC0415
    from moveit_msgs.action import MoveGroup  # noqa: PLC0415
    from moveit_msgs.msg import (Constraints, JointConstraint, MoveItErrorCodes, OrientationConstraint,  # noqa: PLC0415
                                 PositionConstraint, RobotState)
    from moveit_msgs.srv import GetPositionFK  # noqa: PLC0415
    from rclpy.action import ActionClient  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415
    from shape_msgs.msg import SolidPrimitive  # noqa: PLC0415

    rclpy.init()
    n = Node("moveit_demo_check")
    js = {}
    n.create_subscription(JointState, "/joint_states", lambda m: js.update(zip(m.name, m.position)), 20)
    act = ActionClient(n, MoveGroup, "/move_action")
    fk = n.create_client(GetPositionFK, "/compute_fk")
    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    def wait(fut, timeout):
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)

    if not act.wait_for_server(timeout_sec=30.0):
        check("move_group up", False, "no /move_action within 30 s (demo.launch.py running?)")
        return 1
    spin(1.0)

    def request(constraints, plan_only):
        g = MoveGroup.Goal()
        r = g.request
        r.group_name = "arm"
        r.num_planning_attempts = 3
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = 0.3
        r.max_acceleration_scaling_factor = 0.3
        r.start_state.is_diff = True
        r.goal_constraints = [constraints]
        g.planning_options.plan_only = plan_only
        g.planning_options.replan = False
        handle = wait(act.send_goal_async(g), 10.0)
        if handle is None or not handle.accepted:
            return None
        res = wait(handle.get_result_async(), 60.0)
        return None if res is None else res.result

    def joint_goal(q):
        c = Constraints()
        for name, v in zip(JOINTS, q):
            jc = JointConstraint()
            jc.joint_name, jc.position = name, float(v)
            jc.tolerance_above = jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def describe(res):
        if res is None:
            return "no result"
        pts = res.planned_trajectory.joint_trajectory.points
        dur = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9 if pts else 0.0
        return f"error code {res.error_code.val}, {len(pts)} points, {dur:.2f} s, planning {res.planning_time:.3f} s"

    def at(q, tol=0.01):
        return all(abs(js.get(j, 1e9) - v) < tol for j, v in zip(JOINTS, q))

    ok_code = MoveItErrorCodes.SUCCESS
    start = [js.get(j) for j in JOINTS]
    res = request(joint_goal(CANNED), plan_only=True)
    pts = res.planned_trajectory.joint_trajectory.points if res is not None else []
    end_ok = bool(pts) and all(abs(a - b) < 0.002 for a, b in zip(pts[-1].positions, CANNED))
    check("plan home -> canned joint pose (no execution)", res is not None and res.error_code.val == ok_code and end_ok,
          f"from {[round(v, 2) if v is not None else None for v in start]}: {describe(res)}")
    spin(1.0)
    check("planning only moved nothing", at(start, 1e-4) if None not in start else False, "joint states unchanged")

    res = request(joint_goal(CANNED), plan_only=False)
    spin(1.0)
    check("execute on the mock hardware", res is not None and res.error_code.val == ok_code and at(CANNED),
          f"{describe(res)}; joints {[round(js.get(j, float('nan')), 3) for j in JOINTS]}")

    req = GetPositionFK.Request()
    req.header.frame_id = "base_link"
    req.fk_link_names = ["link6"]
    req.robot_state = RobotState()
    req.robot_state.joint_state.name = JOINTS
    req.robot_state.joint_state.position = [float(v) for v in SECOND]
    fk_res = wait(fk.call_async(req), 10.0) if fk.wait_for_service(timeout_sec=10.0) else None
    ok_fk = fk_res is not None and fk_res.error_code.val == ok_code and len(fk_res.pose_stamped) == 1
    check("forward kinematics from move_group", ok_fk,
          str(fk_res.pose_stamped[0].pose.position).replace("geometry_msgs.msg.", "") if ok_fk else "no answer")
    if ok_fk:
        ps = fk_res.pose_stamped[0]
        c = Constraints()
        pc = PositionConstraint()
        pc.header, pc.link_name, pc.weight = ps.header, "link6", 1.0
        box = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.002])
        pc.constraint_region.primitives = [box]
        pc.constraint_region.primitive_poses = [ps.pose]
        oc = OrientationConstraint()
        oc.header, oc.link_name, oc.orientation, oc.weight = ps.header, "link6", ps.pose.orientation, 1.0
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.01
        c.position_constraints, c.orientation_constraints = [pc], [oc]
        res = request(c, plan_only=False)
        spin(1.0)
        chk = GetPositionFK.Request()
        chk.header.frame_id, chk.fk_link_names = "base_link", ["link6"]
        chk.robot_state.joint_state.name = JOINTS
        chk.robot_state.joint_state.position = [float(js.get(j, 0.0)) for j in JOINTS]
        now = wait(fk.call_async(chk), 10.0)
        err = float("nan")
        if now is not None and now.pose_stamped:
            a, b = now.pose_stamped[0].pose.position, ps.pose.position
            err = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
        check("plan and execute to a flange pose (IK)", res is not None and res.error_code.val == ok_code and err < 0.005,
              f"{describe(res)}; flange {1000 * err:.1f} mm from the goal")

    res = request(joint_goal([0.0] * 6), plan_only=False)
    spin(1.0)
    check("back to the SRDF state 'zero'", res is not None and res.error_code.val == ok_code and at([0.0] * 6),
          describe(res))
    ok = all(results)
    print("MOVEIT DEMO", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
