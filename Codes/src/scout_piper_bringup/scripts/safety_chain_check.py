#!/usr/bin/env python3
"""Hardware-free check of the safety supervisor (safety_supervisor.py).

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_base:=true fake_base:=true bringup_servo:=true operator_heartbeat:=true \
        bringup_rviz:=false bringup_pipeline:=false
    ros2 run stem_grasp pipeline_node            # optional: checks that a trip aborts it
    ros2 run scout_piper_bringup safety_chain_check.py

Plays the operator page's heartbeat (/operator/heartbeat, 4 Hz) and checks,
on the fake arm and base only:

  1. a base command reaches the base through the gate (/cmd_vel -> /cmd_vel_safe);
  2. a sender that goes quiet without a zero: the gate stops the base
     (the fake base, like scout_ros2, has no command timeout of its own);
  3. the heartbeat stopping while the base drives: trip, the base stops and
     stays stopped although /cmd_vel keeps coming, the pipeline is aborted,
     the servo bridge refuses ~/enable;
  4. ~/reset with the heartbeat back clears the trip;
  5. the heartbeat stopping while the arm jogs: trip, the bridge disabled, the
     arm stops although the jog keeps coming.

Leaves the supervisor reset and the bridge disabled. Exit code 0 when all pass.
"""

import json
import sys
import time


def main() -> int:
    import rclpy  # noqa: PLC0415
    from control_msgs.msg import JointJog  # noqa: PLC0415
    from geometry_msgs.msg import Twist  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415
    from std_msgs.msg import Empty, String  # noqa: PLC0415
    from std_srvs.srv import SetBool, Trigger  # noqa: PLC0415

    rclpy.init()
    n = Node("safety_chain_check")
    safety, odom, fb, pipe, gated = [], {}, {}, [], []
    n.create_subscription(String, "/safety/state", lambda m: safety.append((time.time(), json.loads(m.data))), 20)
    n.create_subscription(Odometry, "/odom", lambda m: odom.update(v=m.twist.twist.linear.x, t=time.time()), 20)
    n.create_subscription(JointState, "/joint_states_single", lambda m: fb.update(zip(m.name, m.position)), 20)
    n.create_subscription(String, "/stem_grasp/pipeline_state", lambda m: pipe.append(m.data), 20)
    n.create_subscription(Twist, "/cmd_vel_safe", gated.append, 50)
    pub_cmd = n.create_publisher(Twist, "/cmd_vel", 10)
    pub_hb = n.create_publisher(Empty, "/operator/heartbeat", 10)
    pub_jog = n.create_publisher(JointJog, "/servo_node/delta_joint_cmds", 10)
    beat = [True]
    n.create_timer(0.25, lambda: beat[0] and pub_hb.publish(Empty()))

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.01)

    def call(srv_type, name, request, timeout=15.0):
        client = n.create_client(srv_type, name)
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        return future.result()

    def drive(sec, v=0.2):
        end = time.time() + sec
        while time.time() < end:
            m = Twist()
            m.linear.x = v
            pub_cmd.publish(m)
            spin(0.1)

    def jog(sec, vel=0.3):
        end = time.time() + sec
        while time.time() < end:
            m = JointJog()
            m.header.stamp = n.get_clock().now().to_msg()
            m.joint_names, m.velocities = ["piper_joint1"], [vel]
            pub_jog.publish(m)
            spin(0.02)

    def last():
        return safety[-1][1] if safety else {}

    def first_trip_after(t0):
        return next(((t, s) for t, s in safety if t >= t0 and not s.get("ok", True)), (None, None))

    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    # fake drivers only: only they declare these parameters
    for node_name, param in (("/piper_ctrl_single_node", "initial_gripper"), ("/scout_base_node", "slip_k_v")):
        reply = call(GetParameters, f"{node_name}/get_parameters", GetParameters.Request(names=[param]))
        if reply is None or not reply.values or reply.values[0].type == 0:
            print(f"REFUSED: {node_name} is not the fake driver; launch with fake_arm:=true fake_base:=true",
                  flush=True)
            return 2

    end = time.time() + 15.0
    while time.time() < end and not (safety and odom and fb):
        spin(0.1)
    st = last()
    check("supervisor reports, heartbeat required", st.get("ok") and st.get("require_heartbeat"),
          f"{json.dumps(st)[:160]}" + ("" if st.get("require_heartbeat") else "; launch with operator_heartbeat:=true"))
    if not st:
        return 1

    # 1. through the gate
    n0 = len(gated)
    drive(2.0)
    check("base command reaches the base through the gate", odom.get("v", 0.0) > 0.15 and len(gated) - n0 >= 15,
          f"odom {odom.get('v', 0.0):.2f} m/s, {len(gated) - n0} gated commands")

    # 2. sender goes quiet without a zero
    to0 = last().get("cmd_timeouts", 0)
    spin(1.2)
    check("command timeout stops the base", abs(odom.get("v", 1.0)) < 0.02 and last().get("cmd_timeouts", 0) > to0,
          f"odom {odom.get('v', 0.0):.3f} m/s 1.2 s after the last command, "
          f"{last().get('cmd_timeouts', 0) - to0} timeout zero(s)")

    # 3. heartbeat lost while the base drives
    drive(1.0)
    beat[0] = False
    t_lost = time.time()
    drive(3.0)                                  # the sender keeps commanding
    t_trip, s_trip = first_trip_after(t_lost)
    check("heartbeat lost while driving: trip", t_trip is not None and "heartbeat" in s_trip.get("tripped", ""),
          f"after {t_trip - t_lost:.2f} s: {s_trip.get('tripped')}" if t_trip else f"no trip; {json.dumps(last())[:160]}")
    check("base stopped and kept stopped despite /cmd_vel", abs(odom.get("v", 1.0)) < 0.02,
          f"odom {odom.get('v', 0.0):.3f} m/s with 0.2 m/s still commanded")
    if pipe:
        spin(0.5)
        check("pipeline aborted", pipe[-1] == "ABORTED", " -> ".join(pipe[-4:]))
    else:
        print("info no /stem_grasp/pipeline_state (pipeline not running): abort not checked", flush=True)
    r = call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=True))
    check("servo bridge refuses ~/enable while tripped", r is not None and not r.success,
          r.message if r is not None else "no /piper_servo_bridge/enable")

    # 4. reset with the heartbeat back
    beat[0] = True
    spin(1.0)
    r = call(Trigger, "/safety_supervisor/reset", Trigger.Request())
    spin(0.5)
    check("reset with the heartbeat back", r is not None and r.success and last().get("ok"),
          r.message if r is not None else "no /safety_supervisor/reset")

    # 5. heartbeat lost while the arm jogs
    call(Trigger, "/servo_node/start_servo", Trigger.Request())
    spin(1.0)
    r = call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=True))
    q0 = fb.get("joint1", 0.0)
    jog(1.5)
    moved = fb.get("joint1", 0.0) - q0
    beat[0] = False
    t_lost = time.time()
    jog(2.0)
    t_trip, s_trip = first_trip_after(t_lost)
    q1 = fb.get("joint1", 0.0)
    jog(1.0)                                    # still jogging after the trip
    held = abs(fb.get("joint1", 0.0) - q1)
    check("arm jogs with the heartbeat", r is not None and r.success and moved > 0.2,
          f"joint1 {moved:+.3f} rad in 1.5 s")
    check("heartbeat lost while the arm moves: trip", t_trip is not None and "heartbeat" in s_trip.get("tripped", ""),
          f"after {t_trip - t_lost:.2f} s: {s_trip.get('tripped')}" if t_trip else "no trip")
    check("arm holds although the jog continues", held < 0.005, f"joint1 moved {held:.4f} rad in the last 1 s")

    beat[0] = True
    spin(1.0)
    r = call(Trigger, "/safety_supervisor/reset", Trigger.Request())
    check("reset again", r is not None and r.success, r.message if r is not None else "no reset service")
    ok = all(results)
    print("SAFETY CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
