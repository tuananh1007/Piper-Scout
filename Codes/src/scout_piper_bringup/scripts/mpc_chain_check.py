#!/usr/bin/env python3
"""Hardware-free check of the whole-body MPC in execute mode (INSTALL.md 10.13).

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_base:=true fake_base:=true bringup_servo:=true bringup_pipeline:=false
    ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py execute:=true
    ros2 run scout_piper_bringup mpc_chain_check.py 1.0 0.2 0.45     # goal x y z in odom

Runs only against fake_piper_driver.py and fake_scout_base.py. Starts servo,
enables piper_servo_bridge, sends the goal (repeated until the MPC reports, so
DDS discovery cannot swallow it) and waits until the MPC has reported
``reached`` for 2 s. Then checks, independently of the MPC: the TCP from TF is
within ``--tolerance`` of the goal, the base has stopped, the last /cmd_vel is
zero and servo never halted for a singularity or collision. Leaves the bridge
disabled. Exit code 0 when all pass.
"""

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("goal", type=float, nargs=3, help="goal x y z in odom [m]")
    ap.add_argument("--tolerance", type=float, default=0.02, help="TCP error allowed [m]")
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds to reach the goal")
    a = ap.parse_args()

    import numpy as np  # noqa: PLC0415
    import rclpy  # noqa: PLC0415
    import tf2_ros  # noqa: PLC0415
    from geometry_msgs.msg import PointStamped, Twist  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from std_msgs.msg import Int8, String  # noqa: PLC0415
    from std_srvs.srv import SetBool, Trigger  # noqa: PLC0415

    rclpy.init()
    n = Node("mpc_chain_check")
    status, odom, servo, cmd = [], {}, [], []
    n.create_subscription(String, "/whole_body_mpc/status",
                          lambda m: status.append((time.time(), json.loads(m.data))), 50)
    n.create_subscription(Odometry, "/odom", lambda m: odom.update(
        x=m.pose.pose.position.x, y=m.pose.pose.position.y,
        v=m.twist.twist.linear.x, w=m.twist.twist.angular.z), 10)
    n.create_subscription(Int8, "/servo_node/status", lambda m: servo.append(m.data), 100)
    n.create_subscription(Twist, "/cmd_vel", lambda m: cmd.append((m.linear.x, m.angular.z)), 50)
    goal_pub = n.create_publisher(PointStamped, "/whole_body_mpc/goal", 1)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, n)

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.01)

    def call(srv_type, name, request, timeout=15.0):
        client = n.create_client(srv_type, name)
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {name} not available")
        future = client.call_async(request)
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        return future.result()

    def param(node, name):
        reply = call(GetParameters, f"{node}/get_parameters", GetParameters.Request(names=[name]))
        return reply.values[0] if reply.values and reply.values[0].type != 0 else None

    def tcp_world(offset):
        tr = tf_buffer.lookup_transform("odom", "piper_link6", rclpy.time.Time())
        t, q = tr.transform.translation, tr.transform.rotation
        z_axis = np.array([2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x),
                           1 - 2 * (q.x * q.x + q.y * q.y)])
        return np.array([t.x, t.y, t.z]) + offset * z_axis

    for node, name in (("/piper_ctrl_single_node", "initial_gripper"),
                       ("/scout_base_node", "time_constant_s")):
        if param(node, name) is None:
            print(f"REFUSED: {node} is not the fake driver; launch with fake_arm:=true fake_base:=true",
                  flush=True)
            return 2
    execute = param("/whole_body_mpc", "execute")
    if execute is None or not execute.bool_value:
        print("REFUSED: /whole_body_mpc is not running with execute:=true", flush=True)
        return 2
    offset = param("/whole_body_mpc", "tcp_offset_m").double_value

    end = time.time() + 15.0
    p0 = None
    while p0 is None and time.time() < end:
        spin(0.2)
        try:
            p0 = tcp_world(offset)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass
    if p0 is None:
        print("FAIL no TF odom -> piper_link6", flush=True)
        return 1
    call(Trigger, "/servo_node/start_servo", Trigger.Request())
    call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=True))
    spin(1.0)
    print(f"start: TCP {np.round(p0, 3)}, base ({odom.get('x', 0):.3f}, {odom.get('y', 0):.3f}), "
          f"goal {a.goal}", flush=True)

    goal = PointStamped()
    goal.header.frame_id = "odom"
    goal.point.x, goal.point.y, goal.point.z = a.goal
    end = time.time() + 15.0
    while goal_pub.get_subscription_count() == 0 and time.time() < end:
        spin(0.1)
    t0, seen = time.time(), len(status)
    while len(status) == seen and time.time() - t0 < 10.0:     # republish until the MPC reports
        goal_pub.publish(goal)
        spin(0.3)
    reached_since = None
    while time.time() - t0 < a.timeout:
        spin(0.2)
        if status and status[-1][1]["mode"] == "reached":
            reached_since = reached_since or time.time()
            if time.time() - reached_since > 2.0:
                break
        else:
            reached_since = None
    spin(1.0)
    call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=False))

    run = [s for t, s in status if t >= t0]
    modes = [s["mode"] for s in run]
    solve = [s["solve_ms"] for s in run if s["mode"] != "reached"]
    err = float(np.linalg.norm(tcp_world(offset) - np.array(a.goal)))
    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    check("MPC reports reached", modes and modes[-1] == "reached",
          f"after {(reached_since or time.time()) - t0:.1f} s; modes {sorted(set(modes))}")
    check("TCP at the goal (TF)", err < a.tolerance, f"{100 * err:.2f} cm")
    check("base stopped", abs(odom["v"]) < 1e-3 and abs(odom["w"]) < 1e-3,
          f"at ({odom['x']:.3f}, {odom['y']:.3f}), v={odom['v']:.4f} w={odom['w']:.4f}")
    check("last /cmd_vel is zero", cmd and cmd[-1] == (0.0, 0.0), f"{cmd[-1] if cmd else None}")
    check("no servo halt", not [s for s in servo if s in (2, 4)], f"servo status codes {sorted(set(servo))}")
    print(f"info: safety reasons {sorted({s['safety'] for s in run})}; MPPI solve median "
          f"{np.median(solve) if solve else float('nan'):.0f} ms, max {max(solve) if solve else float('nan'):.0f} ms",
          flush=True)
    ok = all(results)
    print("MPC CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
