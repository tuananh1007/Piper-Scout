#!/usr/bin/env python3
"""Hardware-free check of the grasp reach handoff (P0.4.11, INSTALL.md 10.9).

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_base:=true fake_base:=true bringup_servo:=true bringup_pipeline:=false
    ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py execute:=true
    ros2 run stem_grasp pipeline_node --ros-args -p reach_executor:=whole_body_mpc
    ros2 run scout_piper_bringup grasp_chain_check.py 0.95 0.15     # stem x y in odom

Publishes a synthetic stem point cloud (the robot-facing half of a 4 mm
vertical cylinder, z 0.25-0.60 m in odom) on /stem_grasp/filtered_cloud,
starts servo and enables piper_servo_bridge, and waits for stem_grasp to go
REACHING -> SERVOING. Then checks: the TCP from TF is at the pre-grasp
position the pipeline sent the MPC, the MPC is idle and sends no more joint
commands, the base stopped and servo never halted. Reports the approach-axis
error. Runs only against the fake drivers, an MPC with execute:=true and a
pipeline with reach_executor:=whole_body_mpc. Leaves the bridge disabled.
"""

import argparse
import json
import sys
import time


def stem_cloud(x: float, y: float, rng, n: int = 3000, radius: float = 0.004):
    import numpy as np  # noqa: PLC0415
    z = rng.uniform(0.25, 0.60, n)
    a = rng.uniform(-np.pi / 2, np.pi / 2, n)          # the half facing the robot (-x)
    pts = np.stack([x - radius * np.cos(a), y + radius * np.sin(a), z], 1)
    return (pts + rng.normal(0, 0.0005, pts.shape)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", type=float, nargs=2, help="stem x y in odom [m]")
    ap.add_argument("--tolerance", type=float, default=0.02, help="TCP error allowed [m]")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    import numpy as np  # noqa: PLC0415
    import rclpy  # noqa: PLC0415
    import tf2_ros  # noqa: PLC0415
    from control_msgs.msg import JointJog  # noqa: PLC0415
    from geometry_msgs.msg import PoseStamped  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415
    from sensor_msgs.msg import PointCloud2  # noqa: PLC0415
    from sensor_msgs_py import point_cloud2  # noqa: PLC0415
    from std_msgs.msg import Header, Int8, String  # noqa: PLC0415
    from std_srvs.srv import SetBool, Trigger  # noqa: PLC0415

    rclpy.init()
    n = Node("grasp_chain_check")
    states, mpc, goals, jogs, servo, odom = [], [], [], [], [], {}

    def on_state(m):
        if not states or states[-1][1] != m.data:
            states.append((time.time(), m.data))

    n.create_subscription(String, "/stem_grasp/pipeline_state", on_state, 10)
    n.create_subscription(String, "/whole_body_mpc/status",
                          lambda m: mpc.append((time.time(), json.loads(m.data))), 50)
    n.create_subscription(PoseStamped, "/whole_body_mpc/goal_pose", goals.append, 10)
    n.create_subscription(JointJog, "/servo_node/delta_joint_cmds", lambda m: jogs.append(time.time()), 50)
    n.create_subscription(Int8, "/servo_node/status", lambda m: servo.append(m.data), 100)
    n.create_subscription(Odometry, "/odom", lambda m: odom.update(
        x=m.pose.pose.position.x, y=m.pose.pose.position.y,
        v=m.twist.twist.linear.x, w=m.twist.twist.angular.z), 10)
    cloud_pub = n.create_publisher(PointCloud2, "/stem_grasp/filtered_cloud", 5)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, n)
    pts = stem_cloud(a.stem[0], a.stem[1], np.random.default_rng(0)).tolist()

    def publish_cloud():
        h = Header()
        h.frame_id = "odom"
        h.stamp = n.get_clock().now().to_msg()
        cloud_pub.publish(point_cloud2.create_cloud_xyz32(h, pts))

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
    executor = param("/stem_grasp_pipeline", "reach_executor")
    if executor is None or executor.string_value != "whole_body_mpc":
        print("REFUSED: /stem_grasp_pipeline is not running with reach_executor:=whole_body_mpc", flush=True)
        return 2
    offset = param("/whole_body_mpc", "tcp_offset_m").double_value

    n.create_timer(0.2, publish_cloud)
    call(Trigger, "/servo_node/start_servo", Trigger.Request())
    call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=True))
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        spin(0.2)
        if states and states[-1][1] in ("SERVOING", "ABORTED"):
            break
    spin(2.0)
    call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=False))

    seq = [s for _, s in states]
    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    t_reach = next((t for t, s in states if s == "REACHING"), None)
    t_servo = next((t for t, s in states if s == "SERVOING"), None)
    check("stem_grasp went REACHING -> SERVOING", seq[-2:] == ["REACHING", "SERVOING"],
          " -> ".join(seq) + (f" (reach {t_servo - t_reach:.1f} s)" if t_reach and t_servo else ""))
    if goals and t_servo is not None:
        g = goals[-1]
        goal_p = np.array([g.pose.position.x, g.pose.position.y, g.pose.position.z])
        q = g.pose.orientation
        goal_z = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2]
        tr = tf_buffer.lookup_transform("odom", "piper_link6", rclpy.time.Time())
        r = tr.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        t = tr.transform.translation
        tcp = np.array([t.x, t.y, t.z]) + offset * R[:, 2]
        err = float(np.linalg.norm(tcp - goal_p))
        angle = float(np.degrees(np.arccos(np.clip(R[:, 2] @ goal_z, -1.0, 1.0))))
        check("TCP at the pre-grasp position (TF)", err < a.tolerance,
              f"{100 * err:.2f} cm from {np.round(goal_p, 3)}")
        print(f"info: approach-axis error {angle:.1f} deg (MPC approach_tolerance_deg gates 'reached')",
              flush=True)
    else:
        check("TCP at the pre-grasp position (TF)", False, "no goal sent or no handoff")
    check("MPC idle after the handoff", mpc and mpc[-1][1]["mode"] == "idle",
          f"last mode {mpc[-1][1]['mode'] if mpc else None}")
    late = [t for t in jogs if t_servo is not None and t > t_servo + 0.5]
    check("no MPC joint commands after the handoff", t_servo is not None and not late,
          f"{len(late)} JointJog messages")
    check("base stopped", abs(odom.get("v", 1.0)) < 1e-3 and abs(odom.get("w", 1.0)) < 1e-3,
          f"at ({odom.get('x', 0.0):.3f}, {odom.get('y', 0.0):.3f})")
    check("no servo halt", not [s for s in servo if s in (2, 4)], f"servo status codes {sorted(set(servo))}")
    ok = all(results)
    print("GRASP CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
