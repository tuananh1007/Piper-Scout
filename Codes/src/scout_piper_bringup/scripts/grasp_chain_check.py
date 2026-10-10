#!/usr/bin/env python3
"""Hardware-free check of the grasp reach and stem servo (P0.4.11-12, INSTALL.md 10.9).

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_base:=true fake_base:=true bringup_servo:=true bringup_pipeline:=false
    ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py execute:=true
    ros2 run stem_grasp pipeline_node --ros-args -p reach_executor:=whole_body_mpc
    ros2 run scout_piper_bringup grasp_chain_check.py 0.95 0.15     # stem x y in odom

With ``-p approach_enabled:=true`` on the pipeline it also checks the final
approach (P0.4.13): REACHING -> APPROACHING -> AT_GRASP, the TCP at the grasp
point and the arm holding there. Adding ``-p grasp_close_gripper:=true`` puts
an object as wide as the stem between the fake fingers (the fake driver's
``object_width_m``) and checks AT_GRASP -> GRASPING -> GRASPED with the
gripper opened for the approach and closed on the stem. ``--release`` then
calls the pipeline's ``~/release`` and checks RELEASING -> RETREATING -> IDLE:
the gripper opens and the gripper backs out along its axis.

``--push-force N`` (servo only, with effort_force_node from
bringup_force_estimate:=true): after ``--push-after`` s of servoing it applies
N newtons at the fake arm's TCP (the fake driver's ``external_force_n``) and
checks that the estimate stayed below the pipeline's max_force_n while
servoing and that the pipeline then aborts to SCANNING within a second.

Publishes a synthetic stem (a 4 mm vertical cylinder, z 0.25-0.60 m in odom):
its robot-facing half as a point cloud on /stem_grasp/filtered_cloud, and, as
a synthetic eye-in-hand camera, its mask on /stem_grasp/mask rendered from
the live TF pose of camera_color_optical_frame plus /camera/color/camera_info
(640x480, f = 380 px). Starts servo, enables piper_servo_bridge and waits for
stem_grasp to go REACHING -> SERVOING, then lets the image-based servo run
for --servo-seconds. Checks: the TCP from TF was at the pre-grasp position
the pipeline sent the MPC, the MPC is idle and sends no more joint commands,
the base stopped, the servo's image error ended small (or, with the approach,
the TCP reached the grasp point and the arm holds there), the gripper approach
axis passes through the grasp point, and servo never halted (singularity,
collision or joint limit). Runs only
against the fake drivers, an MPC with execute:=true and a pipeline with
reach_executor:=whole_body_mpc. Leaves the bridge disabled.
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


def render_mask(points_cam, K, hw=(480, 640), dilate=1):
    """Binary mask of 3-D points (camera optical frame) through a pinhole camera."""
    import numpy as np  # noqa: PLC0415
    mask = np.zeros(hw, np.uint8)
    p = points_cam[points_cam[:, 2] > 0.02]
    u = np.round(K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int)
    for du in range(-dilate, dilate + 1):
        for dv in range(-dilate, dilate + 1):
            ok = (u + du >= 0) & (u + du < hw[1]) & (v + dv >= 0) & (v + dv < hw[0])
            mask[v[ok] + dv, u[ok] + du] = 255
    return mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", type=float, nargs=2, help="stem x y in odom [m]")
    ap.add_argument("--tolerance", type=float, default=0.02, help="TCP error allowed [m]")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--servo-seconds", type=float, default=15.0, help="servo time after the handoff")
    ap.add_argument("--release", action="store_true", help="after the grasp, release and retreat")
    ap.add_argument("--push-force", type=float, default=0.0, help="N at the TCP while servoing (force abort)")
    ap.add_argument("--push-after", type=float, default=4.0, help="servo seconds before the push")
    a = ap.parse_args()

    import numpy as np  # noqa: PLC0415
    import rclpy  # noqa: PLC0415
    import tf2_ros  # noqa: PLC0415
    from control_msgs.msg import JointJog  # noqa: PLC0415
    from geometry_msgs.msg import PoseStamped, WrenchStamped  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters, SetParameters  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2  # noqa: PLC0415
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
    # the MPC's joint commands only: the stem_grasp MPPI servo (servo_controller: mppi) tags its own
    n.create_subscription(JointJog, "/servo_node/delta_joint_cmds",
                          lambda m: m.header.frame_id != "stem_grasp" and jogs.append(time.time()), 50)
    n.create_subscription(Int8, "/servo_node/status", lambda m: servo.append(m.data), 100)
    forces = []
    n.create_subscription(WrenchStamped, "/ft_sensor/raw", lambda m: forces.append((time.time(), float(np.linalg.norm(
        [m.wrench.force.x, m.wrench.force.y, m.wrench.force.z])))), 100)
    ibvs = []
    n.create_subscription(String, "/stem_grasp/servo_status",
                          lambda m: ibvs.append((time.time(), json.loads(m.data))), 50)
    n.create_subscription(Odometry, "/odom", lambda m: odom.update(
        x=m.pose.pose.position.x, y=m.pose.pose.position.y,
        v=m.twist.twist.linear.x, w=m.twist.twist.angular.z), 10)
    cloud_pub = n.create_publisher(PointCloud2, "/stem_grasp/filtered_cloud", 5)
    mask_pub = n.create_publisher(Image, "/stem_grasp/mask", 5)
    info_pub = n.create_publisher(CameraInfo, "/camera/color/camera_info", 5)
    K = np.array([[380.0, 0.0, 320.0], [0.0, 380.0, 240.0], [0.0, 0.0, 1.0]])
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, n)
    pts = stem_cloud(a.stem[0], a.stem[1], np.random.default_rng(0)).tolist()
    zz = np.linspace(0.25, 0.60, 700)                  # full stem surface for the camera
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    stem_surface = np.stack([np.repeat(a.stem[0] + 0.004 * np.cos(ang)[None], len(zz), 0).ravel(),
                             np.repeat(a.stem[1] + 0.004 * np.sin(ang)[None], len(zz), 0).ravel(),
                             np.repeat(zz[:, None], len(ang), 1).ravel(), np.ones(len(zz) * len(ang))])

    def publish_camera():
        try:
            tr = tf_buffer.lookup_transform("camera_color_optical_frame", "odom", rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        T = np.eye(4)
        r = tr.transform.rotation
        T[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        T[:3, 3] = [tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z]
        mask = render_mask((T @ stem_surface)[:3].T, K)
        stamp = n.get_clock().now().to_msg()
        img = Image(height=mask.shape[0], width=mask.shape[1], encoding="mono8", step=mask.shape[1],
                    data=mask.tobytes())
        img.header.stamp, img.header.frame_id = stamp, "camera_color_optical_frame"
        info = CameraInfo(height=480, width=640, k=K.ravel().tolist())
        info.header = img.header
        mask_pub.publish(img)
        info_pub.publish(info)

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
    approach = param("/stem_grasp_pipeline", "approach_enabled")
    approach = approach is not None and approach.bool_value
    servo_state = "APPROACHING" if approach else "SERVOING"
    grasp = param("/stem_grasp_pipeline", "grasp_close_gripper")
    grasp = approach and grasp is not None and grasp.bool_value
    final = "GRASPED" if grasp else "AT_GRASP"
    stem_width = 0.008                                   # the synthetic stem: 4 mm radius
    widths = []
    n.create_subscription(JointState, "/joint_states_single", lambda m: widths.append(
        (time.time(), dict(zip(m.name, m.position)).get("gripper"))), 50)

    def set_object_width(w):
        v = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(w))
        call(SetParameters, "/piper_ctrl_single_node/set_parameters",
             SetParameters.Request(parameters=[Parameter(name="object_width_m", value=v)]))

    if grasp:
        set_object_width(stem_width)

    def set_force(f):
        v = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=[float(x) for x in f])
        call(SetParameters, "/piper_ctrl_single_node/set_parameters",
             SetParameters.Request(parameters=[Parameter(name="external_force_n", value=v)]))

    if a.push_force and (approach or param("/effort_force_estimator", "wrench_topic") is None):
        print("REFUSED: --push-force needs servo only (no approach) and bringup_force_estimate:=true", flush=True)
        return 2

    def push_force_check() -> int:
        """Servo, push N at the TCP, expect the force abort (``--push-force``)."""
        t_servo = next((t for t, s in states if s == servo_state), None)
        spin(a.push_after)
        max_force = param("/stem_grasp_pipeline", "max_force_n").double_value
        before = [f for t, f in forces if t_servo is not None and t > t_servo]
        t_push = time.time()
        set_force([0.0, 0.0, -a.push_force])
        while time.time() - t_push < 3.0 and states[-1][1] == servo_state:
            spin(0.05)
        t_abort = next((t for t, s in states if s == "SCANNING" and t > t_push), None)
        set_force([0.0, 0.0, 0.0])
        call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=False))
        seq = [s for _, s in states]
        ok = []
        for name, cond, detail in (
                ("stem_grasp went REACHING -> " + servo_state, t_servo is not None, " -> ".join(seq)),
                ("force estimate below max_force_n while servoing", before and max(before) < max_force,
                 f"max {max(before or [float('nan')]):.2f} N over {len(before)} estimates "
                 f"(max_force_n {max_force} N)"),
                (f"{a.push_force} N at the TCP aborts to SCANNING", t_abort is not None and t_abort - t_push < 1.0,
                 f"after {t_abort - t_push:.2f} s" if t_abort else "no abort: " + " -> ".join(seq))):
            ok.append(bool(cond))
            print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)
        print("GRASP CHAIN", "OK" if all(ok) else "FAILED", flush=True)
        n.destroy_node()
        rclpy.shutdown()
        return 0 if all(ok) else 1

    def link6():
        try:
            return tf_buffer.lookup_transform("odom", "piper_link6", rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    n.create_timer(0.2, publish_cloud)
    n.create_timer(1.0 / 15.0, publish_camera)
    call(Trigger, "/servo_node/start_servo", Trigger.Request())
    call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=True))
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        spin(0.2)
        if states and states[-1][1] in (servo_state, "ABORTED"):
            break
    t_handoff = time.time()
    spin(0.5)
    pre_grasp_tcp = link6()
    at_grasp = held = released = None
    grip_final = 0.0
    if approach:
        t0 = time.time()
        while time.time() - t0 < a.timeout and states[-1][1] not in (final, "ABORTED"):
            spin(0.2)
        at_grasp = link6()
        grip_final = next((w for _, w in reversed(widths) if w is not None), 0.0)
        spin(1.0)
        held = link6()
        if a.release and states[-1][1] == final:
            t_release = time.time()
            reply = call(Trigger, "/stem_grasp_pipeline/release", Trigger.Request())
            while time.time() - t_release < 30.0 and states[-1][1] not in ("IDLE", "ABORTED"):
                spin(0.2)
            released = (reply, link6(), t_release)
    elif a.push_force:
        return push_force_check()
    else:
        spin(a.servo_seconds)
    call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=False))
    if grasp:
        set_object_width(0.0)

    seq = [s for _, s in states]
    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    t_reach = next((t for t, s in states if s == "REACHING"), None)
    t_servo = next((t for t, s in states if s == servo_state), None)
    t_done = next((t for t, s in states if s == "AT_GRASP"), None)
    expect = ["REACHING", servo_state] + (["AT_GRASP"] if approach else []) + \
        (["GRASPING", "GRASPED"] if grasp else []) + \
        (["RELEASING", "RETREATING", "IDLE"] if approach and a.release else [])
    timing = f" (reach {t_servo - t_reach:.1f} s" if t_reach and t_servo else ""
    if timing and t_done:
        timing += f", approach {t_done - t_servo:.1f} s"
    check("stem_grasp went " + " -> ".join(expect), seq[-len(expect):] == expect,
          " -> ".join(seq) + (timing + ")" if timing else ""))
    def tcp_pose(tr):
        r = tr.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        t = tr.transform.translation
        return np.array([t.x, t.y, t.z]) + offset * R[:, 2], R[:, 2]

    grasp_point = None
    if goals and t_servo is not None and pre_grasp_tcp is not None:
        g = goals[-1]
        goal_p = np.array([g.pose.position.x, g.pose.position.y, g.pose.position.z])
        q = g.pose.orientation
        goal_z = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2]
        stand_off = param("/stem_grasp_pipeline", "target_position_offset_m").double_value
        grasp_point = goal_p + stand_off * goal_z
        tcp, z = tcp_pose(pre_grasp_tcp)
        err = float(np.linalg.norm(tcp - goal_p))
        angle = float(np.degrees(np.arccos(np.clip(z @ goal_z, -1.0, 1.0))))
        check("TCP at the pre-grasp position at the handoff (TF)", err < a.tolerance,
              f"{100 * err:.2f} cm from {np.round(goal_p, 3)}; approach axis off by {angle:.1f} deg")
    else:
        check("TCP at the pre-grasp position at the handoff (TF)", False, "no goal sent or no handoff")
    served = [s for t, s in ibvs if t > t_handoff]
    if approach and grasp_point is not None and at_grasp is not None and held is not None:
        tcp, z = tcp_pose(at_grasp)
        along = float((grasp_point - tcp) @ z)
        miss = float(np.linalg.norm(np.cross(grasp_point - tcp, z)))
        steps = max([s.get("step", 0) for s in served] or [0])
        # where the gripper axis passes the (vertical) stem centreline
        c_xy = np.array(a.stem) - tcp[:2]
        s_ax = float(c_xy @ z[:2] / max(z[:2] @ z[:2], 1e-9))
        p_ax = tcp + s_ax * z
        to_stem = float(np.linalg.norm(p_ax[:2] - np.array(a.stem)))
        # Pass on the stem itself: the grasp point comes from the skeleton of the
        # robot-facing half of the stem, a few mm in front of the centreline.
        check("TCP at the grasp point, gripper axis through the stem (TF)",
              abs(along) < 0.015 and to_stem < 0.004,
              f"axis {100 * to_stem:.2f} cm from the stem centreline at z {p_ax[2]:.3f}; "
              f"{100 * along:.2f} cm short of and {100 * miss:.2f} cm beside the grasp point "
              f"{np.round(grasp_point, 3)} along the gripper axis, after {steps} steps")
        moved = float(np.linalg.norm(tcp_pose(held)[0] - tcp))
        check("arm holds at the grasp point", moved < 0.002, f"moved {1000 * moved:.1f} mm in 1 s")
        if grasp:
            during = [w for t, w in widths if w is not None and t_servo is not None and t > t_servo]
            opened = max(during or [0.0])
            check("gripper opened for the approach and closed on the stem",
                  opened > 0.05 and abs(grip_final - stem_width) < 0.001,
                  f"opened to {1000 * opened:.1f} mm, closed to {1000 * grip_final:.1f} mm "
                  f"on a {1000 * stem_width:.0f} mm stem")
        if a.release:
            if released is None or released[1] is None:
                check("released and retreated", False, "no release (state " + seq[-1] + ")")
            else:
                reply, after, t_rel = released
                back = float((tcp - tcp_pose(after)[0]) @ z)
                side = float(np.linalg.norm(np.cross(tcp_pose(after)[0] - tcp, z)))
                opened_after = max([w for t, w in widths if w is not None and t > t_rel] or [0.0])
                check("released and retreated", reply is not None and reply.success and opened_after > 0.05
                      and 0.09 < back < 0.13 and side < 0.01,
                      f"gripper opened to {1000 * opened_after:.1f} mm, gripper backed "
                      f"{100 * back:.1f} cm along its axis ({100 * side:.2f} cm sideways)")
    elif approach:
        check("TCP at the grasp point, gripper axis through the stem (TF)", False,
              "no AT_GRASP: " + " -> ".join(seq))
    elif served and grasp_point is not None:
        first, last = served[0]["error_px"], float(np.median([s["error_px"] for s in served[-10:]]))
        check("servo image error ends small", last < 8.0,
              f"{first:.1f} px -> {last:.1f} px at depth {served[-1]['depth_m']:.3f} m "
              f"(desired uv {np.round(served[-1]['desired_uv'], 1)})")
        tcp, z = tcp_pose(tf_buffer.lookup_transform("odom", "piper_link6", rclpy.time.Time()))
        miss = float(np.linalg.norm(np.cross(grasp_point - tcp, z)))
        check("gripper axis passes through the grasp point", miss < 0.015,
              f"{100 * miss:.2f} cm from the axis (stem grasp point {np.round(grasp_point, 3)})")
    else:
        check("servo image error ends small", False, f"{len(served)} servo status messages")
    check("MPC idle after the handoff", mpc and mpc[-1][1]["mode"] == "idle",
          f"last mode {mpc[-1][1]['mode'] if mpc else None}")
    late = [t for t in jogs if t_servo is not None and t > t_servo + 0.5]   # MPC JointJog only
    check("no MPC joint commands after the handoff", t_servo is not None and not late,
          f"{len(late)} JointJog messages")
    check("base stopped", abs(odom.get("v", 1.0)) < 1e-3 and abs(odom.get("w", 1.0)) < 1e-3,
          f"at ({odom.get('x', 0.0):.3f}, {odom.get('y', 0.0):.3f})")
    # 2 / 4 halt for a singularity / collision, 5 stops at a joint limit
    check("no servo halt", not [s for s in servo if s in (2, 4, 5)], f"servo status codes {sorted(set(servo))}")
    ok = all(results)
    print("GRASP CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
