#!/usr/bin/env python3
"""Hardware-free check of the active-perception loop (Phase 4, P4.3).

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_base:=true fake_base:=true bringup_servo:=true bringup_pipeline:=false
    ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py execute:=true
    ros2 run scout_piper_bringup active_perception_chain_check.py [--seed 3]

Stands in for the camera and the semantic scene with a simulated plant
(scout_piper_active_perception ``make_ap_scene``: the target in the middle of
a branch, leaves in front of it on the approach line), placed in odom:

  1. drives the fake robot to the scan pose through the MPC (/whole_body_mpc/goal_pose);
  2. renders the view from the camera pose in TF, integrates it into a CPU
     semantic voxel map and publishes the field (/scene_repr/distance_field),
     the target's points (/stem_grasp/filtered_cloud, odom), camera_info and a
     noisy target point (/piper_jepa/target_point);
  3. starts active_perception_node, calls ~/plan_view, forwards the view to
     the MPC and waits for it to be reached;
  4. renders and publishes from the camera pose actually reached, repeats
     the plan once more, and checks: views planned and reached, the camera
     looking at the target, the joint estimate fitted from the new views
     with a small axis error, and the uncertainty reported falling.

With ``--pipeline`` the stem_grasp pipeline runs as well (launched with
``-p active_perception_service:=/active_perception/plan_view``): it asks for
the views itself (VIEWING, full-orientation MPC goals) while active
perception reports needs_view, goes back to the scan pose when the last view
does not show the stem, and then reaches; this script only renders and
publishes what the camera sees.

Starts moveit_servo and enables the servo bridge (the MPC's arm commands go
through them); runs only against the fake drivers; cancels the MPC and
disables the bridge at the end.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np


def _pose_matrix(pose) -> np.ndarray:
    """4x4 of a geometry_msgs Pose."""
    w, x, y, z = pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--views", type=int, default=2, help="views to plan after the first")
    ap.add_argument("--reach-timeout", type=float, default=60.0)
    ap.add_argument("--pipeline", action="store_true",
                    help="the stem_grasp pipeline asks for the views (active_perception_service) and then reaches")
    ap.add_argument("--timeout", type=float, default=240.0, help="--pipeline: seconds until REACHING")
    a = ap.parse_args()

    import rclpy  # noqa: PLC0415
    from geometry_msgs.msg import PointStamped, PoseStamped  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters  # noqa: PLC0415
    from scout_piper_active_perception.candidates import camera_pose_fn  # noqa: PLC0415
    from scout_piper_active_perception.scene import (CLASSES, Camera, DepthSensor, backproject,  # noqa: PLC0415
                                                     make_ap_scene, render)
    from scout_piper_active_perception.sim import initial_state  # noqa: PLC0415
    from scout_piper_active_perception.stem_joint import Cylinder, axis_error  # noqa: PLC0415
    from scout_piper_scene_repr.msg import SemanticDistanceField  # noqa: PLC0415
    from scout_piper_scene_repr_py.distance_query import SemanticDistanceQuery  # noqa: PLC0415
    from scout_piper_scene_repr_py.field import export_field, fill_msg  # noqa: PLC0415
    from scout_piper_scene_repr_py.voxel_map import SemanticVoxelMap, grid_around  # noqa: PLC0415
    from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel  # noqa: PLC0415
    from sensor_msgs.msg import CameraInfo, PointCloud2  # noqa: PLC0415
    from sensor_msgs_py import point_cloud2  # noqa: PLC0415
    from std_msgs.msg import Empty, Header, String  # noqa: PLC0415
    from std_srvs.srv import SetBool, Trigger  # noqa: PLC0415
    from tf2_ros import Buffer, TransformListener  # noqa: PLC0415

    rclpy.init()
    n = Node("active_perception_chain_check")
    tf_buf = Buffer()
    TransformListener(tf_buf, n)
    pub_field = n.create_publisher(SemanticDistanceField, "/scene_repr/distance_field", 1)
    pub_cloud = n.create_publisher(PointCloud2, "/stem_grasp/filtered_cloud", 1)
    pub_info = n.create_publisher(CameraInfo, "/camera/color/camera_info", 1)
    pub_target = n.create_publisher(PointStamped, "/piper_jepa/target_point", 1)
    pub_goal = n.create_publisher(PoseStamped, "/whole_body_mpc/goal_pose", 1)
    pub_cancel = n.create_publisher(Empty, "/whole_body_mpc/cancel", 1)
    mpc, ap_status, views = [], [], []
    n.create_subscription(String, "/whole_body_mpc/status", lambda m: mpc.append((time.time(), json.loads(m.data))), 20)
    n.create_subscription(String, "/active_perception/status", lambda m: ap_status.append(json.loads(m.data)), 10)
    n.create_subscription(PoseStamped, "/active_perception/next_view", lambda m: views.append(m), 5)

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.01)

    def call(srv_type, name, request, timeout=30.0):
        cl = n.create_client(srv_type, name)
        if not cl.wait_for_service(timeout_sec=timeout):
            return None
        fut = cl.call_async(request)
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        return fut.result()

    reply = call(GetParameters, "/piper_ctrl_single_node/get_parameters", GetParameters.Request(names=["initial_gripper"]), 15.0)
    if reply is None or not reply.values or reply.values[0].type == 0:
        print("REFUSED: /piper_ctrl_single_node is not the fake driver; launch with fake_arm:=true", flush=True)
        return 2

    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    def lookup(target, source):
        try:
            tr = tf_buf.lookup_transform(target, source, rclpy.time.Time()).transform
        except Exception:  # noqa: BLE001
            return None
        q, t = tr.rotation, tr.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    cam_frame, world = "camera_color_optical_frame", "odom"
    for _ in range(100):
        spin(0.1)
        if lookup(world, cam_frame) is not None and lookup("piper_link6", cam_frame) is not None:
            break
    T_l6c = lookup("piper_link6", cam_frame)
    if T_l6c is None:
        check("TF odom -> camera", False, "no transform")
        return 1

    scene = make_ap_scene(a.seed, prefer_branch=not a.pipeline)    # the pipeline grasps a main stem
    p_true, d_true, _ = scene.joint_axis()
    camera = Camera()
    sensor = DepthSensor()
    rng = np.random.default_rng(a.seed)
    model = WholeBodyModel()
    cam_fn = camera_pose_fn(model, T_l6c)
    vmap = SemanticVoxelMap(grid_around(p_true, 0.3, 0.01), classes=CLASSES)
    query = SemanticDistanceQuery(vmap)
    K = camera.K
    info = CameraInfo()
    info.header.frame_id = cam_frame
    info.height, info.width = camera.image_hw
    info.k = [float(v) for v in K.reshape(-1)]
    target_noisy = scene.target + rng.normal(0.0, 0.005, 3)

    def pose_msg(T):
        from scout_piper_active_perception.active_perception_node import _pose_msg  # noqa: PLC0415
        return _pose_msg(T, world, n.get_clock().now().to_msg())

    def drive(T_tcp, label):
        """Send a TCP pose goal and wait until the MPC holds it for 1 s:
        "reached", or "handoff_ready" within 2 cm (the approach axis just
        outside the MPC's 15 deg tolerance does not matter for a view)."""
        mpc.clear()
        t0 = time.time()
        since = None
        while time.time() - t0 < a.reach_timeout:
            pub_goal.publish(pose_msg(T_tcp))
            spin(0.5)
            last = mpc[-1][1] if mpc else {}
            if last.get("mode") == "reached" or (last.get("mode") == "handoff_ready"
                                                  and last.get("tcp_error_m", 1.0) < 0.02):
                since = since or time.time()
                if time.time() - since > 1.0:
                    break
            else:
                since = None
        T_now = lookup(world, "piper_link6")
        tcp = (T_now @ np.r_[0, 0, 0.14, 1.0])[:3] if T_now is not None else np.full(3, np.nan)
        err = float(np.linalg.norm(tcp - T_tcp[:3, 3]))
        last = mpc[-1][1] if mpc else {}
        check(f"MPC reaches the {label}", since is not None and err < 0.03,
              f"{time.time() - t0:.1f} s, TCP {100 * err:.1f} cm from the goal"
              + ("" if since is not None else f"; last MPC status {json.dumps(last)[:200]}"))
        pub_cancel.publish(Empty())
        spin(0.5)

    stamp = [0.0]

    def observe():
        """Render from the camera in TF, integrate; the target's points (with
        --pipeline: the stem and target points within 8 cm of the target, the
        pipeline's stem cloud)."""
        T = lookup(world, cam_frame)
        fr = sensor.sense(render(scene, T, camera), camera, rng)
        masks = {c: fr["label"] == i for i, c in enumerate(CLASSES)}
        stamp[0] = n.get_clock().now().nanoseconds * 1e-9
        vmap.integrate(np.nan_to_num(fr["depth"], nan=0.0), masks, K, T, stamp[0], stride=2)
        if a.pipeline:
            P = backproject(fr["depth"], K, T, masks["target"] | masks["stem"])
            P = P[np.linalg.norm(P - scene.target, axis=1) < 0.08]
        else:
            P = backproject(fr["depth"], K, T, masks["target"])
        return T, P, int(masks["target"].sum())

    def publish(P, sec=3.0):
        snap = export_field(query)
        end = time.time() + sec
        while time.time() < end:
            now = n.get_clock().now()
            snap.stamp = now.nanoseconds * 1e-9
            pub_field.publish(fill_msg(SemanticDistanceField(), snap, world))
            info.header.stamp = now.to_msg()
            pub_info.publish(info)
            pt = PointStamped()
            pt.header.frame_id, pt.header.stamp = world, now.to_msg()
            pt.point.x, pt.point.y, pt.point.z = map(float, target_noisy)
            pub_target.publish(pt)
            if len(P):
                h = Header(frame_id=world, stamp=now.to_msg())
                pub_cloud.publish(point_cloud2.create_cloud_xyz32(h, P.astype(np.float32).tolist()))
            spin(0.3)

    def pipeline_flow(T_render, P):
        """The pipeline asks for views (VIEWING) while active perception says
        needs_view, then reaches. This loop is the camera: it re-renders when
        the pipeline is back in SCANNING at a settled new pose, and publishes
        the stem cloud only for the pose it was rendered at."""
        states, planned_cam, view_err = [], [], []
        n.create_subscription(String, "/stem_grasp/pipeline_state",
                              lambda m: (not states or states[-1] != m.data) and states.append(m.data), 20)
        n.create_subscription(PoseStamped, "/active_perception/next_view_camera", planned_cam.append, 5)
        renders, px_views, returned = 0, [], []
        moving_since, last_T = None, lookup(world, cam_frame)
        t0 = time.time()
        first = None
        while time.time() - t0 < a.timeout:
            Tc = lookup(world, cam_frame)
            moved = Tc is not None and (np.linalg.norm(Tc[:3, 3] - T_render[:3, 3]) > 0.005
                                        or abs(np.trace(Tc[:3, :3].T @ T_render[:3, :3]) - 3.0) > 1e-3)
            still = Tc is not None and last_T is not None and np.linalg.norm(Tc[:3, 3] - last_T[:3, 3]) < 1e-3
            last_T = Tc
            scanning = bool(states) and states[-1] == "SCANNING"
            if moved:
                moving_since = moving_since or time.time()
                # render once the pipeline is back in SCANNING at a settled pose (the view
                # reached): rendering at every pause of a motion only starves the MPC
                if scanning and still and time.time() - moving_since > 0.6:
                    T_render, P, px = observe()
                    renders += 1
                    if "VIEWING" in states and "REACHING" not in states and len(px_views) >= len(planned_cam):
                        returned.append(px)            # more renders than planned views: the return
                        print(f"info back at the scan pose (no stem cloud after the views): {px} target pixels",
                              flush=True)
                    elif "VIEWING" in states and "REACHING" not in states:      # a view, not the reach
                        px_views.append(px)
                        rel = scene.target - T_render[:3, 3]
                        ang = np.degrees(np.arccos(np.clip(rel @ T_render[:3, 2] / np.linalg.norm(rel), -1, 1)))
                        lc = (ap_status[-1] if ap_status else {}).get("last_choice") or {}
                        err, px_plan = float("nan"), None
                        if planned_cam:
                            pc = planned_cam[-1].pose
                            err = float(np.linalg.norm(T_render[:3, 3] - [pc.position.x, pc.position.y, pc.position.z]))
                            T_plan = _pose_matrix(pc)
                            px_plan = int((render(scene, T_plan, camera)["label"] == CLASSES.index("target")).sum())
                        view_err.append(err)
                        print(f"info view {len(px_views)}: camera {100 * err:.1f} cm from the planned one, {px} "
                              f"target pixels ({px_plan} at the planned pose, {lc.get('vis_px')} predicted), "
                              f"{ang:.0f} deg off axis at "
                              f"{np.linalg.norm(rel):.2f} m, turn {lc.get('turn_deg')} deg", flush=True)
                    moving_since = None
            now = n.get_clock().now()
            snap = export_field(query)
            snap.stamp = now.nanoseconds * 1e-9
            pub_field.publish(fill_msg(SemanticDistanceField(), snap, world))
            info.header.stamp = now.to_msg()
            pub_info.publish(info)
            pt = PointStamped()
            pt.header.frame_id, pt.header.stamp = world, now.to_msg()
            pt.point.x, pt.point.y, pt.point.z = map(float, target_noisy)
            pub_target.publish(pt)
            st = ap_status[-1] if ap_status else {}
            if first is None and "u_total" in st:
                first = st
            # the stem cloud only once active perception reports (else the pipeline grasps
            # without asking) and only for the pose it was rendered at
            if first is not None and not moved and len(P):
                h = Header(frame_id=world, stamp=now.to_msg())
                pub_cloud.publish(point_cloud2.create_cloud_xyz32(h, P.astype(np.float32).tolist()))
            spin(0.25)
            if "REACHING" in states:
                spin(2.0)
                break
        views = states.count("VIEWING") - len(returned)
        last = ap_status[-1] if ap_status else {}
        check("pipeline asked for views before reaching (VIEWING)", views >= 1 and "REACHING" in states
              and states.index("VIEWING") < states.index("REACHING") if views else False,
              f"states {' -> '.join(states)}; {renders} new camera poses rendered")
        # views after the joint is known look into the approach corridor, not at the target
        check("views reached (camera within 4 cm of the planned camera pose)",
              view_err and all(e < 0.04 for e in view_err), f"{[round(100 * e, 1) for e in view_err]} cm")
        check("uncertainty fell over the views", first is not None and (
              last.get("u_joint", 1.0) < first.get("u_joint", 1.0)
              or (last.get("corridor_unknown") or 1.0) < (first.get("corridor_unknown") or 1.0)),
              f"u_joint {first and first.get('u_joint')} -> {last.get('u_joint')}, corridor unknown "
              f"{first and first.get('corridor_unknown')} -> {last.get('corridor_unknown')}, unmet {last.get('unmet')}")
        ok = all(results)
        print("ACTIVE PERCEPTION GRASP", "OK" if ok else "FAILED", flush=True)
        return 0 if ok else 1

    # the MPC's arm commands go through moveit_servo and the servo bridge
    call(Trigger, "/servo_node/start_servo", Trigger.Request())
    en = call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=True))
    if en is None or not en.success:
        check("servo bridge enabled", False, "no /piper_servo_bridge/enable")
        return 1

    # 1. scan pose: camera on the approach line, 0.4 m from the target
    axis = np.r_[target_noisy[:2], 0.0] / np.linalg.norm(target_noisy[:2])
    x_scan = initial_state(model, cam_fn, scene, target_noisy, axis, rng=np.random.default_rng(a.seed + 3))
    drive(model.tcp_world(x_scan), "scan pose")
    T0, P, px = observe()
    print(f"info first view: {px} target pixels (leaves in front of the target)", flush=True)

    node = subprocess.Popen(["ros2", "run", "scout_piper_active_perception", "active_perception_node", "--ros-args",
                             "-p", "required:=[joint, corridor]", "-p", "n_candidates:=120"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=True)
    try:
        if a.pipeline:
            return pipeline_flow(T0, P)
        publish(P, 6.0)
        first = ap_status[-1] if ap_status else {}
        check("status published with the uncertainty terms", "u_total" in first and first.get("field"),
              {k: first.get(k) for k in ("u_total", "u_joint", "u_corridor", "needs_view", "field", "state")})
        sig0 = first.get("u_joint", 1.0)
        corr0 = first.get("corridor_unknown")
        for k in range(a.views):
            views.clear()
            res = call(Trigger, "/active_perception/plan_view", Trigger.Request(), 60.0)
            spin(0.5)
            ok = res is not None and res.success and views
            check(f"view {k + 1} planned", ok, (res.message[:160] if res is not None else "no service"))
            if not ok:
                break
            v = views[-1]
            q = v.pose.orientation
            from scipy.spatial.transform import Rotation  # noqa: PLC0415
            T_goal = np.eye(4)
            T_goal[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T_goal[:3, 3] = [v.pose.position.x, v.pose.position.y, v.pose.position.z]
            drive(T_goal, f"view {k + 1}")
            T, P, px = observe()
            rel = scene.target - T[:3, 3]
            ang = np.degrees(np.arccos(np.clip(rel @ T[:3, 2] / np.linalg.norm(rel), -1, 1)))
            # the MPC's approach tolerance (15 deg) and the camera's offset from the TCP axis
            check(f"camera at view {k + 1} sees the target", ang < 25.0 and 0.12 < np.linalg.norm(rel) < 0.55
                  and px >= 30, f"{ang:.1f} deg off axis at {np.linalg.norm(rel):.2f} m, {px} target pixels")
            publish(P, 4.0)
        st = ap_status[-1] if ap_status else {}
        j = st.get("joint") or {}
        if j.get("fitted"):
            est = Cylinder(np.array(j["p"]), np.array(j["d"]), j["r_mm"] / 1000.0)
            err = axis_error(est, p_true, d_true, 0.03)
        else:
            err = float("inf")
        check("stem joint estimated from the planned views", j.get("fitted") and j.get("views", 0) >= 1 and err < 0.003,
              f"{j.get('views')} views, {j.get('points')} points, axis error {1000 * err:.2f} mm, "
              f"sigma {st.get('axis_sigma_mm')} mm")
        check("uncertainty falls", st.get("u_joint", 1.0) < sig0 or (corr0 is not None and
                                                                      (st.get("corridor_unknown") or 1.0) < corr0),
              f"u_joint {sig0} -> {st.get('u_joint')}, corridor unknown {corr0} -> {st.get('corridor_unknown')}, "
              f"needs_view {st.get('needs_view')} {st.get('unmet')}")
    finally:
        pub_cancel.publish(Empty())
        spin(0.3)
        call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=False), 5.0)
        try:
            os.killpg(node.pid, signal.SIGINT)        # ros2 run's child too
            out, _ = node.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(node.pid, signal.SIGKILL)
            out = ""
        except ProcessLookupError:
            out = ""
        bad = [ln for ln in (out or "").splitlines() if "Traceback" in ln or "Error" in ln]
        if bad:
            print("info active_perception_node: " + " | ".join(bad[:3]), flush=True)
    ok = all(results)
    print("ACTIVE PERCEPTION CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
