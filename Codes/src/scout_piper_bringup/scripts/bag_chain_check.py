#!/usr/bin/env python3
"""Hardware-free check of the dataset path: record_bag.sh -> rosbag2 -> episode -> offline map.

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_base:=true fake_base:=true bringup_servo:=true bringup_rviz:=false bringup_pipeline:=false
    ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py execute:=true
    ros2 run scout_piper_bringup bag_chain_check.py --record-script Codes/scripts/record_bag.sh --out-dir /tmp/bag_check

Stands in for the camera and the segmentation: renders a simulated plant
(scout_piper_active_perception ``make_ap_scene``, in odom) from the camera pose
in TF and publishes colour (rgb8), aligned depth (16UC1, mm), both
camera_info topics, the merged label image (/stem_grasp/semantic_label: stem
1, branch 2, leaf 3, target 4) and the target mask, while the whole-body MPC
moves the robot through two poses. ``record_bag.sh scene`` records it all
(with TF, odom and joint states from the fake drivers); then the bag is
exported with ``scout_piper_jepa.episode.export_bag`` and checked against what
was published: frames and stamps, depth in metres, intrinsics, the camera pose
of every frame from the recorded TF, whole-body states and gripper, masks and
labels; finally ``scout_piper_scene_repr_py.offline.map_from_episode`` builds
the semantic map from the episode and must hold the target where it is.

Runs only against the fake drivers. Exit code 0 when all pass.
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

LABEL_ID = {"stem": 1, "branch": 2, "leaf": 3, "target": 4}
PALETTE = {"stem": (60, 140, 60), "branch": (80, 120, 50), "leaf": (40, 170, 40), "target": (220, 60, 60),
           "other": (120, 120, 120)}


def _rot_deg(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-script", required=True, help="Codes/scripts/record_bag.sh")
    ap.add_argument("--out-dir", default="/tmp/bag_chain_check")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--goal-timeout", type=float, default=30.0)
    a = ap.parse_args()

    import rclpy  # noqa: PLC0415
    from geometry_msgs.msg import PoseStamped  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from scout_piper_active_perception.active_perception_node import _pose_msg  # noqa: PLC0415
    from scout_piper_active_perception.candidates import camera_pose_fn  # noqa: PLC0415
    from scout_piper_active_perception.scene import CLASSES, Camera, make_ap_scene, render  # noqa: PLC0415
    from scout_piper_active_perception.sim import initial_state  # noqa: PLC0415
    from scout_piper_jepa.episode import export_bag  # noqa: PLC0415
    from scout_piper_scene_repr_py.offline import map_from_episode  # noqa: PLC0415
    from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel  # noqa: PLC0415
    from sensor_msgs.msg import CameraInfo, Image  # noqa: PLC0415
    from std_msgs.msg import Empty, String  # noqa: PLC0415
    from std_srvs.srv import SetBool, Trigger  # noqa: PLC0415
    from tf2_ros import Buffer, TransformListener  # noqa: PLC0415

    rclpy.init()
    n = Node("bag_chain_check")
    tf_buf = Buffer()
    TransformListener(tf_buf, n)
    pubs = {k: n.create_publisher(m, t, 5) for k, m, t in (
        ("rgb", Image, "/camera/color/image_raw"), ("depth", Image, "/camera/aligned_depth_to_color/image_raw"),
        ("info", CameraInfo, "/camera/color/camera_info"),
        ("dinfo", CameraInfo, "/camera/aligned_depth_to_color/camera_info"),
        ("label", Image, "/stem_grasp/semantic_label"), ("target", Image, "/stem_grasp/target_mask"))}
    pub_goal = n.create_publisher(PoseStamped, "/whole_body_mpc/goal_pose", 1)
    pub_cancel = n.create_publisher(Empty, "/whole_body_mpc/cancel", 1)
    mpc = []
    n.create_subscription(String, "/whole_body_mpc/status", lambda m: mpc.append(json.loads(m.data)), 20)

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.01)

    def call(srv_type, name, request, timeout=20.0):
        cl = n.create_client(srv_type, name)
        if not cl.wait_for_service(timeout_sec=timeout):
            return None
        fut = cl.call_async(request)
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        return fut.result()

    reply = call(GetParameters, "/piper_ctrl_single_node/get_parameters", GetParameters.Request(names=["initial_gripper"]))
    if reply is None or not reply.values or reply.values[0].type == 0:
        print("REFUSED: /piper_ctrl_single_node is not the fake driver; launch with fake_arm:=true", flush=True)
        return 2
    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    def lookup(target, source, with_stamp=False):
        try:
            ts = tf_buf.lookup_transform(target, source, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return (None, None) if with_stamp else None
        tr = ts.transform
        q, t = tr.rotation, tr.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = [t.x, t.y, t.z]
        return (T, ts.header.stamp) if with_stamp else T

    cam_frame, world = "camera_color_optical_frame", "odom"
    for _ in range(100):
        spin(0.1)
        if lookup(world, cam_frame) is not None and lookup("piper_link6", cam_frame) is not None:
            break
    T_l6c = lookup("piper_link6", cam_frame)
    if T_l6c is None:
        check("TF odom -> camera", False, "no transform")
        return 1
    call(Trigger, "/servo_node/start_servo", Trigger.Request())
    en = call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=True))
    if en is None or not en.success:
        check("servo bridge enabled", False, en.message if en is not None else "no /piper_servo_bridge/enable")
        return 1

    scene = make_ap_scene(a.seed, prefer_branch=False)
    camera = Camera(image_hw=(120, 212))              # half the check's usual size: ~4 renders a second
    K = camera.K
    H, W = camera.image_hw
    model = WholeBodyModel()
    cam_fn = camera_pose_fn(model, T_l6c)
    axis = np.r_[scene.target[:2], 0.0] / np.linalg.norm(scene.target[:2])
    x_scan = initial_state(model, cam_fn, scene, scene.target, axis, rng=np.random.default_rng(a.seed + 3))
    T1 = model.tcp_world(x_scan)
    T2 = T1.copy()
    T2[:3, 3] += [0.0, 0.05, 0.03]                    # a second view 5 cm aside, 3 cm up

    def image(arr, enc, stamp):
        m = Image()
        m.header.frame_id, m.header.stamp = cam_frame, stamp
        m.height, m.width = arr.shape[:2]
        m.encoding, m.is_bigendian = enc, 0
        m.step = arr.strides[0]
        m.data = arr.tobytes()
        return m

    published = {}                                    # stamp ns -> (T, depth m, label ids, target mask)

    def publish_frame():
        # the image carries the stamp of the TF it was rendered from (rendering takes
        # ~0.2 s: stamped afterwards, a moving camera's frames would be 0.2 s late)
        T, stamp = lookup(world, cam_frame, with_stamp=True)
        if T is None:
            return
        r = render(scene, T, camera)
        lab = r["label"]
        ids = np.zeros((H, W), np.uint8)
        rgb = np.full((H, W, 3), (200, 200, 220), np.uint8)
        for i, c in enumerate(CLASSES):
            m = lab == i
            ids[m] = LABEL_ID.get(c, 0)
            rgb[m] = PALETTE[c]
        # the D405's range: no return beyond far_m (nor in front of near_m)
        d = np.where(np.isfinite(r["depth"]) & (r["depth"] >= camera.near_m) & (r["depth"] <= camera.far_m),
                     r["depth"], 0.0)
        shade = np.clip(1.2 - d, 0.5, 1.0)[..., None]
        rgb = (rgb * np.where(d[..., None] > 0, shade, 1.0)).astype(np.uint8)
        mm = np.round(d * 1000.0).astype(np.uint16)
        tgt = ((lab == CLASSES.index("target")) * 255).astype(np.uint8)
        for key, arr, enc in (("rgb", rgb, "rgb8"), ("depth", mm, "16UC1"), ("label", ids, "mono8"),
                              ("target", tgt, "mono8")):
            pubs[key].publish(image(arr, enc, stamp))
        for key in ("info", "dinfo"):
            info = CameraInfo()
            info.header.frame_id, info.header.stamp = cam_frame, stamp
            info.height, info.width = H, W
            info.k = [float(v) for v in K.reshape(-1)]
            pubs[key].publish(info)
        published[stamp.sec * 10 ** 9 + stamp.nanosec] = (T, mm / 1000.0, ids, tgt > 0)

    os.makedirs(a.out_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(a.out_dir, "scene_*")))
    rec = subprocess.Popen([a.record_script, "scene", "smoke"], env=dict(os.environ, BAG_DIR=a.out_dir),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    spin(3.0)                                         # the recorder subscribes

    reached = []
    for label, T_goal in (("scan pose", T1), ("second view", T2)):
        mpc.clear()
        t0, since = time.time(), None
        while time.time() - t0 < a.goal_timeout:
            pm = _pose_msg(T_goal, world, n.get_clock().now().to_msg())
            pub_goal.publish(pm)
            publish_frame()
            spin(0.05)
            last = mpc[-1] if mpc else {}
            if last.get("mode") == "reached" or (last.get("mode") == "handoff_ready" and last.get("tcp_error_m", 1) < 0.02):
                since = since or time.time()
                if time.time() - since > 1.0:
                    break
            else:
                since = None
        reached.append((label, since is not None, round(time.time() - t0, 1)))
    pub_cancel.publish(Empty())
    for _ in range(3):
        publish_frame()
        spin(0.1)
    spin(1.0)
    os.killpg(rec.pid, signal.SIGINT)                 # Ctrl-C: rosbag2 closes the bag
    try:
        rec.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(rec.pid, signal.SIGKILL)
    call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=False), 5.0)
    check("MPC moved the robot through both poses", all(r[1] for r in reached), str(reached))

    bags = sorted(set(glob.glob(os.path.join(a.out_dir, "scene_*"))) - before)
    bag = next((b for b in bags if os.path.isdir(b)), None)
    note = next((b for b in bags if b.endswith(".json")), None)
    check("record_bag.sh wrote a bag and its note", bag is not None and note is not None, f"{bags}")
    if bag is None:
        print("BAG CHAIN FAILED", flush=True)
        return 1
    import rosbag2_py  # noqa: PLC0415
    info = rosbag2_py.Info().read_metadata(bag, "")
    counts = {t.topic_metadata.name: t.message_count for t in info.topics_with_message_count}
    need = ("/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", "/camera/color/camera_info",
            "/tf", "/tf_static", "/odom", "/joint_states", "/stem_grasp/semantic_label", "/stem_grasp/target_mask")
    check("bag holds the scene preset's topics", all(counts.get(t, 0) > 0 for t in need),
          ", ".join(f"{t.split('/')[-1]} {counts.get(t, 0)}" for t in need))

    out = os.path.join(a.out_dir, "episode.npz")
    summary = export_bag(bag, out, "/camera/color/image_raw", 1, "/camera/aligned_depth_to_color/image_raw",
                         "/camera/color/camera_info", "/stem_grasp/target_mask", world, "",
                         "/stem_grasp/semantic_label")
    ep = dict(np.load(out))
    nf = len(ep["frames"])
    check("episode exported with every frame", nf >= 0.95 * len(published) and nf > 10,
          f"{nf} frames of {len(published)} published, fields {summary['fields']}")
    keys = sorted(published)
    match = [min(keys, key=lambda k: abs(k - s * 1e9)) for s in ep["stamps"]]
    dt = max(abs(k - s * 1e9) for k, s in zip(match, ep["stamps"])) * 1e-9
    check("frame stamps are the published ones", dt < 1e-6, f"max offset {dt * 1e6:.2f} us")
    derr, perr, aerr, mask_ok, lab_ok = [], [], [], 0, 0
    for i, k in enumerate(match):
        T, d, ids, tgt = published[k]
        de = ep["depth"][i].astype(np.float32)
        v = d > 0
        derr.append(float(np.abs(de[v] - d[v]).max()) if v.any() else 0.0)
        Te = ep["T_world_cam"][i]
        perr.append(float(np.linalg.norm(Te[:3, 3] - T[:3, 3])) if np.isfinite(Te).all() else np.inf)
        aerr.append(_rot_deg(Te[:3, :3], T[:3, :3]) if np.isfinite(Te).all() else np.inf)
        mask_ok += bool(np.array_equal(ep["seg_masks"][i], tgt))
        lab_ok += bool(np.array_equal(ep["labels"][i], ids))
    check("depth in metres as published (16UC1 mm -> m)", max(derr) < 1e-3, f"max error {1000 * max(derr):.2f} mm")
    check("intrinsics", np.allclose(ep["K"], K), str(np.round(ep["K"], 1).tolist()))
    check("camera pose of every frame from the recorded TF", max(perr) < 0.01 and max(aerr) < 2.0,
          f"max {100 * max(perr):.2f} cm, {max(aerr):.2f} deg against the pose rendered from")
    st = ep["states"]
    moved = float(np.abs(st[-1] - st[0]).max()) if np.isfinite(st).all() else 0.0
    check("whole-body states and gripper in every frame", np.isfinite(st).all() and moved > 0.02
          and np.isfinite(ep.get("gripper", np.array([np.nan]))).all(),
          f"state change {moved:.3f}, gripper {np.nanmean(ep.get('gripper', np.array([np.nan]))) * 1000:.1f} mm")
    check("target masks and labels as published", mask_ok == nf and lab_ok == nf,
          f"{mask_ok} / {lab_ok} of {nf} frames identical")

    vmap = map_from_episode(ep)
    tp = vmap.occupied_points("target")
    sp = vmap.occupied_points("stem")
    dmin = float(np.linalg.norm(tp - scene.target, axis=1).min()) if len(tp) else np.inf
    check("offline semantic map from the episode holds the target", len(tp) > 0 and dmin < 0.02,
          f"{len(tp)} target and {len(sp)} stem voxels; nearest target voxel {100 * dmin:.1f} cm from the target")
    ok = all(results)
    print("BAG CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
