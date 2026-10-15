#!/usr/bin/env python3
"""Hardware-free check of the operator layer (Phase 5): language, countdown, GUI, stop.

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_base:=true fake_base:=true bringup_servo:=true bringup_pipeline:=false
    ros2 run scout_piper_bringup operator_chain_check.py

Starts scout_piper_language's instruction_node and operator_gui_node (port
8099) and publishes stem_grasp-style detections (a red and a yellow flower
40 cm in front of the camera, a stem between them) in the camera frame. Then,
through the GUI's HTTP API as the page would:

  1. "grasp the yellow flower": a proposal for the yellow one, committed after
     the 2 s countdown: /operator/target at the yellow flower's point in odom
     (TF camera -> odom), /piper_jepa/init_mask from its outline;
  2. "grasp the flower": a question with two options; the answer "the left one"
     gives a proposal for the red one; cancel: no target published;
  3. "cut the stem": refused in plain language, nothing proposed;
  4. drive enabled, STOP: zero /cmd_vel, MPC cancel, servo bridge disabled
     (confirmed), drive disabled.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np


def main() -> int:
    import rclpy  # noqa: PLC0415
    from geometry_msgs.msg import PointStamped, Twist  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters  # noqa: PLC0415
    from sensor_msgs.msg import Image  # noqa: PLC0415
    from std_msgs.msg import Empty, String  # noqa: PLC0415
    from tf2_ros import Buffer, TransformListener  # noqa: PLC0415

    rclpy.init()
    n = Node("operator_chain_check")
    tf_buf = Buffer()
    TransformListener(tf_buf, n)
    pub_dets = n.create_publisher(String, "/stem_grasp/detections", 1)
    events, says, targets, masks, base, cancels, session = [], [], [], [], [], [], []
    n.create_subscription(String, "/operator/proposal", lambda m: events.append(json.loads(m.data)), 20)
    n.create_subscription(String, "/operator/say", lambda m: says.append(m.data), 20)
    n.create_subscription(PointStamped, "/operator/target", lambda m: targets.append(m), 5)
    n.create_subscription(Image, "/piper_jepa/init_mask", lambda m: masks.append(m), 5)
    n.create_subscription(Twist, "/cmd_vel", lambda m: base.append((time.time(), m.linear.x, m.angular.z)), 50)
    n.create_subscription(Empty, "/whole_body_mpc/cancel", lambda m: cancels.append(time.time()), 5)
    n.create_subscription(String, "/operator/state", lambda m: session.append(json.loads(m.data)), 5)

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.01)

    cl = n.create_client(GetParameters, "/piper_ctrl_single_node/get_parameters")
    if not cl.wait_for_service(timeout_sec=15.0):
        print("REFUSED: no /piper_ctrl_single_node; launch the fake arm", flush=True)
        return 2
    fut = cl.call_async(GetParameters.Request(names=["initial_gripper"]))
    end = time.time() + 10
    while not fut.done() and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    if fut.result() is None or not fut.result().values or fut.result().values[0].type == 0:
        print("REFUSED: /piper_ctrl_single_node is not the fake driver; launch with fake_arm:=true", flush=True)
        return 2

    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    port = 8099
    url = f"http://127.0.0.1:{port}/"

    def post(path, body=None):
        req = urllib.request.Request(url + path, data=json.dumps(body or {}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def get(path):
        with urllib.request.urlopen(url + path, timeout=5) as r:
            return json.loads(r.read())

    cam = "camera_color_optical_frame"
    dets = [  # camera optical frame: x right, y down, z forward; image 480 x 640, f 430
        {"index": 0, "class": "flower", "conf": 0.9, "bbox": [150, 160, 210, 220], "centroid": [180.0, 190.0],
         "area_px": 2800, "colour": "red", "p_camera": [-0.13, -0.05, 0.40],
         "polygon": [[150, 160], [210, 160], [210, 220], [150, 220]]},
        {"index": 1, "class": "flower", "conf": 0.85, "bbox": [430, 150, 490, 210], "centroid": [460.0, 180.0],
         "area_px": 2700, "colour": "yellow", "p_camera": [0.13, -0.06, 0.40],
         "polygon": [[430, 150], [490, 150], [490, 210], [430, 210]]},
        {"index": 2, "class": "stem", "conf": 0.8, "bbox": [310, 200, 330, 470], "centroid": [320.0, 335.0],
         "area_px": 5000, "colour": "green", "p_camera": [0.0, 0.05, 0.42],
         "polygon": [[310, 200], [330, 200], [330, 470], [310, 470]]},
    ]

    def publish_dets(sec):
        end = time.time() + sec
        while time.time() < end:
            pub_dets.publish(String(data=json.dumps({"stamp": time.time(), "frame_id": cam, "image_hw": [480, 640],
                                                     "detections": dets})))
            spin(0.2)

    procs = [subprocess.Popen(["ros2", "run", "scout_piper_language", "instruction_node", "--ros-args",
                               "-p", "field_topic:=''", "-p", "countdown_s:=2.0"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=True),
             subprocess.Popen(["ros2", "run", "scout_piper_language", "operator_gui_node", "--ros-args",
                               "-p", f"port:={port}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=True)]
    try:
        for _ in range(60):                   # both nodes up: GUI serving, instruction node publishing
            publish_dets(0.5)
            try:
                if get("api/state").get("detections") and session and session[-1].get("detections"):
                    break
            except OSError:
                pass
        publish_dets(1.0)                     # discovery of the GUI's publishers by the instruction node
        st = get("api/state")
        check("GUI serves the robot state", len(st.get("detections", [])) == 3,
              f"{len(st.get('detections', []))} detections, pipeline {st.get('pipeline')}")

        # 1. propose, count down, commit
        events.clear(); targets.clear(); masks.clear()
        post("api/instruction", {"text": "grasp the yellow flower"})
        t0 = time.time()
        publish_dets(1.0)
        props = [e for e in events if e["type"] == "proposal"]
        check("proposal for the yellow flower", props and props[-1]["target"]["index"] == 1,
              props[-1]["message"] if props else f"events {events}, says {says[-3:]}")
        publish_dets(2.0)
        commit = [e for e in events if e["type"] == "commit"]
        T = None
        try:
            tr = tf_buf.lookup_transform("odom", cam, rclpy.time.Time()).transform
            from scipy.spatial.transform import Rotation  # noqa: PLC0415
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]).as_matrix()
            T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
        except Exception:  # noqa: BLE001
            pass
        exp = None if T is None else T[:3, :3] @ np.array(dets[1]["p_camera"]) + T[:3, 3]
        got = None if not targets else np.array([targets[-1].point.x, targets[-1].point.y, targets[-1].point.z])
        err = float(np.linalg.norm(got - exp)) if got is not None and exp is not None else float("inf")
        check("committed after the countdown, target in odom", commit and err < 0.01,
              f"commit {time.time() - t0:.1f} s after the instruction, target error {100 * err:.2f} cm")
        m = masks[-1] if masks else None
        if m is not None:
            img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width)
            inside, total = int(img[150:211, 430:491].astype(bool).sum()), int(img.astype(bool).sum())
        else:
            inside = total = 0
        check("tracker re-grounded with the target's mask", total > 0 and inside == total,
              f"{total} mask pixels, {inside} inside the yellow flower")

        # 2. clarify, answer, cancel
        events.clear(); targets.clear()
        post("api/instruction", {"text": "grasp the flower"})
        publish_dets(0.8)
        q = [e for e in events if e["type"] == "question"]
        check("ambiguous instruction asks back", q and len(q[-1]["options"]) == 2, q[-1]["text"] if q else str(events))
        post("api/instruction", {"text": "the left one"})
        publish_dets(0.8)
        props = [e for e in events if e["type"] == "proposal"]
        post("api/cancel")
        publish_dets(2.5)
        check("answer picks the left one; cancel publishes nothing",
              props and props[-1]["target"]["index"] == 0 and any(e["type"] == "cancelled" for e in events)
              and not targets, f"{len(props)} proposals, {len(targets)} targets")

        # 3. refusal
        says.clear(); events.clear()
        post("api/instruction", {"text": "cut the stem"})
        publish_dets(0.8)
        check("unsupported action refused in plain language", says and "can't cut" in says[-1] and not events,
              says[-1] if says else "no answer")

        # 4. stop from the GUI
        r = post("api/drive_enable", {"enable": True})
        publish_dets(1.0)
        base.clear(); cancels.clear()
        t_stop = time.time()
        res = post("api/abort")
        publish_dets(1.0)
        st = get("api/state")
        zero = [b for b in base if b[0] >= t_stop and b[1] == 0.0 and b[2] == 0.0]
        check("STOP: zero base command, MPC cancelled, servo bridge disabled",
              res.get("stopped") and zero and cancels and st.get("stop_bridge_confirmed"),
              f"{len(zero)} zero /cmd_vel, {len(cancels)} cancels, bridge confirmed {st.get('stop_bridge_confirmed')}")
        check("drive disabled after STOP", r.get("enabled") and st.get("drive_enabled") is False,
              f"drive enabled before {r.get('enabled')}, after {st.get('drive_enabled')}")
    finally:
        for p in procs:                       # ros2 run's child too: the whole process group
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        for p in procs:
            try:
                out, _ = p.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                out = ""
            bad = [ln for ln in (out or "").splitlines() if "Traceback" in ln or "Error" in ln]
            if bad:
                print("info " + " | ".join(bad[:3]), flush=True)
    ok = all(results)
    print("OPERATOR CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
