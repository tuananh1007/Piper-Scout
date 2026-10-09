"""ROS 2 recorders for the robot calibration (P3A.7 / WE1, P0.2.6).

    ros2 run scout_piper_whole_body_mpc calibrate_slip [--execute] [--pose-topic /mocap/pose --pose-type pose]
    ros2 run scout_piper_whole_body_mpc calibrate_tcp
    ros2 run scout_piper_whole_body_mpc calibrate_hand_eye --board-topic /board_pose
    ros2 run scout_piper_whole_body_mpc calibrate_hand_eye --aruco --marker-length 0.08

``calibrate_slip`` drives the Scout through ``calibration.excitation_plan``
(straight runs, arcs, turns on the spot; about 50 s; it ends near the start)
on /cmd_vel and records the commands and an external pose reference. Without
``--execute`` it only prints the plan. The pose reference must see slip:
motion capture, a fixed camera on an AprilTag on the base, or a total
station; wheel odometry (/odom) cannot, so with /odom the fit returns ≈ 1 and
says nothing. It stops the base (zero twist) at the end, on Ctrl-C and when
the base leaves ``--max-radius`` of its start.

``calibrate_tcp`` and ``calibrate_hand_eye`` capture one sample per Enter
(``q`` + Enter solves): the flange pose from TF (``--base-frame`` →
``--flange-frame``), and for hand-eye the board pose in the camera frame,
either from a detector that publishes geometry_msgs/PoseStamped
(``--board-topic``) or from the built-in single ArUco marker detection
(``--aruco``, needs OpenCV with the aruco module). Results are printed and
saved as JSON (``--out``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from typing import List, Optional

import numpy as np

from .calibration import excitation_plan, hand_eye, identify_slip, pivot_calibration, rpy_from_matrix


def _quat_T(t, q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


# ------------------------------------------------------------------ slip
def slip_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Skid-steer slip identification (WE1)")
    ap.add_argument("--execute", action="store_true", help="drive the base (otherwise print the plan)")
    ap.add_argument("--pose-topic", default="/odom")
    ap.add_argument("--pose-type", choices=["odom", "pose"], default="odom",
                    help="nav_msgs/Odometry or geometry_msgs/PoseStamped")
    ap.add_argument("--v", type=float, default=0.15, help="m/s")
    ap.add_argument("--w", type=float, default=0.4, help="rad/s")
    ap.add_argument("--segment", type=float, default=3.0, help="seconds per motion")
    ap.add_argument("--rate", type=float, default=20.0, help="command rate Hz")
    ap.add_argument("--max-radius", type=float, default=1.5, help="m from the start before aborting")
    ap.add_argument("--out", default="slip_identification.json")
    a = ap.parse_args(argv)
    plan = excitation_plan(a.v, a.w, a.segment)
    total = sum(d for d, _, _ in plan)
    print(f"plan: {len(plan)} segments, {total:.0f} s, |v| ≤ {a.v} m/s, |ω| ≤ {a.w} rad/s")
    for d, v, w in plan:
        print(f"  {d:4.1f} s  v={v:+.2f}  ω={w:+.2f}")
    if not a.execute:
        print("dry run: add --execute to drive the base (area clear, power cut-off in reach)")
        return

    import rclpy  # noqa: PLC0415
    from geometry_msgs.msg import PoseStamped, Twist  # noqa: PLC0415
    from nav_msgs.msg import Odometry  # noqa: PLC0415

    rclpy.init()
    node = rclpy.create_node("calibrate_slip")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    poses: List[tuple] = []

    def on_pose(msg) -> None:
        pz = msg.pose.pose if a.pose_type == "odom" else msg.pose
        poses.append((time.monotonic(), pz.position.x, pz.position.y, _yaw(pz.orientation)))

    node.create_subscription(Odometry if a.pose_type == "odom" else PoseStamped, a.pose_topic, on_pose, 50)
    executor, spin = _spin_in_thread(rclpy, node)
    cmds: List[tuple] = []

    def send(v: float, w: float) -> None:
        tw = Twist()
        tw.linear.x, tw.angular.z = float(v), float(w)
        pub.publish(tw)
        cmds.append((time.monotonic(), v, w))

    aborted = None
    try:
        t_wait = time.monotonic() + 5.0
        while not poses and time.monotonic() < t_wait:
            time.sleep(0.05)
        if not poses:
            raise RuntimeError(f"no pose on {a.pose_topic}")
        start = np.array(poses[-1][1:3])
        for d, v, w in plan:
            t_end = time.monotonic() + d
            while time.monotonic() < t_end:
                send(v, w)
                if np.linalg.norm(np.array(poses[-1][1:3]) - start) > a.max_radius:
                    aborted = f"left {a.max_radius} m radius"
                    raise KeyboardInterrupt
                time.sleep(1.0 / a.rate)
    except KeyboardInterrupt:
        aborted = aborted or "interrupted"
    finally:
        for _ in range(5):
            send(0.0, 0.0)
            time.sleep(0.05)
        time.sleep(0.5)
    P = np.array(poses)
    C = np.array(cmds)
    result = {"aborted": aborted, "pose_topic": a.pose_topic, "n_poses": len(P), "n_commands": len(C)}
    try:
        est, rep = identify_slip(C[:, 0], C[:, 1:], P[:, 0], P[:, 1:])
        result.update(rep)
        print(json.dumps(result, indent=1))
        print("\nwhole_body_mpc.yaml:\n"
              f"    k_v: {est.k_v:.3f}\n    k_omega: {est.k_omega:.3f}")
        if a.pose_topic == "/odom":
            print("note: /odom is wheel odometry; it does not see slip (use an external reference)")
    except ValueError as exc:
        result["error"] = str(exc)
        print(json.dumps(result, indent=1))
    np.savez(a.out.replace(".json", "") + "_raw.npz", poses=P, commands=C)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    _stop(rclpy, node, executor, spin)


def _spin_in_thread(rclpy, node):
    from rclpy.executors import SingleThreadedExecutor  # noqa: PLC0415
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread


def _stop(rclpy, node, executor, thread) -> None:
    executor.shutdown(timeout_sec=1.0)
    thread.join(timeout=2.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


# --------------------------------------------------------- TF sampling
class _Capture:
    """Enter-driven sampling of TF (and optionally a board pose)."""

    def __init__(self, name: str, base: str, flange: str):
        import rclpy  # noqa: PLC0415
        import tf2_ros  # noqa: PLC0415

        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node(name)
        self.buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buf, self.node)
        self.base, self.flange = base, flange
        self.executor, self.spin = _spin_in_thread(rclpy, self.node)

    def flange_pose(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        from rclpy.time import Time  # noqa: PLC0415
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if self.buf.can_transform(self.base, self.flange, Time()):
                tr = self.buf.lookup_transform(self.base, self.flange, Time()).transform
                return _quat_T(tr.translation, tr.rotation)
            time.sleep(0.05)
        return None

    def close(self) -> None:
        _stop(self.rclpy, self.node, self.executor, self.spin)


def _prompt_loop(take) -> None:
    print("Enter: capture a sample   q + Enter: solve and quit")
    for line in sys.stdin:
        if line.strip().lower() == "q":
            return
        take()


def tcp_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="TCP pivot calibration")
    ap.add_argument("--base-frame", default="piper_base_link")
    ap.add_argument("--flange-frame", default="piper_link6")
    ap.add_argument("--out", default="tcp_calibration.json")
    a = ap.parse_args(argv)
    print("Touch one fixed point (a cone tip or a marked spot) with the gripper tip from at least "
          "4 clearly different flange orientations (≥ 30° apart); capture each.")
    cap = _Capture("calibrate_tcp", a.base_frame, a.flange_frame)
    Ts: List[np.ndarray] = []

    def take() -> None:
        T = cap.flange_pose()
        if T is None:
            print(f"  no TF {a.base_frame} -> {a.flange_frame}")
            return
        Ts.append(T)
        print(f"  sample {len(Ts)}: flange at {np.round(T[:3, 3], 4)}")

    try:
        _prompt_loop(take)
    finally:
        cap.close()
    out = {"samples": [T.tolist() for T in Ts]}
    try:
        r = pivot_calibration(Ts)
        out.update({"tcp_in_flange_m": r.tcp_in_flange.tolist(), "pivot_in_base_m": r.pivot_in_base.tolist(),
                    "rms_m": r.rms_m})
        print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1))
        print(f"\ntcp_offset_m (along link6 z): {r.tcp_in_flange[2]:.4f}  "
              f"(off-axis {1000 * np.linalg.norm(r.tcp_in_flange[:2]):.1f} mm)")
    except (ValueError, np.linalg.LinAlgError) as exc:
        out["error"] = str(exc)
        print(f"not solved: {exc}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def _aruco_detector(marker_length: float, dictionary: str):
    import cv2  # noqa: PLC0415
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV without the aruco module: use --board-topic with an external detector")
    aruco = cv2.aruco
    dic = aruco.getPredefinedDictionary(getattr(aruco, dictionary))
    if hasattr(aruco, "ArucoDetector"):
        det = aruco.ArucoDetector(dic, aruco.DetectorParameters())
        detect = det.detectMarkers
    else:
        params = aruco.DetectorParameters_create()
        detect = lambda img: aruco.detectMarkers(img, dic, parameters=params)  # noqa: E731
    h = marker_length / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)

    def pose(gray: np.ndarray, K: np.ndarray, dist: np.ndarray, marker_id: int) -> Optional[np.ndarray]:
        corners, ids, _ = detect(gray)
        if ids is None:
            return None
        for c, i in zip(corners, ids.ravel()):
            if marker_id >= 0 and int(i) != marker_id:
                continue
            ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float32), K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                T = np.eye(4)
                T[:3, :3], _ = cv2.Rodrigues(rvec)
                T[:3, 3] = tvec.ravel()
                return T
        return None
    return pose


def hand_eye_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Eye-in-hand camera calibration")
    ap.add_argument("--base-frame", default="piper_base_link")
    ap.add_argument("--flange-frame", default="piper_link6")
    ap.add_argument("--camera-frame", default="camera_color_optical_frame",
                    help="frame of the board poses; compared with the current TF at the end")
    ap.add_argument("--board-topic", default="", help="geometry_msgs/PoseStamped board pose in the camera frame")
    ap.add_argument("--aruco", action="store_true", help="detect one ArUco marker in the colour image")
    ap.add_argument("--marker-length", type=float, default=0.08, help="m, black square side")
    ap.add_argument("--marker-id", type=int, default=-1, help="-1: first marker found")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--image-topic", default="/camera/color/image_raw")
    ap.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    ap.add_argument("--out", default="hand_eye_calibration.json")
    a = ap.parse_args(argv)
    if bool(a.board_topic) == a.aruco:
        ap.error("choose one of --board-topic or --aruco")
    print("Fix the board/marker in front of the robot. Move the arm to ≥ 8 poses that view it "
          "from different angles, rotating about different axes; hold still and capture each.")
    cap = _Capture("calibrate_hand_eye", a.base_frame, a.flange_frame)
    latest = {"board": None, "img": None, "K": None, "D": None}
    from geometry_msgs.msg import PoseStamped  # noqa: PLC0415
    if a.board_topic:
        cap.node.create_subscription(PoseStamped, a.board_topic,
                                     lambda m: latest.__setitem__("board", (time.monotonic(),
                                                                            _quat_T(m.pose.position,
                                                                                    m.pose.orientation))), 5)
    else:
        from sensor_msgs.msg import CameraInfo, Image  # noqa: PLC0415
        detect = _aruco_detector(a.marker_length, a.dictionary)

        def on_info(m) -> None:
            latest["K"] = np.array(m.k, float).reshape(3, 3)
            latest["D"] = np.array(m.d, float) if len(m.d) else np.zeros(5)

        def on_img(m) -> None:
            arr = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)
            ch = 3 if m.encoding in ("rgb8", "bgr8") else 1
            img = arr[:, :m.width * ch].reshape(m.height, m.width, ch)
            latest["img"] = img.mean(-1).astype(np.uint8) if ch == 3 else img[..., 0]

        cap.node.create_subscription(CameraInfo, a.camera_info_topic, on_info, 2)
        cap.node.create_subscription(Image, a.image_topic, on_img, 2)
    E: List[np.ndarray] = []
    C: List[np.ndarray] = []

    def take() -> None:
        T = cap.flange_pose()
        if T is None:
            print(f"  no TF {a.base_frame} -> {a.flange_frame}")
            return
        if a.board_topic:
            b = latest["board"]
            if b is None or time.monotonic() - b[0] > 0.5:
                print("  no fresh board pose")
                return
            B = b[1]
        else:
            if latest["img"] is None or latest["K"] is None:
                print("  no image / camera info yet")
                return
            B = detect(latest["img"], latest["K"], latest["D"], a.marker_id)
            if B is None:
                print("  marker not found")
                return
        E.append(T)
        C.append(B)
        print(f"  sample {len(E)}: board at {np.round(B[:3, 3], 3)} m from the camera")

    try:
        _prompt_loop(take)
    finally:
        current = None
        if E:
            buf_T = None
            try:
                from rclpy.time import Time  # noqa: PLC0415
                if cap.buf.can_transform(a.flange_frame, a.camera_frame, Time()):
                    tr = cap.buf.lookup_transform(a.flange_frame, a.camera_frame, Time()).transform
                    buf_T = _quat_T(tr.translation, tr.rotation)
            except Exception:  # noqa: BLE001
                buf_T = None
            current = buf_T
        cap.close()
    out = {"flange_poses": [T.tolist() for T in E], "board_poses": [T.tolist() for T in C]}
    try:
        r = hand_eye(E, C)
        X = r.T_flange_camera
        rpy = rpy_from_matrix(X[:3, :3])
        out.update({"T_flange_camera": X.tolist(), "rot_residual_deg": r.rot_residual_deg,
                    "trans_residual_m": r.trans_residual_m})
        print(json.dumps({k: out[k] for k in ("rot_residual_deg", "trans_residual_m")}, indent=1))
        print(f"\n{a.flange_frame} -> {a.camera_frame}:\n"
              f'  <origin xyz="{X[0, 3]:.4f} {X[1, 3]:.4f} {X[2, 3]:.4f}" '
              f'rpy="{rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}"/>')
        if current is not None:
            d = np.linalg.norm(current[:3, 3] - X[:3, 3])
            ang = np.degrees(np.arccos(np.clip((np.trace(current[:3, :3].T @ X[:3, :3]) - 1) / 2, -1, 1)))
            out["difference_to_current_tf"] = {"translation_m": float(d), "rotation_deg": float(ang)}
            print(f"differs from the current TF by {100 * d:.1f} cm and {ang:.1f} deg")
    except (ValueError, np.linalg.LinAlgError) as exc:
        out["error"] = str(exc)
        print(f"not solved: {exc}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
