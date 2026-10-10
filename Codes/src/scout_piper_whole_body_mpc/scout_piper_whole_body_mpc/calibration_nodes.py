"""ROS 2 recorders for the robot calibration (P3A.7 / WE1, P0.2.6).

    ros2 run scout_piper_whole_body_mpc calibrate_slip [--execute] [--pose-topic /mocap/pose --pose-type pose]
    ros2 run scout_piper_whole_body_mpc calibrate_tcp
    ros2 run scout_piper_whole_body_mpc calibrate_hand_eye --board-topic /board_pose
    ros2 run scout_piper_whole_body_mpc calibrate_hand_eye --aruco --marker-length 0.08
    ros2 run scout_piper_whole_body_mpc calibrate_effort [--execute | --record-only] [--center q1,..,q6]

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

``calibrate_effort`` fits the joint-effort model of the contact-force
estimate (dynamics/effort.py, effort_force_node) on free motion: nothing may
touch the arm or the gripper. With ``--execute`` it moves the arm through
``effort_excitation`` (a slow multi-sine, about 0.3 rad/s at most, starting and
ending at its centre) by JointJog on moveit_servo (``--jog-topic``; servo
started and piper_servo_bridge enabled, as for the MPC), after checking the
motion's clearance above the Scout's top plate. The centre is the current
pose, or ``--center``, which the arm first moves to on a straight joint path
(checked as well). It stops the arm at the end, on Ctrl-C and when the arm
stops following. ``--record-only`` records while another tool moves the arm.
Without either it only checks the motion around ``--center``. The fit uses the samples slower than ``--max-speed`` and
reports a held-out residual (last quarter of the run) and the force noise
at the start pose and at the servo pose; ``--out`` is the
effort_force_node ``calibration_file``.
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

SERVO_POSE = (0.0, 1.2, -1.0, 0.0, 0.6, 0.0)          # typical pose while servoing on a stem
EFFORT_CENTER = (0.0, 0.8, -1.2, 0.0, 0.45, 0.0)      # clears the top plate and the joint limits


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


# ------------------------------------------------------------------ effort
def _parse_q(text: str) -> np.ndarray:
    q = np.array([float(v) for v in text.split(",")])
    if q.shape != (6,):
        raise ValueError(f"need 6 joint values, got {text!r}")
    return q


def _clearance_problem(chk: dict, min_clearance: float) -> Optional[str]:
    if not chk["within_limits"]:
        return f"comes within {chk['min_limit_margin_rad']:.2f} rad of a joint limit (servo halts at 0.1)"
    if chk["min_deck_clearance_m"] < min_clearance:
        return (f"comes within {100 * chk['min_deck_clearance_m']:.1f} cm of the top plate "
                f"(< {100 * min_clearance:.0f} cm)")
    return None


def _check_effort_motion(q0: np.ndarray, duration: float, kin, min_clearance: float,
                         q_start: Optional[np.ndarray] = None, speed: float = 0.15) -> Optional[str]:
    """Problem with the calibration motion around q0 (and the move there from
    ``q_start``), or None."""
    from .dynamics.effort import excitation_check, joint_move, motion_clearance  # noqa: PLC0415
    if q_start is not None:
        T = joint_move(q_start, q0, 0.0, speed)[2]
        path = motion_clearance([joint_move(q_start, q0, t, speed)[0] for t in np.linspace(0.0, T, 50)], kin)
        print(f"move from q = {np.round(q_start, 3).tolist()} ({T:.0f} s): {json.dumps(path)}")
        why = _clearance_problem(path, min_clearance)
        if why:
            return "move to the centre " + why
    chk = excitation_check(q0, duration, kin)
    print(f"motion around q = {np.round(q0, 3).tolist()}: {json.dumps(chk)}")
    return _clearance_problem(chk, min_clearance)


def fit_effort_report(kin, Q, QD, TAU, payload_kg: float = 0.0, holdout: float = 0.25):
    """Fit on all samples; refit without the last ``holdout`` fraction to
    report the residual on unseen poses; force noise at the servo pose."""
    from .dynamics.effort import ContactForceEstimator, fit_effort_model  # noqa: PLC0415
    model = fit_effort_model(kin, Q, QD, TAU, payload_kg=payload_kg)
    n = int(len(Q) * (1.0 - holdout))
    rep = {"samples": int(len(Q)), "identified": model.identified, "a": np.round(model.a, 4).tolist(),
           "sigma_nm": np.round(model.sigma, 4).tolist()}
    if n >= 50 and len(Q) - n >= 20:
        train = fit_effort_model(kin, Q[:n], QD[:n], TAU[:n], payload_kg=payload_kg)
        res = TAU[n:] - train.free_torque(kin, Q[n:], QD[n:])
        rep["holdout_sigma_nm"] = np.round(np.std(res, axis=0), 4).tolist()
        rep["holdout_bias_nm"] = np.round(np.mean(res, axis=0), 4).tolist()
    noise = ContactForceEstimator(kin, model).force_noise(np.array(SERVO_POSE))
    rep["force_noise_servo_pose_n"] = np.round(noise, 3).tolist()
    rep["threshold_3sigma_n"] = round(3.0 * float(np.linalg.norm(noise)), 3)
    return model, rep


def effort_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Joint-effort model for the contact-force estimate")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="move the arm (moveit_servo JointJog)")
    mode.add_argument("--record-only", action="store_true", help="record while another tool moves the arm")
    ap.add_argument("--center", default=None,
                    help="q1,..,q6 centre of the motion (default: the current pose; without --execute "
                         + ",".join(str(v) for v in EFFORT_CENTER) + ")")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds")
    ap.add_argument("--joint-topic", default="/joint_states")
    ap.add_argument("--joint-names", default=",".join(f"piper_joint{i}" for i in range(1, 7)))
    ap.add_argument("--jog-topic", default="/servo_node/delta_joint_cmds")
    ap.add_argument("--rate", type=float, default=50.0, help="command rate Hz")
    ap.add_argument("--gain", type=float, default=2.0, help="1/s, feedback onto the motion")
    ap.add_argument("--qd-max", type=float, default=0.3, help="rad/s per joint")
    ap.add_argument("--move-speed", type=float, default=0.15, help="rad/s peak for the move to --center")
    ap.add_argument("--max-tracking-error", type=float, default=0.25, help="rad before aborting")
    ap.add_argument("--min-clearance", type=float, default=0.05, help="m above the top plate")
    ap.add_argument("--max-speed", type=float, default=0.5, help="rad/s; faster samples are not fitted")
    ap.add_argument("--tcp-offset", type=float, default=0.14)
    ap.add_argument("--payload-kg", type=float, default=0.0,
                    help="mass on link6 that the URDF inertials lack (camera, mount)")
    ap.add_argument("--out", default="effort_calibration.json")
    a = ap.parse_args(argv)
    from .dynamics.effort import effort_excitation, joint_move  # noqa: PLC0415
    from .dynamics.piper import PiperKinematics  # noqa: PLC0415
    kin = PiperKinematics(tcp_offset_m=a.tcp_offset)
    center = _parse_q(a.center) if a.center else None
    if not (a.execute or a.record_only):
        why = _check_effort_motion(center if center is not None else np.array(EFFORT_CENTER), a.duration, kin,
                                   a.min_clearance)
        print(f"not safe from this pose: {why}" if why else "clear from this pose")
        print("dry run: --execute moves the arm (nothing near it, stop in reach)")
        return 1 if why else 0

    import rclpy  # noqa: PLC0415
    from control_msgs.msg import JointJog  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415

    names = [n.strip() for n in a.joint_names.split(",")]
    rclpy.init()
    node = rclpy.create_node("calibrate_effort")
    pub = node.create_publisher(JointJog, a.jog_topic, 10)
    rows: List[tuple] = []
    no_effort = []

    def on_joints(msg) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in names):
            return
        if len(msg.effort) != len(msg.name):
            no_effort.append(1)
            return
        sel = [idx[n] for n in names]
        vel = [msg.velocity[i] for i in sel] if len(msg.velocity) == len(msg.name) else [0.0] * 6
        rows.append((time.monotonic(), [msg.position[i] for i in sel], vel, [msg.effort[i] for i in sel]))

    node.create_subscription(JointState, a.joint_topic, on_joints, 100)
    executor, spin = _spin_in_thread(rclpy, node)

    def send(qd) -> None:
        jog = JointJog()
        jog.header.stamp = node.get_clock().now().to_msg()
        jog.header.frame_id = "calibrate_effort"
        jog.joint_names = names
        jog.velocities = [float(v) for v in qd]
        pub.publish(jog)

    aborted = None
    i0 = 0
    try:
        t_wait = time.monotonic() + 5.0
        while not rows and time.monotonic() < t_wait:
            time.sleep(0.05)
        if not rows:
            raise RuntimeError(f"no joint states with effort on {a.joint_topic}"
                               + (" (messages carry no effort)" if no_effort else ""))
        q_now = np.array(rows[-1][1])
        q0 = q_now if center is None else center
        if a.execute:
            why = _check_effort_motion(q0, a.duration, kin, a.min_clearance,
                                       None if center is None else q_now, a.move_speed)
            if why:
                raise RuntimeError(f"motion {why}; choose another --center (e.g. "
                                   + ",".join(str(v) for v in EFFORT_CENTER) + ")")

        def follow(path, duration: float) -> Optional[str]:
            """Track path(t) -> (q, q̇) by JointJog with position feedback."""
            t0 = time.monotonic()
            while time.monotonic() - t0 < duration:
                if time.monotonic() - rows[-1][0] > 0.5:
                    return f"no joint states on {a.joint_topic} for 0.5 s"
                q_des, qd_des = path(time.monotonic() - t0)
                err = q_des - np.array(rows[-1][1])
                if time.monotonic() - t0 > 5.0 and np.abs(err).max() > a.max_tracking_error:
                    return (f"arm {np.abs(err).max():.2f} rad off the motion: not following "
                            "(servo started, bridge enabled?)")
                send(np.clip(qd_des + a.gain * err, -a.qd_max, a.qd_max))
                time.sleep(1.0 / a.rate)
            return None

        if a.execute and center is not None:
            T = joint_move(q_now, q0, 0.0, a.move_speed)[2]
            print(f"moving to the centre ({T:.0f} s; Ctrl-C stops the arm)")
            aborted = follow(lambda t: joint_move(q_now, q0, t, a.move_speed)[:2], T + 1.0)
        i0 = len(rows)
        if aborted is None and a.execute:
            print(f"calibration motion for {a.duration:.0f} s (Ctrl-C stops the arm)")
            aborted = follow(lambda t: effort_excitation(q0, t, a.duration), a.duration)
        elif aborted is None:
            print(f"recording for {a.duration:.0f} s: move the arm slowly through its range, nothing touching it")
            time.sleep(a.duration)
    except KeyboardInterrupt:
        aborted = "interrupted"
    except RuntimeError as exc:
        aborted = str(exc)
    finally:
        if a.execute:
            for _ in range(5):
                send(np.zeros(6))
                time.sleep(0.05)
    _stop(rclpy, node, executor, spin)
    result = {"aborted": aborted, "joint_topic": a.joint_topic, "duration_s": a.duration}
    rows = rows[i0:]                                  # the calibration motion only
    if rows:
        T = np.array([r[0] for r in rows])
        Q, QD, TAU = (np.array([r[k] for r in rows], float) for k in (1, 2, 3))
        np.savez(a.out.replace(".json", "") + "_raw.npz", t=T, q=Q, qd=QD, effort=TAU)
        slow = np.max(np.abs(QD), axis=1) <= a.max_speed
        try:
            model, rep = fit_effort_report(kin, Q[slow], QD[slow], TAU[slow], a.payload_kg)
            result.update(rep)
            if aborted is None:
                model.save(a.out)
                with open(a.out, encoding="utf-8") as f:
                    saved = json.load(f)
                saved["report"] = result
                with open(a.out, "w", encoding="utf-8") as f:
                    json.dump(saved, f, indent=1)
                print(json.dumps(result, indent=1))
                print(f"\nsaved {a.out}: effort_force_node calibration_file; set stem_grasp contact_threshold_n "
                      f"and max_force_n above {rep['threshold_3sigma_n']} N (3-σ at the servo pose)")
                return 0
        except (ValueError, np.linalg.LinAlgError) as exc:
            result["error"] = str(exc)
    failed = a.out.replace(".json", "") + "_failed.json"
    print(json.dumps(result, indent=1))
    print(f"not calibrated ({aborted or result.get('error', 'no samples')}): report in {failed}")
    with open(failed, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    return 1


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
