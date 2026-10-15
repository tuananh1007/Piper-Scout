"""Turn a recorded episode bag into a frame-aligned dataset.

    ros2 run piper_quickstart export_episode data/episodes/<episode>

For every color frame it writes the image, the aligned depth image, the arm's
joint positions at the frame's timestamp (linearly interpolated from the
200 Hz measured stream), and the camera pose in the arm base frame at that
same timestamp (from the recorded TF: forward kinematics plus the hand-eye
mount). The full-rate trajectory and every executed MoveIt trajectory are
exported alongside, so nothing is lost by the per-frame resampling.

Output (in <episode>/export/ by default):
    color/000000.png ...      RGB frames (PNG, lossless; --jpg for JPEG)
    depth/000000.png ...      aligned depth, uint16, millimetres
    frames.csv                one row per color frame (see columns below)
    joints.csv                full-rate measured joint states
    commands.csv              setpoints streamed to the arm
    trajectories.json         MoveIt trajectories as executed (absolute start times)
    camera.json               intrinsics (color, aligned depth), depth units,
                              extrinsics (flange->camera, depth->color), calibration provenance
    episode.npz               the same per-frame data as arrays (+ 4x4 poses)

frames.csv columns:
    frame, stamp_ns, color_file, depth_file, depth_dt_ms, joint_gap_ms,
    joint1..joint7, gripper_opening_m,
    cam_x, cam_y, cam_z, cam_qx, cam_qy, cam_qz, cam_qw   (T_base_camera_optical)
    ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw           (T_base_flange)
    pose_ok
``joint_gap_ms`` is the distance to the nearest measured joint sample; large
values mean the arm stream had a gap there and the interpolation is less
trustworthy. ``depth_dt_ms`` is |depth stamp - color stamp|.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rosbag2_py
import yaml
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer

from piper_quickstart.trajectory import ARM_JOINTS, GRIPPER_JOINT
from piper_quickstart.transforms import make_transform, split

COLOR = "/camera/color/image_raw"
DEPTH = "/camera/aligned_depth_to_color/image_raw"
JOINTS = ARM_JOINTS + [GRIPPER_JOINT]


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _reader(bag: Path, topics: Optional[List[str]] = None):
    meta = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=meta["storage_identifier"]),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if topics is not None:
        r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    while r.has_next():
        topic, data, _t = r.read_next()
        yield topic, deserialize_message(data, get_message(types[topic]))


def decode_image(msg) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
    if enc in ("16uc1", "mono16"):
        dt = np.dtype(">u2" if msg.is_bigendian else "<u2")
        img = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 2)[:, : msg.width]
        return img.astype(np.uint16)
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def interp_joints(stamps: np.ndarray, pos: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate (N, J) joint samples at times t; NaN outside the sampled range.

    Also returns the gap (ns) from each t to its nearest sample.
    """
    out = np.full((len(t), pos.shape[1]), np.nan)
    if len(stamps) == 0:
        return out, np.full(len(t), np.inf)
    inside = (t >= stamps[0]) & (t <= stamps[-1])
    for j in range(pos.shape[1]):
        out[inside, j] = np.interp(t[inside], stamps, pos[:, j])
    idx = np.searchsorted(stamps, t)
    left = np.clip(idx - 1, 0, len(stamps) - 1)
    right = np.clip(idx, 0, len(stamps) - 1)
    gap = np.minimum(np.abs(t - stamps[left]), np.abs(stamps[right] - t)).astype(float)
    return out, gap


def _tf_matrix(buf: Buffer, target: str, source: str, t_ns: Optional[int]) -> Optional[np.ndarray]:
    try:
        when = Time() if t_ns is None else Time(nanoseconds=int(t_ns))
        tf = buf.lookup_transform(target, source, when)
    except Exception:
        return None
    tr, q = tf.transform.translation, tf.transform.rotation
    return make_transform([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w])


def _pose7(m: Optional[np.ndarray]) -> List[float]:
    if m is None:
        return [float("nan")] * 7
    t, q = split(m)
    return [*t.tolist(), *q.tolist()]


def export(episode: Path, out: Path, base_frame: str, flange_frame: str, jpg: bool) -> Dict:
    bag = episode / "bag"
    meta = json.loads((episode / "meta.json").read_text()) if (episode / "meta.json").exists() else {}

    # ---- pass 1: everything except images --------------------------------
    js_t, js_p, js_rows = [], [], []
    cmd_rows, trajs, infos, extrinsics, mount = [], [], {}, None, None
    tf_msgs, tf_static = [], []
    for topic, msg in _reader(bag, ["/joint_states", "/joint_ctrl_single", "/piper_bridge/executed_trajectory",
                                    "/tf", "/tf_static", "/camera/color/camera_info",
                                    "/camera/aligned_depth_to_color/camera_info",
                                    "/camera/extrinsics/depth_to_color", "/camera_mount/info"]):
        if topic == "/joint_states":
            d = dict(zip(msg.name, msg.position))
            if all(j in d for j in JOINTS):
                t = stamp_ns(msg.header.stamp)
                js_t.append(t)
                js_p.append([d[j] for j in JOINTS])
                v = dict(zip(msg.name, msg.velocity)) if len(msg.velocity) == len(msg.name) else {}
                e = dict(zip(msg.name, msg.effort)) if len(msg.effort) == len(msg.name) else {}
                js_rows.append([t, *[d[j] for j in JOINTS], *[v.get(j, np.nan) for j in JOINTS],
                                *[e.get(j, np.nan) for j in JOINTS]])
        elif topic == "/joint_ctrl_single":
            cmd_rows.append([stamp_ns(msg.header.stamp), *list(msg.position)[:7]])
        elif topic == "/piper_bridge/executed_trajectory":
            start = stamp_ns(msg.header.stamp)
            trajs.append({
                "controller": msg.header.frame_id,
                "start_stamp_ns": start,
                "joint_names": list(msg.joint_names),
                "points": [{
                    "t_from_start_s": p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
                    "positions": list(p.positions),
                    "velocities": list(p.velocities),
                    "accelerations": list(p.accelerations),
                } for p in msg.points],
            })
        elif topic == "/tf":
            tf_msgs.extend(msg.transforms)
        elif topic == "/tf_static":
            tf_static.extend(msg.transforms)
        elif topic.endswith("camera_info"):
            infos.setdefault(topic, msg)
        elif topic == "/camera/extrinsics/depth_to_color" and extrinsics is None:
            extrinsics = {"rotation_row_major": list(msg.rotation), "translation": list(msg.translation)}
        elif topic == "/camera_mount/info" and mount is None:
            mount = json.loads(msg.data)

    if not js_t:
        raise SystemExit(f"{bag}: no /joint_states with {JOINTS}; was the arm running?")
    order = np.argsort(js_t)
    js_t = np.asarray(js_t, dtype=np.int64)[order]
    js_p = np.asarray(js_p, dtype=float)[order]

    span = (max(t.header.stamp.sec for t in tf_msgs) - min(t.header.stamp.sec for t in tf_msgs) + 60) if tf_msgs else 60
    buf = Buffer(cache_time=Duration(seconds=span))
    for t in tf_static:
        buf.set_transform_static(t, "bag")
    for t in tf_msgs:
        buf.set_transform(t, "bag")

    color_info = infos.get("/camera/color/camera_info")
    cam_frame = color_info.header.frame_id if color_info else "camera_color_optical_frame"

    # ---- pass 2: images, streamed to disk ----------------------------------
    # Depth first, saved under its stamp; then each color frame takes the
    # nearest depth frame. Keeps memory flat however long the episode is.
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    raw_depth = out / "depth" / ".by_stamp"
    raw_depth.mkdir(exist_ok=True)
    dstamps = []
    for _topic, msg in _reader(bag, [DEPTH]):
        t = stamp_ns(msg.header.stamp)
        cv2.imwrite(str(raw_depth / f"{t}.png"), decode_image(msg))
        dstamps.append(t)
    dstamps = np.asarray(sorted(dstamps), dtype=np.int64)

    ext = ".jpg" if jpg else ".png"
    rows, cam_T, ee_T, c_t = [], [], [], []
    for _topic, msg in _reader(bag, [COLOR]):
        c_t.append(stamp_ns(msg.header.stamp))
    if not c_t:
        raise SystemExit(f"{bag}: no color frames on {COLOR}")
    c_t = np.asarray(c_t, dtype=np.int64)
    q, gap = interp_joints(js_t, js_p, c_t)

    for i, (_topic, msg) in enumerate(_reader(bag, [COLOR])):
        t = int(c_t[i])
        cfile = f"color/{i:06d}{ext}"
        cv2.imwrite(str(out / cfile), decode_image(msg), [cv2.IMWRITE_JPEG_QUALITY, 95] if jpg else [])
        dfile, ddt = "", float("nan")
        if len(dstamps):
            k = int(np.argmin(np.abs(dstamps - t)))
            ddt = abs(int(dstamps[k]) - t) / 1e6
            if ddt < 20.0:
                dfile = f"depth/{i:06d}.png"
                shutil.copyfile(raw_depth / f"{int(dstamps[k])}.png", out / dfile)
        m_cam = _tf_matrix(buf, base_frame, cam_frame, t)
        m_ee = _tf_matrix(buf, base_frame, flange_frame, t)
        cam_T.append(m_cam if m_cam is not None else np.full((4, 4), np.nan))
        ee_T.append(m_ee if m_ee is not None else np.full((4, 4), np.nan))
        rows.append([i, t, cfile, dfile, round(ddt, 3), round(gap[i] / 1e6, 3), *q[i].tolist(),
                     q[i][-1] * 2.0, *_pose7(m_cam), *_pose7(m_ee), int(m_cam is not None and m_ee is not None)])
    shutil.rmtree(raw_depth)

    header = (["frame", "stamp_ns", "color_file", "depth_file", "depth_dt_ms", "joint_gap_ms"] + JOINTS
              + ["gripper_opening_m"] + [f"cam_{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]
              + [f"ee_{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw")] + ["pose_ok"])
    _write_csv(out / "frames.csv", header, rows)
    _write_csv(out / "joints.csv",
               ["stamp_ns"] + JOINTS + [f"vel_{j}" for j in JOINTS] + [f"effort_{j}" for j in JOINTS],
               [js_rows[i] for i in order])
    _write_csv(out / "commands.csv", ["stamp_ns"] + JOINTS[: 6] + ["joint7_cmd"], sorted(cmd_rows))
    (out / "trajectories.json").write_text(json.dumps(trajs, indent=1))

    t_flange_cam = _tf_matrix(buf, flange_frame, cam_frame, None)
    depth_frame = infos.get("/camera/aligned_depth_to_color/camera_info")
    camera = {
        "base_frame": base_frame,
        "flange_frame": flange_frame,
        "camera_frame": cam_frame,
        "color": meta.get("camera", {}).get("color") or _info(color_info),
        "aligned_depth": meta.get("camera", {}).get("aligned_depth") or _info(depth_frame),
        "depth_units_m": 0.001,
        "T_flange_camera": None if t_flange_cam is None else t_flange_cam.tolist(),
        "T_depth_to_color_realsense": extrinsics,
        "camera_mount": mount or meta.get("camera_mount"),
        "note": "frames.csv cam_* is T_base_camera (pose of the color optical frame in base_frame); "
                "OpenCV pinhole: u = fx*X/Z + cx, v = fy*Y/Z + cy in camera_frame.",
    }
    (out / "camera.json").write_text(json.dumps(camera, indent=2))
    np.savez_compressed(out / "episode.npz", stamp_ns=c_t, joints=q, joint_names=np.array(JOINTS),
                        joint_gap_ns=gap, T_base_camera=np.asarray(cam_T), T_base_flange=np.asarray(ee_T),
                        K=np.asarray(camera["color"]["K"]).reshape(3, 3) if camera["color"] else np.full((3, 3), np.nan),
                        joints_full_stamp_ns=js_t, joints_full=js_p)

    pose_ok = sum(r[-1] for r in rows)
    summary = {
        "frames": len(rows),
        "frames_with_depth": sum(1 for r in rows if r[3]),
        "frames_with_pose": pose_ok,
        "frames_with_joints": int(np.isfinite(q[:, 0]).sum()),
        # Frames before the first / after the last joint sample have NaN joints and
        # are excluded here; this is the worst gap *inside* the arm stream.
        "max_joint_gap_ms": float(gap[np.isfinite(q[:, 0])].max() / 1e6) if np.isfinite(q[:, 0]).any() else None,
        "joint_samples": int(len(js_t)),
        "executed_trajectories": len(trajs),
        "camera_calibrated": bool((mount or {}).get("calibration", {}).get("calibrated", False)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _info(m) -> Optional[Dict]:
    if m is None:
        return None
    return {"frame_id": m.header.frame_id, "width": m.width, "height": m.height,
            "distortion_model": m.distortion_model, "K": list(m.k), "D": list(m.d), "R": list(m.r), "P": list(m.p)}


def _write_csv(path: Path, header: List[str], rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", type=Path, help="episode folder (contains bag/ and meta.json)")
    ap.add_argument("-o", "--out", type=Path, default=None, help="default: <episode>/export")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--flange-frame", default="link6")
    ap.add_argument("--jpg", action="store_true", help="write color frames as JPEG (smaller, lossy)")
    args, _ = ap.parse_known_args(argv)

    if not (args.episode / "bag" / "metadata.yaml").exists():
        print(f"error: {args.episode / 'bag'} is not a finished rosbag2 recording", file=sys.stderr)
        return 1
    out = args.out or args.episode / "export"
    summary = export(args.episode, out, args.base_frame, args.flange_frame, args.jpg)
    print(json.dumps(summary, indent=2))
    if summary["frames_with_pose"] < summary["frames"]:
        print("warning: some frames have no camera pose (TF missing at that time); see pose_ok in frames.csv")
    if not summary["camera_calibrated"]:
        print("warning: camera extrinsics were NOMINAL (uncalibrated) for this episode")
    print(f"exported to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
