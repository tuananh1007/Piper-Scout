"""Record one data-collection episode while quickstart.launch.py is running.

    ros2 run piper_quickstart record_episode --name pick_tomato_01
    # ... move the arm with MoveIt ... Ctrl-C to stop
    ros2 run piper_quickstart export_episode data/episodes/<episode>

Before recording it checks that every required stream is live, and refuses to
start otherwise. It then snapshots the static context (camera intrinsics,
hand-eye calibration, URDF) into meta.json and runs `ros2 bag record` (MCAP)
on the raw topics. Nothing is resampled at record time, so the bag keeps
every original message and timestamp. export_episode builds the
frame-aligned dataset from it.

Episode layout:
    <ws>/data/episodes/<YYYYmmdd_HHMMSS>_<name>/
        meta.json   static context + recording summary
        bag/        rosbag2 MCAP with the topics below
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

from piper_quickstart.paths import episodes_dir

COLOR = "/camera/color/image_raw"
COLOR_INFO = "/camera/color/camera_info"
DEPTH = "/camera/aligned_depth_to_color/image_raw"
DEPTH_INFO = "/camera/aligned_depth_to_color/camera_info"
POINTS = "/camera/depth/color/points"

# Always recorded. Measured state, commands, planned trajectories, TF and the
# camera streams needed to reconstruct every frame's pose and intrinsics.
TOPICS: List[str] = [
    "/joint_states",                      # measured, URDF names (from the bridge), 200 Hz
    "/joint_states_single",               # measured, raw driver message
    "/joint_ctrl_single",                 # setpoints streamed to the arm by the bridge
    "/joint_ctrl",                        # setpoints as echoed back by the arm controller
    "/piper_bridge/executed_trajectory",  # every trajectory MoveIt sent for execution
    "/display_planned_path",              # MoveIt plans (including ones never executed)
    "/arm_status",
    "/end_pose_stamped",                  # arm firmware's own FK of the flange
    "/tf",
    "/tf_static",
    "/robot_description",
    "/camera_mount/info",
    "/scene_objects/info",                # static collision boxes in the planning scene
    COLOR,
    COLOR_INFO,
    DEPTH,
    DEPTH_INFO,
    "/camera/depth/image_rect_raw",
    "/camera/depth/camera_info",
    "/camera/extrinsics/depth_to_color",
]

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class _Snapshot(Node):
    """Wait for one message on each required topic and keep it."""

    def __init__(self) -> None:
        super().__init__("record_episode_preflight")
        self.got: Dict[str, object] = {}
        specs = [
            ("/joint_states", JointState, 10),
            (COLOR, Image, qos_profile_sensor_data),
            (DEPTH, Image, qos_profile_sensor_data),
            (COLOR_INFO, CameraInfo, qos_profile_sensor_data),
            (DEPTH_INFO, CameraInfo, qos_profile_sensor_data),
            ("/camera_mount/info", String, LATCHED),
            ("/robot_description", String, LATCHED),
        ]
        for topic, typ, qos in specs:
            self.create_subscription(typ, topic, lambda m, t=topic: self.got.setdefault(t, m), qos)
        self.required = [s[0] for s in specs]


HINTS = {
    "/joint_states": "is quickstart.launch.py running with the arm (arm:=true)?",
    COLOR: "is the camera running (camera:=true)?",
    DEPTH: "aligned depth is off: the launch enables align_depth; is the camera running?",
    COLOR_INFO: "is the camera running?",
    DEPTH_INFO: "is the camera running?",
    "/camera_mount/info": "camera_mount_publisher is not running (it starts with the camera)",
    "/robot_description": "robot_state_publisher is not running",
}


def _snapshot(timeout_s: float) -> Dict[str, object]:
    rclpy.init()
    node = _Snapshot()
    end = time.monotonic() + timeout_s
    while time.monotonic() < end and len(node.got) < len(node.required):
        rclpy.spin_once(node, timeout_sec=0.1)
    missing = [t for t in node.required if t not in node.got]
    got = dict(node.got)
    node.destroy_node()
    rclpy.shutdown()
    if missing:
        lines = "\n".join(f"  - {t}: {HINTS.get(t, '')}" for t in missing)
        raise SystemExit(f"Refusing to record: no data within {timeout_s:.0f}s on\n{lines}")
    return got


def _camera_info_dict(m: CameraInfo) -> Dict:
    return {
        "frame_id": m.header.frame_id,
        "width": m.width,
        "height": m.height,
        "distortion_model": m.distortion_model,
        "K": list(m.k),
        "D": list(m.d),
        "R": list(m.r),
        "P": list(m.p),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="episode", help="short label, used in the folder name")
    ap.add_argument("--task", default="", help="task description stored in meta.json")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <workspace>/data/episodes")
    ap.add_argument("--pointcloud", action="store_true", help=f"also record {POINTS} (large; derivable from depth)")
    ap.add_argument("--wait-s", type=float, default=5.0, help="how long to wait for each stream before refusing")
    ap.add_argument("--max-duration", type=float, default=0.0, help="stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--export", action="store_true", help="run export_episode when recording stops")
    args, _ = ap.parse_known_args(argv)

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.name).strip("_") or "episode"
    root = (args.out_dir or episodes_dir()) / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{name}"

    print("Checking streams...", flush=True)
    snap = _snapshot(args.wait_s)
    mount = json.loads(snap["/camera_mount/info"].data)
    if not mount["calibration"].get("calibrated", False):
        print("\n  WARNING: camera extrinsics are NOMINAL (uncalibrated). Poses in this episode will be off.\n"
              "  Run the hand-eye calibration (calibrate:=true) before collecting real data.\n", flush=True)

    root.mkdir(parents=True, exist_ok=False)
    topics = TOPICS + ([POINTS] if args.pointcloud else [])
    meta = {
        "format": "piper_quickstart/episode@1",
        "name": name,
        "task": args.task,
        "notes": args.notes,
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "bag": "bag",
        "topics": topics,
        "camera": {
            "color": _camera_info_dict(snap[COLOR_INFO]),
            "aligned_depth": _camera_info_dict(snap[DEPTH_INFO]),
            "color_encoding": snap[COLOR].encoding,
            "depth_encoding": snap[DEPTH].encoding,
            "depth_units_m": 0.001,  # realsense2_camera publishes Z16 depth in millimetres
        },
        "camera_mount": mount,
        "joint_names": list(snap["/joint_states"].name),
        "robot_description_urdf": snap["/robot_description"].data,
        "clock": "All header stamps are host wall-clock (RealSense global time, piper_sdk receive time).",
    }
    meta_path = root / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    cmd = ["ros2", "bag", "record", "-s", "mcap", "-o", str(root / "bag"), *topics]
    print(f"Recording to {root}\n  Ctrl-C to stop.", flush=True)
    proc = subprocess.Popen(cmd)
    t0 = time.monotonic()
    try:
        while proc.poll() is None:
            time.sleep(0.2)
            if args.max_duration and time.monotonic() - t0 > args.max_duration:
                proc.send_signal(signal.SIGINT)
                break
    except KeyboardInterrupt:
        pass  # the terminal's SIGINT also reached ros2 bag record
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        proc.wait()

    meta["stopped_at"] = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
    meta["duration_s"] = round(time.monotonic() - t0, 3)
    meta_path.write_text(json.dumps(meta, indent=2))
    if not (root / "bag" / "metadata.yaml").exists():
        print(f"error: the bag in {root / 'bag'} was not finalised (ros2 bag record exited {proc.returncode})",
              file=sys.stderr)
        return 1
    subprocess.run(["ros2", "bag", "info", str(root / "bag")])
    print(f"\nEpisode saved: {root}")
    if args.export:
        from piper_quickstart.export_episode import main as export_main
        return export_main([str(root)])
    print(f"Export the aligned dataset with:\n  ros2 run piper_quickstart export_episode {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
