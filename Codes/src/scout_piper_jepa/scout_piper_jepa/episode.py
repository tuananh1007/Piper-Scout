"""E1 / E3 episodes: rosbag2 -> .npz export and offline tracking evaluation.

Episode file (``.npz``):
  frames       (T, H, W, 3) uint8 RGB
  stamps       (T,) float seconds
  target_masks (T, H, W) bool   — all-False where the target is truly hidden
  distractors  (T, D, H, W) bool — optional, other instances of the class
  depth        (T, H, W) float16 metres, aligned to colour (NaN: none) — optional
  K            (3, 3) colour intrinsics — optional
  T_world_cam  (T, 4, 4) camera optical frame in world_frame (NaN: no TF) — optional
  states       (T, 9) whole-body state [x, y, θ, q1..q6] (NaN: missing) — E3
  gripper      (T,) gripper opening in metres, finger joint 7 − joint 8 (NaN: missing) — optional,
               the Δg of the action embedding
  seg_masks    (T, H, W) bool framewise segmentation from the bag (E1 baseline T0) — optional
  labels       (T, H, W) uint8 merged semantic labels (/stem_grasp/semantic_label) — optional,
               for building the semantic map offline (scout_piper_scene_repr offline.py)

Export a bag recorded with ``scripts/record_bag.sh e1|e3`` (needs ROS 2
``rosbag2_py``); target masks then come from ``scout_piper_jepa.annotate``:
  python3 -m scout_piper_jepa.episode export BAG OUT.npz --rgb /camera/color/image_raw

Evaluate a tracker configuration on episodes:
  python3 -m scout_piper_jepa.episode eval ep1.npz ep2.npz --encoder color_patch
  python3 -m scout_piper_jepa.episode eval ep1.npz --encoder color_patch --no-prior
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import List, Optional

import numpy as np

from .encoder import make_encoder
from .image_codec import image_to_numpy
from .metrics import TrackingScores, score_sequence
from .target_memory import TargetMemory, TargetMemoryConfig


def run_tracker(frames: np.ndarray, init_mask: np.ndarray, encoder, cfg: TargetMemoryConfig,
                clip_length: int = 2) -> List[Optional[np.ndarray]]:
    mem = TargetMemory(cfg)
    clip = [frames[0]]
    mem.initialize(encoder.encode(clip), init_mask)
    preds = []
    for f in frames:
        clip = (clip + [f])[-clip_length:]
        s = mem.update(encoder.encode(clip), image_hw=f.shape[:2])
        preds.append(s.u_mean if s.visible else None)
    return preds


def evaluate_episode(path: str, encoder, cfg: TargetMemoryConfig) -> TrackingScores:
    ep = np.load(path)
    frames, tm = ep["frames"], ep["target_masks"]
    first = int(np.argmax(tm.reshape(len(tm), -1).any(1)))
    preds = run_tracker(frames[first:], tm[first], encoder, cfg)
    gts = [m if m.any() else None for m in tm[first:]]
    dis = [list(d) for d in ep["distractors"][first:]] if "distractors" in ep else ()
    return score_sequence(preds, gts, dis)


def _stamp(msg) -> float:
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def _yaw(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


class EpisodeAssembler:
    """Builds an episode from messages in recorded order (no ROS needed: tests
    feed plain objects). ``tf_lookup(target, source, stamp) -> 4x4 | None``.

    Depth, segmentation and label images are matched to a colour frame by
    header stamp (closest within ``max_dt``), whether they arrive before or
    after it: a segmentation mask carries its colour frame's stamp but is
    recorded one inference later."""

    def __init__(self, rgb_topic: str, depth_topic: str = "", info_topic: str = "", odom_topic: str = "/odom",
                 joints_topic: str = "/joint_states", seg_topic: str = "", label_topic: str = "",
                 joint_names=tuple(f"piper_joint{i}" for i in range(1, 7)), world_frame: str = "odom",
                 camera_frame: str = "", tf_lookup=None, stride: int = 1, max_dt: float = 0.05,
                 finger_joints=("piper_joint7", "piper_joint8")):
        self.t = dict(rgb=rgb_topic, depth=depth_topic, info=info_topic, odom=odom_topic,
                      joints=joints_topic, seg=seg_topic, label=label_topic)
        self.joint_names, self.world, self.cam_frame = list(joint_names), world_frame, camera_frame
        self.finger_joints = list(finger_joints)
        self.gripper = np.nan
        self.tf_lookup, self.stride, self.max_dt = tf_lookup, stride, max_dt
        self.K = None
        self.base = self.q = self.depth = self.seg = self.label = None
        self.rows, self.k = [], 0

    def add(self, topic: str, msg) -> None:
        t = self.t
        if topic == t["info"] and self.K is None:
            self.K = np.array(msg.k, float).reshape(3, 3)
        elif topic == t["depth"]:
            d = image_to_numpy(msg).astype(np.float32)
            self.depth = (_stamp(msg), d / 1000.0 if msg.encoding in ("16UC1", "mono16") else d)
            self._attach("depth", self.depth)
        elif topic == t["seg"]:
            self.seg = (_stamp(msg), image_to_numpy(msg) > 0)
            self._attach("seg", self.seg)
        elif topic == t["label"]:
            self.label = (_stamp(msg), image_to_numpy(msg).astype(np.uint8))
            self._attach("label", self.label)
        elif topic == t["odom"]:
            pz = msg.pose.pose
            self.base = np.array([pz.position.x, pz.position.y, _yaw(pz.orientation)])
        elif topic == t["joints"]:
            idx = {n: i for i, n in enumerate(msg.name)}
            if all(n in idx for n in self.joint_names):
                self.q = np.array([msg.position[idx[n]] for n in self.joint_names])
            if len(self.finger_joints) == 2 and all(n in idx for n in self.finger_joints):
                # each finger moves half the opening (stem_grasp convention)
                self.gripper = float(msg.position[idx[self.finger_joints[0]]] - msg.position[idx[self.finger_joints[1]]])
        elif topic == t["rgb"]:
            if self.k % self.stride == 0:
                self._frame(msg)
            self.k += 1

    def _attach(self, key: str, item) -> None:
        """A late image: give it to the frames it is closer to than their current match."""
        ts = item[0]
        for r in reversed(self.rows):
            if r["stamp"] < ts - self.max_dt:
                break
            dt = abs(r["stamp"] - ts)
            if dt <= self.max_dt and dt < r["dt"][key]:
                r[key], r["dt"][key] = item[1], dt

    def _frame(self, msg) -> None:
        img = image_to_numpy(msg)
        img = img[..., ::-1] if msg.encoding == "bgr8" else img
        ts = _stamp(msg)
        T = None
        if self.tf_lookup is not None:
            T = self.tf_lookup(self.world, self.cam_frame or msg.header.frame_id, ts)
        state = np.r_[self.base, self.q] if self.base is not None and self.q is not None else np.full(9, np.nan)
        row = dict(frame=np.ascontiguousarray(img), stamp=ts, T=T, state=state, gripper=self.gripper,
                   depth=None, seg=None, label=None, dt={"depth": np.inf, "seg": np.inf, "label": np.inf})
        for key in ("depth", "seg", "label"):                # the latest one received before the frame
            item = getattr(self, key)
            if item is not None and abs(item[0] - ts) <= self.max_dt:
                row[key], row["dt"][key] = item[1], abs(item[0] - ts)
        self.rows.append(row)

    def result(self) -> dict:
        if not self.rows:
            raise ValueError(f"no frames on {self.t['rgb']}")
        frames = np.stack([r["frame"] for r in self.rows])
        Tn = len(frames)
        H, W = frames.shape[1:3]
        out = {"frames": frames, "stamps": np.array([r["stamp"] for r in self.rows]),
               "target_masks": np.zeros((Tn, H, W), bool),
               "states": np.stack([r["state"] for r in self.rows])}
        g = np.array([r["gripper"] for r in self.rows], float)
        if np.isfinite(g).any():
            out["gripper"] = g
        if any(r["depth"] is not None for r in self.rows):
            out["depth"] = np.stack([r["depth"] if r["depth"] is not None and r["depth"].shape == (H, W)
                                     else np.full((H, W), np.nan, np.float32) for r in self.rows]).astype(np.float16)
        if any(r["seg"] is not None for r in self.rows):
            out["seg_masks"] = np.stack([r["seg"] if r["seg"] is not None and r["seg"].shape == (H, W)
                                         else np.zeros((H, W), bool) for r in self.rows])
        if any(r["label"] is not None for r in self.rows):
            out["labels"] = np.stack([r["label"] if r["label"] is not None and r["label"].shape == (H, W)
                                      else np.zeros((H, W), np.uint8) for r in self.rows])
        if self.K is not None:
            out["K"] = self.K
        if any(r["T"] is not None for r in self.rows):
            out["T_world_cam"] = np.stack([r["T"] if r["T"] is not None else np.full((4, 4), np.nan)
                                           for r in self.rows])
        return out


def export_bag(bag: str, out: str, rgb_topic: str, stride: int = 1, depth_topic: str = "",
               info_topic: str = "", seg_topic: str = "", world_frame: str = "odom",
               camera_frame: str = "", label_topic: str = "") -> dict:
    """Bag -> episode .npz. Returns a summary (frames, which optional fields)."""
    import rosbag2_py  # noqa: PLC0415 — ROS-only
    import tf2_ros  # noqa: PLC0415
    from rclpy.duration import Duration  # noqa: PLC0415
    from rclpy.serialization import deserialize_message  # noqa: PLC0415
    from rclpy.time import Time  # noqa: PLC0415
    from rosidl_runtime_py.utilities import get_message  # noqa: PLC0415

    def open_reader(topics=None):
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
        if topics:
            r.set_filter(rosbag2_py.StorageFilter(topics=topics))
        return r

    reader = open_reader()
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    buf = tf2_ros.Buffer(cache_time=Duration(seconds=24 * 3600))
    # pass 1: all TF, so every frame can be looked up at its own stamp
    tf_topics = [t for t in ("/tf", "/tf_static") if t in types]
    if tf_topics:
        r = open_reader(tf_topics)
        while r.has_next():
            topic, data, _ = r.read_next()
            for tr in deserialize_message(data, get_message(types[topic])).transforms:
                (buf.set_transform_static if topic == "/tf_static" else buf.set_transform)(tr, "bag")

    def lookup(target, source, ts):
        try:
            tr = buf.lookup_transform(target, source, Time(nanoseconds=int(ts * 1e9))).transform
        except Exception:  # noqa: BLE001 — no TF at that time
            return None
        q, p = tr.rotation, tr.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    asm = EpisodeAssembler(rgb_topic, depth_topic, info_topic, seg_topic=seg_topic, label_topic=label_topic,
                           world_frame=world_frame, camera_frame=camera_frame, tf_lookup=lookup, stride=stride)
    wanted = [t for t in (rgb_topic, depth_topic, info_topic, seg_topic, label_topic, "/odom", "/joint_states")
              if t and t in types]
    # pass 2: everything else in recorded order
    reader = open_reader(wanted)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        asm.add(topic, deserialize_message(data, get_message(types[topic])))
    ep = asm.result()
    np.savez_compressed(out, **ep)
    return {"frames": len(ep["frames"]), "fields": sorted(ep)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("bag"); e.add_argument("out")
    e.add_argument("--rgb", default="/camera/color/image_raw")
    e.add_argument("--stride", type=int, default=1)
    e.add_argument("--depth", default="/camera/aligned_depth_to_color/image_raw", help="'' to skip")
    e.add_argument("--info", default="/camera/color/camera_info")
    e.add_argument("--seg", default="/stem_grasp/target_mask", help="framewise masks for T0 ('' to skip)")
    e.add_argument("--labels", default="/stem_grasp/semantic_label", help="merged labels ('' to skip)")
    e.add_argument("--world-frame", default="odom")
    e.add_argument("--camera-frame", default="", help="default: the image's frame_id")
    v = sub.add_parser("eval")
    v.add_argument("episodes", nargs="+")
    v.add_argument("--encoder", default="color_patch")
    v.add_argument("--hub-entry", default="")
    v.add_argument("--no-prior", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        info = export_bag(a.bag, a.out, a.rgb, a.stride, a.depth, a.info, a.seg, a.world_frame, a.camera_frame,
                          a.labels)
        print(f"wrote {info['frames']} frames to {a.out} with {info['fields']} "
              "(target_masks are empty: annotate with python3 -m scout_piper_jepa.annotate)")
        return
    enc = make_encoder(a.encoder, **({"hub_entry": a.hub_entry} if a.encoder == "vjepa" else {}))
    cfg = TargetMemoryConfig(prior_sigma_px=None) if a.no_prior else TargetMemoryConfig()
    for p in a.episodes:
        print(json.dumps({"episode": p, **asdict(evaluate_episode(p, enc, cfg))}))


if __name__ == "__main__":
    main()
