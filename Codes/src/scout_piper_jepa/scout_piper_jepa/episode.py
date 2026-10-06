"""E1 episodes: rosbag2 -> .npz export and offline tracking evaluation.

Episode file (``.npz``):
  frames       (T, H, W, 3) uint8 RGB
  stamps       (T,) float seconds
  target_masks (T, H, W) bool   — all-False where the target is truly hidden
  distractors  (T, D, H, W) bool — optional, other instances of the class

Export a bag (needs ROS 2 ``rosbag2_py``; masks come from your annotation tool):
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


def export_bag(bag: str, out: str, rgb_topic: str, stride: int = 1) -> int:
    import rosbag2_py  # noqa: PLC0415 — ROS-only
    from rclpy.serialization import deserialize_message  # noqa: PLC0415
    from sensor_msgs.msg import Image  # noqa: PLC0415

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[rgb_topic]))
    frames, stamps, k = [], [], 0
    while reader.has_next():
        _, data, t = reader.read_next()
        if k % stride == 0:
            msg = deserialize_message(data, Image)
            img = image_to_numpy(msg)
            frames.append(img[..., ::-1] if msg.encoding == "bgr8" else img)
            stamps.append(t * 1e-9)
        k += 1
    frames = np.stack(frames)
    np.savez_compressed(out, frames=frames, stamps=np.array(stamps),
                        target_masks=np.zeros(frames.shape[:3], bool))
    return len(frames)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("bag"); e.add_argument("out")
    e.add_argument("--rgb", default="/camera/color/image_raw")
    e.add_argument("--stride", type=int, default=1)
    v = sub.add_parser("eval")
    v.add_argument("episodes", nargs="+")
    v.add_argument("--encoder", default="color_patch")
    v.add_argument("--hub-entry", default="")
    v.add_argument("--no-prior", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(f"wrote {export_bag(a.bag, a.out, a.rgb, a.stride)} frames to {a.out} "
              "(target_masks are empty: annotate before eval)")
        return
    enc = make_encoder(a.encoder, **({"hub_entry": a.hub_entry} if a.encoder == "vjepa" else {}))
    cfg = TargetMemoryConfig(prior_sigma_px=None) if a.no_prior else TargetMemoryConfig()
    for p in a.episodes:
        print(json.dumps({"episode": p, **asdict(evaluate_episode(p, enc, cfg))}))


if __name__ == "__main__":
    main()
