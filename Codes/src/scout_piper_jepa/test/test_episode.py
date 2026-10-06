"""Image codec and offline episode evaluation (no ROS)."""

from types import SimpleNamespace

import numpy as np

from scout_piper_jepa.encoder import ColorPatchEncoder
from scout_piper_jepa.episode import evaluate_episode
from scout_piper_jepa.image_codec import image_to_numpy
from scout_piper_jepa.target_memory import TargetMemoryConfig
from test_target_memory import _scene


def _msg(arr, enc, pad=0, big=False):
    a = np.ascontiguousarray(arr)
    if big:
        a = a.astype(a.dtype.newbyteorder(">"))
    row = a.reshape(a.shape[0], -1).view(np.uint8)
    row = np.pad(row, ((0, 0), (0, pad)))
    return SimpleNamespace(height=a.shape[0], width=a.shape[1], encoding=enc,
                           step=row.shape[1], is_bigendian=int(big), data=row.tobytes())


def test_codec_roundtrip_with_row_padding_and_endianness():
    rgb = np.random.default_rng(0).integers(0, 255, (5, 7, 3), np.uint8)
    assert np.array_equal(image_to_numpy(_msg(rgb, "rgb8", pad=3)), rgb)
    d = np.arange(35, dtype=np.uint16).reshape(5, 7) * 100
    assert np.array_equal(image_to_numpy(_msg(d, "16UC1", pad=2)), d)
    assert np.array_equal(image_to_numpy(_msg(d, "16UC1", big=True)), d)
    f = np.linspace(0, 1, 35, dtype=np.float32).reshape(5, 7)
    assert np.allclose(image_to_numpy(_msg(f, "32FC1")), f)


def test_evaluate_episode_file(tmp_path):
    frames, tm, dm = _scene()
    H, W = frames[0].shape[:2]
    masks = np.stack([m if m is not None else np.zeros((H, W), bool) for m in tm])
    p = tmp_path / "ep.npz"
    np.savez_compressed(p, frames=np.stack(frames), target_masks=masks,
                        distractors=np.stack([np.stack(d) for d in dm]))
    sc = evaluate_episode(str(p), ColorPatchEncoder(16), TargetMemoryConfig())
    assert sc.n_frames == len(frames) and sc.id_retention > 0.9
