"""Dense video encoders behind one interface.

Every encoder maps a short clip of RGB frames to a dense feature grid for the
*last* frame: ``encode(clip) -> (Hf, Wf, C)`` float32, plus the patch size so
callers can map between pixels and tokens.

* ``ColorPatchEncoder`` — deterministic numpy baseline (per-patch colour and
  gradient statistics). Used by the tests and as a cheap non-learned
  reference; it is **not** a stand-in for V-JEPA results.
* ``VJepaEncoder`` — V-JEPA 2 / 2.1 through ``torch.hub``. torch is imported
  lazily so the rest of the package works without it. The hub repository,
  entry point and checkpoint are configuration, because their exact names
  must be checked against the release being deployed.
* ``DinoV2Encoder`` — DINOv2 patch features through ``torch.hub``: the strong
  dense self-supervised image baseline T2 of the E1 comparison (per frame, no
  temporal context).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np


class DenseEncoder(Protocol):
    patch_size: int

    def encode(self, clip: Sequence[np.ndarray]) -> np.ndarray:
        """clip: list of HxWx3 uint8 RGB frames (oldest first) -> (Hf, Wf, C)."""


def _patchify(img: np.ndarray, p: int) -> np.ndarray:
    """HxWxC -> (H//p, W//p, p, p, C), cropping any remainder."""
    H, W = img.shape[:2]
    h, w = H // p, W // p
    x = img[: h * p, : w * p]
    return x.reshape(h, p, w, p, -1).transpose(0, 2, 1, 3, 4)


@dataclass
class ColorPatchEncoder:
    """Per-patch mean/std colour (normalised RGB + chromaticity) and mean
    gradient magnitude, with a temporal-difference channel from the clip.
    Features are L2-normalised so cosine similarity is a dot product."""

    patch_size: int = 16

    def encode(self, clip: Sequence[np.ndarray]) -> np.ndarray:
        frame = np.asarray(clip[-1], dtype=np.float32) / 255.0
        p = self.patch_size
        P = _patchify(frame, p)                                # h,w,p,p,3
        mean = P.mean(axis=(2, 3))
        std = P.std(axis=(2, 3))
        s = frame.sum(-1, keepdims=True) + 1e-6
        chrom = _patchify(frame / s, p).mean(axis=(2, 3))[..., :2]
        gray = frame.mean(-1)
        gy, gx = np.gradient(gray)
        grad = _patchify(np.hypot(gx, gy)[..., None], p).mean(axis=(2, 3))
        if len(clip) > 1:
            prev = np.asarray(clip[-2], dtype=np.float32).mean(-1) / 255.0
            motion = _patchify(np.abs(gray - prev)[..., None], p).mean(axis=(2, 3))
        else:
            motion = np.zeros_like(grad)
        f = np.concatenate([mean - mean.mean((0, 1)), std, 3.0 * chrom - 1.0,
                            grad, motion], axis=-1)
        n = np.linalg.norm(f, axis=-1, keepdims=True)
        return (f / np.maximum(n, 1e-6)).astype(np.float32)


@dataclass
class VJepaEncoder:
    """V-JEPA 2 / 2.1 dense encoder (lazy torch).

    ``hub_repo`` / ``hub_entry`` follow ``torch.hub.load(hub_repo, hub_entry)``.
    Verify both against the official ``facebookresearch/vjepa2`` README for the
    checkpoint you deploy; they are deliberately not hard-coded.
    """

    hub_repo: str = "facebookresearch/vjepa2"
    hub_entry: str = ""                  # e.g. the ViT-B/16 384 entry point
    image_size: int = 384
    patch_size: int = 16
    tubelet: int = 2
    device: str = "cuda"
    fp16: bool = True
    _model: Optional[object] = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.hub_entry:
            raise ValueError("VJepaEncoder.hub_entry is empty; set it from the "
                             "vjepa2 release you deploy (see config/target_memory.yaml)")
        import torch  # noqa: PLC0415 — optional heavy dependency
        loaded = torch.hub.load(self.hub_repo, self.hub_entry)
        model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        model = model.to(self.device).eval()
        if self.fp16 and self.device.startswith("cuda"):
            model = model.half()
        self._model = model
        return model

    def encode(self, clip: Sequence[np.ndarray]) -> np.ndarray:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        model = self._load()
        frames = list(clip)
        while len(frames) < self.tubelet:          # pad short clips at start
            frames.insert(0, frames[0])
        x = torch.from_numpy(np.stack(frames)).float() / 255.0      # T,H,W,3
        x = x.permute(0, 3, 1, 2)                                     # T,3,H,W
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((x - mean) / std).permute(1, 0, 2, 3).unsqueeze(0)       # 1,3,T,H,W
        x = x.to(self.device)
        if self.fp16 and self.device.startswith("cuda"):
            x = x.half()
        with torch.no_grad():
            tokens = model(x)                                         # 1,N,C
        g = self.image_size // self.patch_size
        t = tokens.shape[1] // (g * g)
        feat = tokens[0].reshape(t, g, g, -1)[-1]                     # last time slot
        feat = F.normalize(feat.float(), dim=-1)
        return feat.cpu().numpy()


@dataclass
class DinoV2Encoder:
    """DINOv2 dense patch features (E1 baseline T2), lazy torch.

    ``hub_entry`` e.g. ``dinov2_vits14`` / ``dinov2_vitb14`` (facebookresearch/dinov2
    README). The frame is resized to ``image_size`` (a multiple of 14) and the
    normalised patch tokens of the last frame are returned."""

    hub_repo: str = "facebookresearch/dinov2"
    hub_entry: str = "dinov2_vits14"
    image_size: int = 448
    patch_size: int = 14
    device: str = "cuda"
    fp16: bool = True
    _model: Optional[object] = None

    def _load(self):
        if self._model is None:
            import torch  # noqa: PLC0415
            model = torch.hub.load(self.hub_repo, self.hub_entry).to(self.device).eval()
            if self.fp16 and self.device.startswith("cuda"):
                model = model.half()
            self._model = model
        return self._model

    def encode(self, clip: Sequence[np.ndarray]) -> np.ndarray:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        model = self._load()
        x = torch.from_numpy(np.asarray(clip[-1])).float().permute(2, 0, 1)[None] / 255.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.device)
        if self.fp16 and self.device.startswith("cuda"):
            x = x.half()
        with torch.no_grad():
            tok = model.forward_features(x)["x_norm_patchtokens"][0]          # N, C
        g = self.image_size // self.patch_size
        feat = F.normalize(tok.float().reshape(g, g, -1), dim=-1)
        return feat.cpu().numpy()


def make_encoder(kind: str, **kw) -> DenseEncoder:
    if kind == "color_patch":
        return ColorPatchEncoder(patch_size=int(kw.get("patch_size", 16)))
    if kind == "vjepa":
        return VJepaEncoder(**kw)
    if kind == "dinov2":
        return DinoV2Encoder(**kw)
    raise ValueError(f"unknown encoder {kind!r}")
