"""Rigid-transform helpers and the hand-eye calibration file format.

The camera mount is stored as one transform ``parent_frame -> child_frame``,
in the same terms MoveIt Calibration exports (``--frame-id``,
``--child-frame-id``, ``--x/--y/--z``, ``--qx/--qy/--qz/--qw``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w), w >= 0."""
    m = r
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def make_transform(translation, quat_xyzw) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(*quat_xyzw)
    t[:3, 3] = np.asarray(translation, dtype=float)
    return t


def invert(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def split(t: np.ndarray):
    """4x4 -> (translation[3], quat_xyzw[4])."""
    return t[:3, 3].copy(), matrix_to_quat(t[:3, :3])


@dataclass
class CameraMount:
    """``parent_frame -> child_frame`` mount transform plus provenance."""

    parent_frame: str
    child_frame: str
    translation: List[float]
    rotation_xyzw: List[float]
    calibrated: bool = False
    source: str = ""
    notes: str = ""
    extra: Dict = field(default_factory=dict)

    def matrix(self) -> np.ndarray:
        return make_transform(self.translation, self.rotation_xyzw)

    def to_dict(self) -> Dict:
        d = {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "translation": [float(v) for v in self.translation],
            "rotation_xyzw": [float(v) for v in self.rotation_xyzw],
            "calibrated": bool(self.calibrated),
            "source": self.source,
            "notes": self.notes,
        }
        d.update(self.extra)
        return d


_KNOWN_KEYS = {"parent_frame", "child_frame", "translation", "rotation_xyzw", "calibrated", "source", "notes"}


def load_camera_mount(path: Path) -> CameraMount:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    missing = [k for k in ("parent_frame", "child_frame", "translation", "rotation_xyzw") if k not in d]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if len(d["translation"]) != 3 or len(d["rotation_xyzw"]) != 4:
        raise ValueError(f"{path}: translation needs 3 values and rotation_xyzw needs 4")
    return CameraMount(
        parent_frame=str(d["parent_frame"]),
        child_frame=str(d["child_frame"]),
        translation=[float(v) for v in d["translation"]],
        rotation_xyzw=[float(v) for v in d["rotation_xyzw"]],
        calibrated=bool(d.get("calibrated", False)),
        source=str(d.get("source", "")),
        notes=str(d.get("notes", "")),
        extra={k: v for k, v in d.items() if k not in _KNOWN_KEYS},
    )


_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def _arg(text: str, flag: str, pattern: str) -> Optional[str]:
    # Matches both `--x 0.1` (XML/YAML exports) and `"--x",\n "0.1",` (Python export).
    m = re.search(r"--" + flag + r"\b[\"']?\s*,?\s*[\"']?" + pattern, text)
    return m.group(1) if m else None


def parse_moveit_calibration_export(text: str) -> Dict:
    """Parse a MoveIt Calibration export (.launch.py / .launch.xml / .launch.yaml).

    Returns parent/child frames, translation and quaternion (xyzw).
    """
    frame = r"([A-Za-z0-9_/\-]+)"
    parent = _arg(text, "frame-id", frame)
    child = _arg(text, "child-frame-id", frame)
    vals = {k: _arg(text, k, _NUM) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    missing = [k for k, v in vals.items() if v is None]
    if parent is None or child is None or missing:
        raise ValueError(
            "not a MoveIt Calibration export: could not find "
            + ", ".join(
                (["--frame-id"] if parent is None else [])
                + (["--child-frame-id"] if child is None else [])
                + [f"--{k}" for k in missing]
            )
        )
    m = re.search(r"(Eye-in-hand|Eye-to-hand)", text, re.IGNORECASE)
    return {
        "parent_frame": parent,
        "child_frame": child,
        "translation": [float(vals["x"]), float(vals["y"]), float(vals["z"])],
        "rotation_xyzw": [float(vals["qx"]), float(vals["qy"]), float(vals["qz"]), float(vals["qw"])],
        "mount_type": m.group(1) if m else "",
    }
