"""Workspace-relative locations for calibration and recorded data.

Both live in the colcon workspace (Codes/ on the host, /workspace in the dev
container), outside install/, so they survive rebuilds:

    <ws>/calibration/hand_eye.local.yaml   per-machine hand-eye result (gitignored)
    <ws>/data/episodes/<episode>/          recordings (gitignored)

Override the workspace with PIPER_WS if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def workspace_root() -> Path:
    env = os.environ.get("PIPER_WS")
    if env:
        return Path(env)
    # <ws>/install/piper_quickstart/share/piper_quickstart
    return Path(os.path.abspath(get_package_share_directory("piper_quickstart"))).parents[3]


def default_hand_eye_file() -> Path:
    return workspace_root() / "calibration" / "hand_eye.local.yaml"


def nominal_hand_eye_file() -> Path:
    return Path(get_package_share_directory("piper_quickstart")) / "config" / "hand_eye.nominal.yaml"


def episodes_dir() -> Path:
    return workspace_root() / "data" / "episodes"
