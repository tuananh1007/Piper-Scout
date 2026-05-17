#!/usr/bin/env python3
"""Regenerate scout_piper_description/urdf/_piper_arm.xacro from upstream.

The upstream piper_description.xacro hardcodes its root link as `base_link`
and declares a `world` link with a fixed joint to it. That clashes with
scout_v2.xacro (which also defines `base_link`) when both are xacro:included
into the same robot.

This script reads upstream piper_description.xacro and emits a forked version
in our package with:

  * the `world` link + `fixed_base_joint` block stripped
  * every link/joint name prefixed with `${prefix}`
  * the whole content wrapped in a <xacro:macro name="piper_arm"
    params="prefix parent *origin"> so the caller chooses both the name
    prefix and the parent link (e.g. piper_mount_link on the Scout)
  * a mount joint inside the macro that welds `${parent}` -> `${prefix}base_link`

Run from any directory:
    ./scripts/fork_piper_arm.py
"""

from __future__ import annotations

import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
WS = HERE.parent
SRC_XACRO = (
    WS / "src" / "piper_ros" / "src" / "piper_description"
       / "urdf" / "piper_description.xacro"
)
DST_XACRO = (
    WS / "src" / "scout_piper_description" / "urdf" / "_piper_arm.xacro"
)

# Link/joint names that need ${prefix} added. Order matters only insofar as
# the regex anchors are exact-name match (handled below).
LINK_NAMES = ["base_link", "gripper_base"] + [f"link{i}" for i in range(1, 9)]
JOINT_NAMES = (
    [f"joint{i}" for i in range(1, 9)]
    + ["joint6_to_gripper_base"]
)


def main() -> None:
    if not SRC_XACRO.exists():
        raise SystemExit(
            f"Upstream piper xacro not found: {SRC_XACRO}\n"
            "Run `vcs import src < repos.yaml` first."
        )

    text = SRC_XACRO.read_text()

    # 1) Strip XML decl + <robot ...> opening and </robot> closing — we'll
    #    add our own.
    text = re.sub(r"^\s*<\?xml[^?]*\?>\s*", "", text, count=1)
    text = re.sub(
        r"<robot[^>]*>", "", text, count=1, flags=re.DOTALL,
    )
    text = re.sub(r"</robot>\s*$", "", text, count=1)

    # 2) Drop the world link + fixed_base_joint block. The block is exactly:
    #       <link name="world"/>
    #       <joint name="fixed_base_joint" type="fixed">
    #         <parent link="world"/>
    #         <child link="base_link"/>
    #       </joint>
    text = re.sub(
        r'<link\s+name="world"\s*/>\s*'
        r'<joint\s+name="fixed_base_joint"[^>]*>.*?</joint>\s*',
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # 3) Prefix-rename link/joint references. Each replacement uses a
    #    word-boundary-style anchor (quote + exact name + quote) so we don't
    #    accidentally match substrings like "link10" or "${joint_name}".
    def add_prefix_to(name: str) -> str:
        # name="<name>", link="<name>", reference="<name>" (Gazebo SDF blocks)
        nonlocal text
        for attr in ("name", "link", "reference"):
            text = re.sub(
                rf'(\b{attr}=")({re.escape(name)})(")',
                r'\1${prefix}\2\3',
                text,
            )
        return text

    for n in LINK_NAMES:
        add_prefix_to(n)
    for n in JOINT_NAMES:
        add_prefix_to(n)

    # 4) Prefix the joint_name argument in transmission_block calls.
    text = re.sub(
        r'(<xacro:transmission_block\s+tran_name=")(tran\d+)(")(\s+joint_name=")(joint\d+)(")',
        r'\1${prefix}\2\3\4${prefix}\5\6',
        text,
    )

    # 5) Wrap in our macro. The transmission_block macro DEFINITION (which
    #    lives in the upstream xacro) goes inside our macro too — that's
    #    legal in xacro and keeps everything self-contained.
    out = f"""<?xml version="1.0" encoding="utf-8"?>
<!--
  Forked from upstream piper_description.xacro by scripts/fork_piper_arm.py.
  DO NOT EDIT BY HAND — re-run the script to regenerate after upstream updates.
-->
<robot xmlns:xacro="http://ros.org/wiki/xacro">

  <xacro:macro name="piper_arm" params="prefix parent *origin">
    <!-- Mount the arm onto whatever the caller chose as parent. -->
    <joint name="${{prefix}}base_mount" type="fixed">
      <xacro:insert_block name="origin"/>
      <parent link="${{parent}}"/>
      <child link="${{prefix}}base_link"/>
    </joint>

    <!-- ===== Arm + gripper (forked from piper_description.xacro) ===== -->
{text.strip()}

  </xacro:macro>

</robot>
"""

    DST_XACRO.parent.mkdir(parents=True, exist_ok=True)
    DST_XACRO.write_text(out)
    print(f"Wrote {DST_XACRO}")


if __name__ == "__main__":
    main()
