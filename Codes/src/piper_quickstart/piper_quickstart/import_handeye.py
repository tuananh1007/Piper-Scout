"""Convert a MoveIt Calibration export into the quickstart's calibration file.

After solving in the HandEyeCalibration RViz panel, use its "Save camera pose"
button to write a .launch.py (or .xml/.yaml). Then:

    ros2 run piper_quickstart import_handeye ~/handeye.launch.py

writes /workspace/calibration/hand_eye.local.yaml (gitignored, per machine),
which quickstart.launch.py loads automatically on the next launch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import yaml

from piper_quickstart.paths import default_hand_eye_file
from piper_quickstart.transforms import parse_moveit_calibration_export


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_file", type=Path, help="file saved by MoveIt Calibration's 'Save camera pose'")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output calibration YAML (default: <workspace>/calibration/hand_eye.local.yaml)")
    ap.add_argument("--notes", default="", help="free text stored with the calibration (target, #samples, residual)")
    args, _ = ap.parse_known_args(argv)

    try:
        parsed = parse_moveit_calibration_export(args.export_file.read_text())
    except (OSError, ValueError) as e:
        print(f"error: {args.export_file}: {e}", file=sys.stderr)
        return 1
    if parsed["mount_type"] and parsed["mount_type"].lower() != "eye-in-hand":
        print(f"error: export is {parsed['mount_type']}, but the camera is wrist-mounted (Eye-in-hand)",
              file=sys.stderr)
        return 1

    out = args.output or default_hand_eye_file()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "parent_frame": parsed["parent_frame"],
        "child_frame": parsed["child_frame"],
        "translation": parsed["translation"],
        "rotation_xyzw": parsed["rotation_xyzw"],
        "calibrated": True,
        "source": f"MoveIt Calibration export {args.export_file.name}",
        "imported_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": args.notes,
    }
    with open(out, "w") as f:
        f.write("# Hand-eye calibration imported from MoveIt Calibration (piper_quickstart import_handeye).\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"wrote {out}\n  {doc['parent_frame']} -> {doc['child_frame']}\n"
          f"  t = {doc['translation']}\n  q = {doc['rotation_xyzw']}\n"
          "Relaunch quickstart.launch.py to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
