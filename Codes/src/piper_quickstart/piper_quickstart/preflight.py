"""Hardware preflight for the Piper quickstart: run before any driver starts.

Checks, in order, and exits non-zero with a fix-it message on the first
hard failure in each group:

  arm     CAN interface exists, is UP at the expected bitrate, is not
          BUS-OFF, and carries Piper joint-feedback frames (0x2A5-0x2A7).
  camera  An Intel RealSense is enumerated on USB (and on a USB 3 link).
  display X11 is reachable for RViz.

Usage (also run automatically by quickstart.launch.py):
    ros2 run piper_quickstart preflight --can-port can0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

PIPER_JOINT_FEEDBACK_IDS = {0x2A5, 0x2A6, 0x2A7}
PIPER_ANY_FEEDBACK_IDS = PIPER_JOINT_FEEDBACK_IDS | {0x2A1, 0x2A2, 0x2A3, 0x2A4, 0x2A8} | set(range(0x251, 0x257))
REALSENSE_VENDOR = "8086"

CAN_ACTIVATE_HINT = (
    "On the HOST (not in the container), with the arm's USB-CAN adapter plugged in:\n"
    "      bash Codes/src/piper_ros/can_activate.sh {can} 1000000"
)

_GREEN, _RED, _YELLOW, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


class Report:
    def __init__(self) -> None:
        self.failed = False

    def ok(self, msg: str) -> None:
        print(f"  {_GREEN}[ OK ]{_RESET} {msg}", flush=True)

    def warn(self, msg: str) -> None:
        print(f"  {_YELLOW}[WARN]{_RESET} {msg}", flush=True)

    def fail(self, msg: str, fix: str = "") -> None:
        self.failed = True
        print(f"  {_RED}[FAIL]{_RESET} {msg}", flush=True)
        if fix:
            print(f"         fix: {fix}", flush=True)


# --------------------------------------------------------------------- CAN
def _ip_details(can: str) -> Optional[dict]:
    if not shutil.which("ip"):
        return None
    try:
        out = subprocess.run(["ip", "-details", "-json", "link", "show", can],
                             capture_output=True, text=True, timeout=3, check=True).stdout
        return json.loads(out)[0]
    except Exception:
        return None


def _listen_can(can: str, seconds: float) -> Tuple[int, set]:
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        s.bind((can,))
        s.settimeout(0.1)
        n, ids = 0, set()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                frame = s.recv(16)
            except socket.timeout:
                continue
            can_id = struct.unpack("=I", frame[:4])[0] & socket.CAN_EFF_MASK
            n += 1
            ids.add(can_id)
        return n, ids
    finally:
        s.close()


def check_arm(r: Report, can: str, bitrate: int, listen_s: float) -> None:
    print(f"Piper arm on {can}:", flush=True)
    sysdir = Path("/sys/class/net") / can
    if not sysdir.exists():
        present = sorted(p.name for p in Path("/sys/class/net").iterdir()
                         if (p / "type").exists() and (p / "type").read_text().strip() == "280")
        r.fail(f"CAN interface '{can}' does not exist (CAN interfaces present: {present or 'none'})",
               CAN_ACTIVATE_HINT.format(can=can) + "\n      If the adapter has a different name, pass can_port:=<name>.")
        return
    flags = int((sysdir / "flags").read_text().strip(), 16)
    if not flags & 0x1:
        r.fail(f"'{can}' exists but is DOWN", CAN_ACTIVATE_HINT.format(can=can))
        return
    r.ok(f"'{can}' is UP")

    info = _ip_details(can)
    if info is None:
        r.fail("cannot read CAN bitrate: `ip` (iproute2) missing in this container",
               "rebuild the dev image: ./docker/build_dev.sh  (piper_sdk also needs `ip`)")
        return
    data = info.get("linkinfo", {}).get("info_data", {})
    actual = data.get("bittiming", {}).get("bitrate")
    state = data.get("state", "?")
    if actual != bitrate:
        r.fail(f"'{can}' bitrate is {actual}, Piper needs {bitrate}", CAN_ACTIVATE_HINT.format(can=can))
        return
    r.ok(f"bitrate {actual}")
    if state == "BUS-OFF":
        r.fail(f"'{can}' is BUS-OFF (too many bus errors)",
               "check CAN-H/CAN-L wiring and termination, then re-run can_activate.sh on the host")
        return
    if state != "ERROR-ACTIVE":
        r.warn(f"controller state is {state} (expected ERROR-ACTIVE): the bus is seeing errors")

    try:
        n, ids = _listen_can(can, listen_s)
    except OSError as e:
        r.fail(f"cannot open a raw socket on '{can}': {e}")
        return
    if n == 0:
        r.fail(f"'{can}' is up but silent for {listen_s:.1f}s: no frames from the arm",
               "power on the Piper (switch on the base) and check the CAN cable from the adapter to the arm")
        return
    if not ids & PIPER_JOINT_FEEDBACK_IDS:
        other = ", ".join(hex(i) for i in sorted(ids)[:8])
        hint = "this looks like a different device's bus (e.g. the Scout base)." if not ids & PIPER_ANY_FEEDBACK_IDS else ""
        r.fail(f"{n} frames on '{can}' but no Piper joint feedback (0x2A5-0x2A7); saw {other}", hint)
        return
    r.ok(f"Piper is talking: {n / listen_s:.0f} frames/s including joint feedback")


# ------------------------------------------------------------------ camera
def _usb_realsense() -> List[dict]:
    found = []
    for dev in Path("/sys/bus/usb/devices").glob("*"):
        try:
            if (dev / "idVendor").read_text().strip() != REALSENSE_VENDOR:
                continue
            product = (dev / "product").read_text().strip() if (dev / "product").exists() else ""
            if "realsense" not in product.lower():
                continue
            found.append({
                "product": product,
                "serial": (dev / "serial").read_text().strip() if (dev / "serial").exists() else "?",
                "speed_mbps": int(float((dev / "speed").read_text().strip())),
                "bus_path": dev.name,
            })
        except OSError:
            continue
    return found


def check_camera(r: Report, serial: str, require_usb3: bool) -> None:
    print("RealSense camera:", flush=True)
    if not Path("/dev/bus/usb").exists():
        r.fail("/dev/bus/usb is not visible in this container",
               "start the container with docker/compose.dev.yml (it bind-mounts /dev/bus/usb)")
        return
    cams = _usb_realsense()
    if not cams:
        r.fail("no Intel RealSense found on USB",
               "plug the camera into a USB 3 port (blue) with a USB 3 cable; check `lsusb | grep 8086` on the host.\n"
               "      If you plugged it in after starting the container, restart the container.")
        return
    if serial:
        matches = [c for c in cams if c["serial"] == serial]
        if not matches:
            r.fail(f"RealSense with serial {serial} not found; present: {[c['serial'] for c in cams]}")
            return
        cams = matches
    elif len(cams) > 1:
        r.warn(f"{len(cams)} RealSense cameras found; the driver will pick one. Pass camera_serial:=<serial>.")
    cam = cams[0]
    r.ok(f"{cam['product']} (serial {cam['serial']}) on USB {cam['bus_path']}")
    if cam["speed_mbps"] < 5000:
        msg = (f"camera is on a USB 2 link ({cam['speed_mbps']} Mb/s): depth+color at 640x480@30 "
               "will drop frames or fail")
        fix = "use a USB 3 port and cable (a USB-C to A cable often negotiates USB 2 when flipped)"
        (r.fail if require_usb3 else r.warn)(msg + ("" if require_usb3 else f" — {fix}"), fix if require_usb3 else "")
    else:
        r.ok(f"USB 3 link ({cam['speed_mbps']} Mb/s)")


# ----------------------------------------------------------------- display
def check_display(r: Report) -> None:
    print("Display (RViz):", flush=True)
    disp = os.environ.get("DISPLAY", "")
    if not disp:
        r.fail("DISPLAY is not set", "start the container via docker/compose.dev.yml, or pass rviz:=false")
        return
    sock = Path("/tmp/.X11-unix") / f"X{disp.split(':')[-1].split('.')[0]}"
    if not sock.exists():
        r.fail(f"X11 socket {sock} missing", "the compose file mounts /tmp/.X11-unix; run `xhost +local:docker` on the host")
        return
    r.ok(f"DISPLAY={disp}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can-port", default="can0")
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--listen-s", type=float, default=1.0)
    ap.add_argument("--camera-serial", default="")
    ap.add_argument("--require-usb3", action="store_true")
    ap.add_argument("--skip-arm", action="store_true")
    ap.add_argument("--skip-camera", action="store_true")
    ap.add_argument("--skip-display", action="store_true")
    args, _ros_args = ap.parse_known_args(argv)

    print("\n=== Piper quickstart preflight ===", flush=True)
    r = Report()
    if not args.skip_arm:
        check_arm(r, args.can_port, args.bitrate, args.listen_s)
    if not args.skip_camera:
        check_camera(r, args.camera_serial.strip("'\""), args.require_usb3)
    if not args.skip_display:
        check_display(r)
    if r.failed:
        print(f"\n{_RED}Preflight FAILED: fix the items above and relaunch.{_RESET}\n", flush=True)
        return 1
    print(f"\n{_GREEN}Preflight passed.{_RESET}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
