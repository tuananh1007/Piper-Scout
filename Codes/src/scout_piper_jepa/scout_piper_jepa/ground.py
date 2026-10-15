"""Publish a target mask to (re)ground the Stage A memory (target_state_node, predictive_mpc_node).

    python3 -m scout_piper_jepa.ground --png mask.png            # non-zero pixels = target
    python3 -m scout_piper_jepa.ground --box 300 200 340 260     # u0 v0 u1 v1 in pixels
    python3 -m scout_piper_jepa.ground --circle 320 240 20       # u v radius

The mask size comes from the PNG or, for --box / --circle, from the colour
camera_info. It is published three times (in case one is dropped) on
/piper_jepa/init_mask as mono8.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def box_mask(hw, u0, v0, u1, v1) -> np.ndarray:
    m = np.zeros(hw, bool)
    m[max(int(v0), 0):int(v1), max(int(u0), 0):int(u1)] = True
    return m


def circle_mask(hw, u, v, r) -> np.ndarray:
    vv, uu = np.mgrid[0:hw[0], 0:hw[1]]
    return (uu + 0.5 - u) ** 2 + (vv + 0.5 - v) ** 2 <= r * r


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--png")
    g.add_argument("--box", type=float, nargs=4, metavar=("U0", "V0", "U1", "V1"))
    g.add_argument("--circle", type=float, nargs=3, metavar=("U", "V", "R"))
    ap.add_argument("--topic", default="/piper_jepa/init_mask")
    ap.add_argument("--camera-info", default="/camera/color/camera_info")
    a = ap.parse_args(argv)

    import rclpy  # noqa: PLC0415
    from sensor_msgs.msg import CameraInfo, Image  # noqa: PLC0415

    rclpy.init()
    node = rclpy.create_node("jepa_ground")
    if a.png:
        from .annotate import _read_png  # noqa: PLC0415
        m = _read_png(a.png)
        mask = (m.max(-1) if m.ndim == 3 else m) > 0
    else:
        info = []
        node.create_subscription(CameraInfo, a.camera_info, info.append, 1)
        t_end = time.monotonic() + 5.0
        while not info and time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not info:
            raise SystemExit(f"no camera_info on {a.camera_info}")
        hw = (info[0].height, info[0].width)
        mask = box_mask(hw, *a.box) if a.box else circle_mask(hw, *a.circle)
    pub = node.create_publisher(Image, a.topic, 2)
    msg = Image()
    msg.height, msg.width = mask.shape
    msg.encoding, msg.step = "mono8", mask.shape[1]
    msg.data = (mask.astype(np.uint8) * 255).tobytes()
    for _ in range(3):
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.3)
    print(f"published a {mask.shape[1]}x{mask.shape[0]} mask with {int(mask.sum())} target pixels on {a.topic}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
