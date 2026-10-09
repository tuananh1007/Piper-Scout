#!/usr/bin/env python3
"""Print the TCP (grasp point) position and a goal command for the whole-body MPC.

    ros2 run scout_piper_bringup print_tcp.py                  # TCP and gripper axis in odom
    ros2 run scout_piper_bringup print_tcp.py --forward 0.05   # plus a goal 5 cm ahead of it
    ros2 run scout_piper_bringup print_tcp.py --ahead 0.05     # plus a goal 5 cm along the gripper axis

The TCP is ``--tcp-offset`` (0.14 m, the MPC's tcp_offset_m) along the z axis
of ``piper_link6``. ``--forward`` moves the goal horizontally along the base's
heading (``base_link`` x) at the TCP's height; ``--ahead`` moves it along the
gripper axis, which may point down. Either prints the ``ros2 topic pub``
command that sends the point goal to /whole_body_mpc/goal. Reads TF only;
publishes nothing.
"""

import argparse
import sys


def tcp_and_axis(translation, quat_xyzw, tcp_offset: float):
    """TCP position and unit approach axis (link6 z) from a link6 pose."""
    import numpy as np  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415
    z = Rotation.from_quat(quat_xyzw).as_matrix()[:, 2]
    return np.asarray(translation, dtype=float) + tcp_offset * z, z


def heading(quat_xyzw):
    """Unit horizontal direction of a frame's x axis (the base's heading)."""
    import numpy as np  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415
    x = Rotation.from_quat(quat_xyzw).as_matrix()[:, 0]
    x[2] = 0.0
    return x / max(float(np.linalg.norm(x)), 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="odom", help="frame to report in (default odom)")
    ap.add_argument("--link", default="piper_link6", help="flange frame")
    ap.add_argument("--tcp-offset", type=float, default=0.14, help="TCP along the flange z [m]")
    ap.add_argument("--base", default="base_link", help="base frame for --forward")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--forward", type=float, default=None,
                   help="also print a goal this far ahead along the base heading [m]")
    g.add_argument("--ahead", type=float, default=None,
                   help="also print a goal this far along the gripper axis [m]")
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for TF")
    a = ap.parse_args()

    import rclpy  # noqa: PLC0415
    import tf2_ros  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from rclpy.time import Time  # noqa: PLC0415

    rclpy.init()
    node = Node("print_tcp")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)
    t_end = node.get_clock().now().nanoseconds * 1e-9 + a.timeout
    frames = [a.link] + ([a.base] if a.forward is not None else [])
    try:
        for f in frames:
            while not buf.can_transform(a.frame, f, Time()):
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.get_clock().now().nanoseconds * 1e-9 > t_end:
                    print(f"no TF from {a.frame} to {f} within {a.timeout:.0f} s "
                          "(is the bringup running, with the base for odom?)", file=sys.stderr)
                    return 1
        tf = buf.lookup_transform(a.frame, a.link, Time()).transform
        tf_base = buf.lookup_transform(a.frame, a.base, Time()).transform if a.forward is not None else None
    finally:
        node.destroy_node()
        rclpy.shutdown()

    tr, q = tf.translation, tf.rotation
    tcp, z = tcp_and_axis([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w], a.tcp_offset)
    fmt = lambda v: " ".join(f"{x:.3f}" for x in v)  # noqa: E731
    print(f"TCP in {a.frame}: {fmt(tcp)}")
    print(f"gripper axis:  {fmt(z)}")
    if a.ahead is not None or tf_base is not None:
        if tf_base is not None:
            qb = tf_base.rotation
            g = tcp + a.forward * heading([qb.x, qb.y, qb.z, qb.w])
            print(f"goal {100 * a.forward:.0f} cm ahead along {a.base} x: {fmt(g)}")
        else:
            g = tcp + a.ahead * z
            print(f"goal {100 * a.ahead:.0f} cm along the gripper axis: {fmt(g)}")
        print("ros2 topic pub --once /whole_body_mpc/goal geometry_msgs/msg/PointStamped "
              f"\"{{header: {{frame_id: {a.frame}}}, point: {{x: {g[0]:.3f}, y: {g[1]:.3f}, "
              f"z: {g[2]:.3f}}}}}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
