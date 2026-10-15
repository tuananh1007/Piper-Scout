#!/usr/bin/env python3
"""Hardware-free check of the servo command chain (INSTALL.md, Step 10.4).

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_servo:=true bringup_pipeline:=false
    ros2 run scout_piper_bringup servo_chain_check.py          # second terminal

Runs only against fake_piper_driver.py (it refuses when the real driver
answers). Checks, in order: the relay publishes piper_joint1..8; with the
bridge disabled a joint jog moves nothing; once enabled, a jog of
piper_joint1 moves joint1 only; every command carries all six joints, the
measured gripper opening and the bridge's speed percent; a +z twist raises the
flange; after disabling, the arm holds; servo never halts for a singularity or
collision. Leaves the bridge disabled. Exit code 0 when all pass.
"""

import sys
import time


def main() -> int:
    import rclpy  # noqa: PLC0415
    import tf2_ros  # noqa: PLC0415
    from control_msgs.msg import JointJog  # noqa: PLC0415
    from geometry_msgs.msg import TwistStamped  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from sensor_msgs.msg import JointState  # noqa: PLC0415
    from std_msgs.msg import Int8  # noqa: PLC0415
    from std_srvs.srv import SetBool, Trigger  # noqa: PLC0415

    rclpy.init()
    n = Node("servo_chain_check")
    fb, js, cmds, status = {}, {}, [], []
    n.create_subscription(JointState, "/joint_states_single", lambda m: fb.update(zip(m.name, m.position)), 10)
    n.create_subscription(JointState, "/joint_states", lambda m: js.update(zip(m.name, m.position)), 10)
    n.create_subscription(JointState, "/piper/joint_cmd", cmds.append, 100)
    n.create_subscription(Int8, "/servo_node/status", lambda m: status.append(m.data), 100)
    jog_pub = n.create_publisher(JointJog, "/servo_node/delta_joint_cmds", 10)
    twist_pub = n.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, n)

    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.01)

    def call(srv_type, name, request, timeout=15.0):
        client = n.create_client(srv_type, name)
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {name} not available")
        future = client.call_async(request)
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        return future.result()

    def stream(make, sec):
        end = time.time() + sec
        while time.time() < end:
            make()
            spin(0.02)

    def jog(sec, vel=0.3):
        def make():
            m = JointJog()
            m.header.stamp = n.get_clock().now().to_msg()
            m.joint_names, m.velocities = ["piper_joint1"], [vel]
            jog_pub.publish(m)
        stream(make, sec)

    def twist(sec, vz=0.05):
        def make():
            m = TwistStamped()
            m.header.stamp = n.get_clock().now().to_msg()
            m.header.frame_id = "piper_base_link"
            m.twist.linear.z = vz
            twist_pub.publish(m)
        stream(make, sec)

    def flange_z():
        return tf_buffer.lookup_transform("piper_base_link", "piper_link6", rclpy.time.Time()).transform.translation.z

    def enable(on):
        return call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=on))

    # refuse to run against the real driver: only the fake declares initial_gripper
    reply = call(GetParameters, "/piper_ctrl_single_node/get_parameters",
                 GetParameters.Request(names=["initial_gripper"]))
    if not reply.values or reply.values[0].type == 0:
        print("REFUSED: /piper_ctrl_single_node is not fake_piper_driver.py; "
              "launch with fake_arm:=true", flush=True)
        return 2

    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    end = time.time() + 10.0                       # DDS discovery can take a few seconds
    while time.time() < end and not (fb and all(f"piper_joint{i}" in js for i in range(1, 9))):
        spin(0.1)
    check("relay publishes piper_joint1..8", all(f"piper_joint{i}" in js for i in range(1, 9)), str(sorted(js)))
    call(Trigger, "/servo_node/start_servo", Trigger.Request())
    spin(2.0)
    enable(False)
    q0, n0 = dict(fb), len(cmds)
    jog(2.0)
    spin(0.5)
    check("bridge disabled: nothing moves", abs(fb["joint1"] - q0["joint1"]) < 1e-6 and len(cmds) == n0,
          f"joint1 {q0['joint1']:+.4f} -> {fb['joint1']:+.4f}")
    enable(True)
    q1, n1 = dict(fb), len(cmds)
    jog(2.0)
    spin(0.5)
    d1 = fb["joint1"] - q1["joint1"]
    others = max(abs(fb[j] - q1[j]) for j in ("joint2", "joint3", "joint4", "joint5", "joint6"))
    check("jog moves joint1 only", 0.2 < d1 < 0.7 and others < 0.01,
          f"joint1 {d1:+.3f} rad for 0.6 commanded, others {others:.4f}")
    sent = cmds[n1:]
    check("commands: 6 joints + held gripper + speed percent",
          sent and all(len(c.position) == 7 and abs(c.position[6] - q1["gripper"]) < 1e-3
                       and len(c.velocity) == 7 and 1 <= c.velocity[6] <= 100 for c in sent),
          f"{len(sent)} commands")
    check("gripper unchanged", abs(fb["gripper"] - q1["gripper"]) < 1e-4, f"{fb['gripper']:.4f} m")
    z0 = flange_z()
    twist(2.0)
    spin(0.5)
    z1 = flange_z()
    check("+z twist raises the flange", 0.03 < z1 - z0 < 0.12, f"{z0:.3f} -> {z1:.3f} m for 0.10 commanded")
    enable(False)
    spin(0.5)
    q2 = dict(fb)
    jog(1.5)
    spin(0.5)
    check("disabled again: arm holds", abs(fb["joint1"] - q2["joint1"]) < 1e-3,
          f"joint1 {q2['joint1']:+.4f} -> {fb['joint1']:+.4f}")
    check("no singularity / collision halt", not [s for s in status if s in (2, 4)],
          f"servo status codes {sorted(set(status))}")
    ok = all(results)
    print("SERVO CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
