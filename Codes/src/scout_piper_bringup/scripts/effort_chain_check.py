#!/usr/bin/env python3
"""Hardware-free check of the joint-effort force estimate (no F/T sensor on the platform).

    ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
        bringup_servo:=true bringup_force_estimate:=true bringup_pipeline:=false
    ros2 run scout_piper_bringup effort_chain_check.py [--calibrate effort_calibration.json]

The fake driver simulates joint efforts (gravity, friction, noise) and the
torque of a force at the TCP (its ``external_force_n`` parameter), the relay
passes them to /joint_states and effort_force_node estimates the force on
/ft_sensor/raw. With the arm at rest the check tares the estimate, applies
known forces and compares the estimate (at rest friction is zero, so the
uncalibrated model suffices). With ``--calibrate OUT`` it then runs
``calibrate_effort --execute`` (servo started and the bridge enabled here) from
the centre pose and checks the fitted model: joints 2-5 identified, their gain
1 (the fake's), a small held-out residual. OUT is the calibration file for
``effort_calibration:=``. Runs only against the fake driver; leaves the bridge
disabled and the fake force at zero.
"""

import argparse
import json
import subprocess
import sys
import time

CENTER = "0.0,0.8,-1.2,0.0,0.45,0.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.3, help="N, estimate vs applied force")
    ap.add_argument("--calibrate", default="", help="run calibrate_effort and write this file")
    ap.add_argument("--duration", type=float, default=60.0, help="calibration motion [s]")
    a = ap.parse_args()

    import numpy as np  # noqa: PLC0415
    import rclpy  # noqa: PLC0415
    from geometry_msgs.msg import WrenchStamped  # noqa: PLC0415
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue  # noqa: PLC0415
    from rcl_interfaces.srv import GetParameters, SetParameters  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from std_msgs.msg import String  # noqa: PLC0415
    from std_srvs.srv import SetBool, Trigger  # noqa: PLC0415

    rclpy.init()
    n = Node("effort_chain_check")
    forces, status = [], []
    n.create_subscription(WrenchStamped, "/ft_sensor/raw", lambda m: forces.append(
        (time.time(), np.array([m.wrench.force.x, m.wrench.force.y, m.wrench.force.z]))), 100)
    n.create_subscription(String, "/effort_force_estimator/status", lambda m: status.append(json.loads(m.data)), 10)

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

    def param(node, name):
        reply = call(GetParameters, f"{node}/get_parameters", GetParameters.Request(names=[name]))
        return reply.values[0] if reply.values and reply.values[0].type != 0 else None

    def set_force(f):
        v = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=[float(x) for x in f])
        call(SetParameters, "/piper_ctrl_single_node/set_parameters",
             SetParameters.Request(parameters=[Parameter(name="external_force_n", value=v)]))

    if param("/piper_ctrl_single_node", "initial_gripper") is None:
        print("REFUSED: /piper_ctrl_single_node is not the fake driver; launch with fake_arm:=true", flush=True)
        return 2
    if param("/effort_force_estimator", "wrench_topic") is None:
        print("REFUSED: /effort_force_estimator is not running; launch with bringup_force_estimate:=true",
              flush=True)
        return 2
    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + f"{name}: {detail}", flush=True)

    def mean_force(sec):
        t0 = time.time()
        spin(sec)
        got = [f for t, f in forces if t > t0]
        return (np.mean(got, axis=0), np.std(got, axis=0)) if got else (np.full(3, np.nan), np.full(3, np.nan))

    spin(2.0)
    check("force estimate published from the relayed efforts", len(forces) > 50 and status and status[-1]["efforts"],
          f"{len(forces)} WrenchStamped in 2 s; status {status[-1] if status else None}")
    set_force([0.0, 0.0, 0.0])
    spin(0.5)
    reply = call(Trigger, "/effort_force_estimator/tare", Trigger.Request())
    zero, sd = mean_force(1.0)
    check("tared estimate at rest is zero", reply is not None and reply.success and np.linalg.norm(zero) < a.tolerance,
          f"{np.round(zero, 3)} N (1-σ {np.round(sd, 3)})")
    for F in ([0.0, 0.0, -3.0], [2.0, -1.0, 0.5]):
        set_force(F)
        spin(1.0)                                    # the estimator's low-pass settles
        got, sd = mean_force(1.0)
        err = float(np.linalg.norm(got - np.array(F)))
        check(f"force {F} N at the TCP is estimated", err < a.tolerance,
              f"{np.round(got, 3)} N, error {err:.3f} N (1-σ {np.round(sd, 3)})")
    set_force([0.0, 0.0, 0.0])
    spin(1.0)
    got, _ = mean_force(0.5)
    check("estimate returns to zero", np.linalg.norm(got) < a.tolerance, f"{np.round(got, 3)} N")
    if status:
        st = status[-1]
        print(f"info estimator: calibrated {st.get('calibrated')}, noise {st.get('noise_n')} N, "
              f"3-σ threshold {st.get('threshold_3sigma_n')} N", flush=True)

    if a.calibrate:
        call(Trigger, "/servo_node/start_servo", Trigger.Request())
        call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=True))
        try:
            proc = subprocess.run(["ros2", "run", "scout_piper_whole_body_mpc", "calibrate_effort", "--execute",
                                   "--center", CENTER, "--duration", str(a.duration), "--out", a.calibrate],
                                  capture_output=True, text=True, timeout=a.duration + 120.0)
        finally:
            call(SetBool, "/piper_servo_bridge/enable", SetBool.Request(data=False))
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        if proc.returncode != 0:
            check("calibrate_effort fits the fake arm", False, f"exit {proc.returncode}: {' | '.join(tail)}")
        else:
            with open(a.calibrate, encoding="utf-8") as f:
                cal = json.load(f)
            rep = cal["report"]
            ok = (rep["identified"] == [False, True, True, True, True, False]
                  and np.allclose(cal["a"], 1.0, atol=0.05) and max(rep["holdout_sigma_nm"]) < 0.05)
            check("calibrate_effort fits the fake arm", ok,
                  f"{rep['samples']} samples, gains {rep['a']}, held-out σ {rep['holdout_sigma_nm']} N·m, "
                  f"3-σ force threshold {rep['threshold_3sigma_n']} N")
    ok = all(results)
    print("EFFORT CHAIN", "OK" if ok else "FAILED", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
