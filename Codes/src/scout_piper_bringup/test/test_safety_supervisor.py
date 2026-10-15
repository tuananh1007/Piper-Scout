"""Safety supervisor decisions (scripts/safety_supervisor.py) and the servo bridge's
reading of its state, without ROS.

    cd Codes/src/scout_piper_bringup && python3 -m pytest test -q
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sup_mod = _load(os.path.join(PKG, "scripts", "safety_supervisor.py"), "safety_supervisor")
bridge = _load(os.path.join(PKG, "scripts", "piper_servo_bridge.py"), "piper_servo_bridge_s")
Limits, Supervisor = sup_mod.Limits, sup_mod.Supervisor


def _sup(**kw):
    return Supervisor(limits=Limits(**kw))


def test_gate_relays_clamped_and_zeroes_after_the_command_timeout():
    s = _sup(max_cmd_v_mps=0.5, max_cmd_w_rps=1.0, cmd_timeout_s=0.5)
    s.odom(0.0, 0.0, 0.0, 0.0, 0.0)
    assert s.command(0.0, 0.8, -2.0) == (0.5, -1.0)
    assert s.tick(0.3) == []                                   # still fresh
    assert s.tick(0.6) == ["zero_base"] and s.cmd_timeouts == 1   # the sender went quiet
    assert s.tick(0.9) == []                                   # one zero only
    s.command(1.0, 0.0, 0.0)                                   # a zero needs no timeout zero
    assert s.tick(2.0) == []


def test_heartbeat_required_only_while_moving():
    s = _sup(require_heartbeat=True, heartbeat_timeout_s=1.0)
    s.odom(0.0, 0.0, 0.0, 0.0, 0.0)
    assert s.tick(5.0) == [] and not s.tripped                 # idle: no heartbeat needed
    # a base command without a heartbeat is blocked and trips
    assert s.command(5.0, 0.2, 0.0) == (0.0, 0.0)
    assert "without operator heartbeat" in s.tripped
    s.heartbeat(6.0)
    s.odom(6.0, 0.0, 0.0, 0.0, 0.0)
    assert s.reset(6.1)[0]
    assert s.command(6.2, 0.2, 0.0) == (0.2, 0.0)
    s.odom(6.3, 0.02, 0.0, 0.2, 0.0)
    assert s.tick(6.5) == [] and not s.tripped                 # heartbeat 0.5 s old
    s.command(6.9, 0.2, 0.0)
    s.odom(7.1, 0.1, 0.0, 0.2, 0.0)
    assert s.tick(7.2) == ["trip"] and "heartbeat lost" in s.tripped
    assert s.command(7.3, 0.2, 0.0) == (0.0, 0.0)              # blocked while tripped
    ok, msg = s.reset(7.4)                                     # base still rolling, no heartbeat
    assert not ok and "still unsafe" in msg
    s.heartbeat(7.5)
    s.odom(7.5, 0.1, 0.0, 0.0, 0.0)
    assert s.reset(7.6)[0] and not s.tripped


def test_arm_motion_counts_as_moving_for_the_heartbeat():
    s = _sup(require_heartbeat=True)
    s.joints(0.0, 0.0, [0.0] * 6)
    s.joints(0.01, 0.01, [0.0] * 6)
    assert s.tick(0.02) == []
    s.joints(0.03, 0.03, [0.002] + [0.0] * 5)                  # 0.1 rad/s
    assert s.arm_moving() and s.tick(0.04) == ["trip"]


def test_overspeed_and_stale_feedback_trip():
    s = _sup(max_base_speed_mps=0.7)
    s.odom(0.0, 0.0, 0.0, 0.9, 0.0)
    assert s.tick(0.1) == ["trip"] and "base overspeed" in s.tripped

    s = _sup(max_joint_speed_rps=1.5)
    s.joints(0.0, 0.0, [0.0] * 6)
    s.joints(0.01, 0.01, [0.03] + [0.0] * 5)                   # 3 rad/s once: a glitch is tolerated
    assert s.tick(0.011) == []
    s.joints(0.02, 0.02, [0.06] + [0.0] * 5)                   # twice in a row
    assert s.tick(0.021) == ["trip"] and "joint overspeed" in s.tripped

    s = _sup()
    s.joints(0.0, 0.0, [0.0] * 6)
    s.joints(0.0005, 0.0005, [0.01] + [0.0] * 5)               # same-instant duplicate: no speed from it
    assert s.qd_max == 0.0

    s = _sup(max_feedback_age_s=0.5)
    s.odom(0.0, 0.0, 0.0, 0.0, 0.0)
    s.command(1.0, 0.2, 0.0)                                   # commanded, odom 1 s old
    assert s.tick(1.05) == ["trip"] and "odom stale" in s.tripped

    s = _sup(max_feedback_age_s=0.5)
    s.joints(0.0, 0.0, [0.0] * 6)
    s.joints(0.01, 0.01, [0.005] + [0.0] * 5)                  # moving at 0.5 rad/s, then silence
    assert s.tick(0.3) == []
    assert s.tick(0.6) == ["trip"] and "joint states stopped" in s.tripped


def test_force_and_geofence():
    s = _sup(max_force_n=20.0)
    s.force(0.0, 25.0)
    assert s.tick(0.1) == ["trip"] and "contact force" in s.tripped
    assert not s.reset(0.2)[0]                                 # still pushing
    s.force(0.3, 2.0)
    assert s.reset(0.4)[0]

    s = _sup(geofence_radius_m=1.0, geofence_rearm_m=0.1)
    s.odom(0.0, 2.0, 3.0, 0.0, 0.0)                            # the first pose is the centre
    s.odom(1.0, 3.05, 3.0, 0.0, 0.0)
    assert s.tick(1.1) == ["trip"] and "geofence" in s.tripped
    assert s.reset(1.2)[0] and abs(s.fence_radius - 1.15) < 1e-9   # re-armed farther out: drive back
    s.odom(2.0, 2.5, 3.0, 0.0, 0.0)
    assert s.tick(2.1) == [] and not s.tripped
    s.odom(3.0, 3.2, 3.0, 0.0, 0.0)
    assert s.tick(3.1) == ["trip"]


def test_trip_latches_once_and_reports_state():
    s = _sup()
    assert s.trip(1.0, "manual") and not s.trip(1.5, "again")
    st = s.state(2.0)
    assert st["ok"] is False and st["tripped"] == "manual" and st["trips"] == 1 and st["trip_age_s"] == 1.0
    assert s.reset(2.1)[0] and s.state(2.2)["ok"] is True
    assert s.reset(2.3) == (True, "not tripped")


def test_bridge_reads_the_supervisor_state():
    sb = bridge.safety_block
    assert sb({"ok": True}, 0.1, 1.0) == ""
    assert "tripped: heartbeat lost" in sb({"ok": False, "tripped": "heartbeat lost"}, 0.1, 1.0)
    assert "silent" in sb({"ok": True}, 2.0, 1.0)
    assert "no state" in sb(None, float("inf"), 1.0)
