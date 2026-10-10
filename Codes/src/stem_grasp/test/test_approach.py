"""Iterative final approach (P0.4.13), stem_grasp.approach."""

import pytest

import numpy as np

from stem_grasp.approach import (ApproachConfig, GripperCloseMonitor, IterativeApproach, RetreatMonitor,
                                 ServoGates, servo_gate)

DT = 0.1


def run(app, d0=0.12, error=2.0, pixels=500, force=0.0, steps=2000, hook=None):
    """Kinematic loop: the gripper moves at the commanded speed."""
    d, t, phases = d0, 0.0, []
    for k in range(steps):
        e, px = (error, pixels) if hook is None else hook(k, d)
        st = app.update(t, e, d, px, force)
        phases.append(st.phase)
        if st.phase in ("done", "abort"):
            return st, d, phases
        d -= st.speed * DT
        t += DT
    return st, d, phases


def test_reaches_the_grasp_point_in_steps():
    app = IterativeApproach(ApproachConfig(), start_t=0.0)
    st, d, phases = run(app)
    assert st.phase == "done" and d <= 0.01
    assert app.steps == 3                       # 0.12 m in 0.05 m steps to within 1 cm
    # the gripper pauses (align) between steps
    switches = [p for i, p in enumerate(phases[1:], 1) if p != phases[i - 1]]
    assert switches.count("advance") == 3


def test_never_advances_while_the_stem_is_off_the_axis():
    app = IterativeApproach(ApproachConfig(), start_t=0.0)
    st, d, _ = run(app, error=20.0, steps=300)
    assert st.phase == "align" and d == pytest.approx(0.12)


def test_interrupted_step_is_resumed_not_counted_again():
    app = IterativeApproach(ApproachConfig(), start_t=0.0)
    spiked = []

    def hook(k, d):
        if 0.09 < d < 0.095 and not spiked:    # mid first step: the stem drifts off the axis
            spiked.append(k)
            return 30.0, 500
        return 2.0, 500

    st, _, _ = run(app, hook=hook)
    assert spiked and st.phase == "done" and app.steps == 3


@pytest.mark.parametrize("cfg, kw, reason", [
    (ApproachConfig(), {"d0": 0.30}, "more than"),                          # too far to start
    (ApproachConfig(step_m=0.01, max_steps=2), {}, "after 2 steps"),
    (ApproachConfig(), {"pixels": 10}, "stem mask"),
    (ApproachConfig(timeout_s=3.0), {"error": 20.0}, "within 3.0 s"),
])
def test_aborts(cfg, kw, reason):
    st, _, _ = run(IterativeApproach(cfg, start_t=0.0), **kw)
    assert st.phase == "abort" and reason in st.reason


def test_a_brief_mask_dropout_only_pauses():
    app = IterativeApproach(ApproachConfig(), start_t=0.0)
    st, _, _ = run(app, hook=lambda k, d: (2.0, 10 if 20 <= k < 30 else 500))
    assert st.phase == "done"


def test_contact_ends_the_approach():
    st, d, _ = run(IterativeApproach(ApproachConfig(), start_t=0.0), force=0.2)
    assert st.phase == "done" and "contact" in st.reason and d == pytest.approx(0.12)


def test_a_single_force_spike_is_not_a_contact():
    spikes = {5, 40, 60}                                   # one-sample spikes (a noisy force estimate)
    app = IterativeApproach(ApproachConfig(), start_t=0.0)
    d, t = 0.12, 0.0
    for k in range(400):
        st = app.update(t, 2.0, d, 500, 0.3 if k in spikes else 0.0)
        if st.phase in ("done", "abort"):
            break
        d -= st.speed * DT
        t += DT
    assert st.phase == "done" and "grasp point" in st.reason   # arrived; the spikes did not end it


def close(widths, dt=0.1, **kw):
    """Feed a measured opening sequence; return (decision, time, width)."""
    mon = GripperCloseMonitor(start_t=0.0, **kw)
    for k, w in enumerate(widths):
        d = mon.update(k * dt, w)
        if d != "wait":
            return d, k * dt, mon.width
    return "wait", None, mon.width


def test_gripper_settling_on_the_stem_is_a_grasp():
    closing = [0.03 - 0.005 * k for k in range(5)] + [0.008] * 10     # stops at an 8 mm stem
    decision, t, w = close(closing)
    assert decision == "grasped" and w == pytest.approx(0.008) and t >= 0.5


@pytest.mark.parametrize("widths, kw, expected", [
    ([0.03 - 0.005 * k for k in range(6)] + [0.0005] * 10, {}, "empty"),   # closed on nothing
    ([None] * 60, {}, "timeout"),                                            # no finger joint states
    ([0.03 - 0.0004 * k for k in range(80)], {"timeout_s": 3.0}, "timeout"),  # never stops moving
])
def test_gripper_close_failures(widths, kw, expected):
    assert close(widths, **kw)[0] == expected


def test_retreat_counts_only_motion_back_along_the_axis():
    axis = np.array([1.0, 0.0, 0.0])                 # the gripper pointed +x at the stem
    mon = RetreatMonitor(start_t=0.0, start_tcp=np.zeros(3), axis=axis, distance_m=0.10)
    assert mon.update(1.0, np.array([0.0, 0.2, 0.0])) == "wait"    # sideways is not back
    assert mon.update(2.0, np.array([-0.05, 0.0, 0.0])) == "wait"
    assert mon.update(3.0, np.array([-0.101, 0.0, 0.0])) == "done" and mon.moved == pytest.approx(0.101)
    stuck = RetreatMonitor(start_t=0.0, start_tcp=np.zeros(3), axis=axis, timeout_s=5.0)
    assert stuck.update(6.0, np.zeros(3)) == "timeout"
    assert stuck.update(6.0, None) == "timeout"


@pytest.mark.parametrize("approaching, force, force_age, mask_age, expected", [
    (False, 0.3, 0.01, 0.05, None),                     # all well
    (False, 2.5, 0.01, 0.05, "retract"),                # force limit: back out
    (True, 2.5, 0.01, 9.0, "retract"),                  # force first, whatever else is wrong
    (False, 0.0, 0.8, 0.05, "stop"),                    # the force source fell silent
    (False, 0.0, None, 0.05, None),                     # no force source at all: no force gate
    (False, 0.0, None, 2.5, "reground"),                # target lost while servoing
    (True, 0.0, None, 2.5, None),                       # approaching: the approach's own mask abort
])
def test_servo_gates(approaching, force, force_age, mask_age, expected):
    out = servo_gate(ServoGates(), approaching, force, force_age, mask_age)
    assert (out[0] if out else None) == expected
    assert servo_gate(ServoGates(force_max_age_s=0.0, lost_target_s=0.0), False, 0.0, 9.0, 9.0) is None   # gates off


@pytest.mark.parametrize("window, arrives", [(0.0, False), (1.5, True)])
def test_a_swaying_stem_needs_the_alignment_window(window, arrives):
    """Error vector swinging +-14 px at 0.7 Hz around a 2 px offset: never within
    8 px for 0.4 s sample by sample, aligned on average."""
    app = IterativeApproach(ApproachConfig(align_window_s=window, timeout_s=40.0), start_t=0.0)
    d, t, st = 0.12, 0.0, None
    for _ in range(int(40.0 / DT)):
        e = np.array([2.0 + 14.0 * np.sin(2 * np.pi * 0.7 * t), 0.0])
        st = app.update(t, float(np.linalg.norm(e)), d, 500, 0.0, error_uv=e)
        if st.phase in ("done", "abort"):
            break
        d -= st.speed * DT
        t += DT
    assert (st.phase == "done") == arrives, (st.phase, st.reason, d)
