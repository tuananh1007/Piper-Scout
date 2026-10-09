"""Iterative final approach (P0.4.13), stem_grasp.approach."""

import pytest

from stem_grasp.approach import ApproachConfig, GripperCloseMonitor, IterativeApproach

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
