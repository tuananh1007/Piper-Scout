"""Simulated final-approach trials (benchmarks/servo_trials.py): the harness itself."""

import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for pkg in ("scout_piper_whole_body_mpc", os.path.join("scout_piper_scene_repr", "python")):   # source tree fallback
    src = os.path.join(HERE, "..", "..", pkg)
    if src not in sys.path:
        sys.path.append(src)
spec = importlib.util.spec_from_file_location("servo_trials", os.path.join(HERE, "..", "benchmarks", "servo_trials.py"))
st = importlib.util.module_from_spec(spec)
sys.modules["servo_trials"] = st                     # dataclasses look their module up there
spec.loader.exec_module(st)


def test_trials_are_reproducible_and_start_at_the_pre_grasp():
    a, b = st.Trial.draw(3), st.Trial.draw(3)
    assert np.allclose(a.calib_xyz, b.calib_xyz) and a.sway_amp == b.sway_amp
    kin = st.WholeBodyModel().kin
    g = st.Scene(a).grasp_point(0.0)
    goal = g - 0.12 * np.array([1.0, 0.0, 0.0]) + a.start_offset
    q = st.ik(kin, goal, a.axis_err_rot @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.2, -1.0, 0.0, 0.5, 0.0]))
    assert np.linalg.norm(kin.tcp(q)[:3, 3] - goal) < 1e-4


def test_contact_model_fingers_and_palm():
    sc = st.Scene(st.Trial.draw(0))
    sc.t.sway_amp = 0.0
    g = sc.grasp_point(0.0)
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])   # TCP z along base x
    T[:3, 3] = g
    assert sc.force(T, 0.0, 150.0) == 0.0                                       # centred between the fingers
    T[:3, 3] = g - [0.0, st.OPEN_W / 2, 0.0]                                     # a finger on the stem
    assert sc.force(T, 0.0, 150.0) > 0.5
    T[:3, 3] = g + [0.05, 0.0, 0.0]                                              # pushed 5 cm past it: palm
    assert sc.force(T, 0.0, 150.0) == pytest.approx(150.0 * (0.05 + st.PALM_S), abs=1e-6)


@pytest.mark.parametrize("controller", ["ibvs", "mppi"])
def test_an_easy_trial_ends_at_the_grasp_point(controller):
    t = st.Trial.draw(0)
    t.sway_amp, t.calib_xyz, t.calib_rot, t.est_bias, t.neighbour = 0.0, np.zeros(3), np.eye(3), np.zeros(3), None
    r = st.run_trial(t, controller, samples=64, timeout_s=30.0)
    assert r["success"] and r["outcome"] == "done" and r["miss_mm"] < 5.0, r
