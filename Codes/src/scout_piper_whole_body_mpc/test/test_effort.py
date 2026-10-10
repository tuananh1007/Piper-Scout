"""Contact force from joint efforts: gravity model, calibration fit, force estimate (no ROS)."""

import os
import re

import numpy as np
import pytest

from scout_piper_whole_body_mpc.dynamics.effort import (GRAVITY, INERTIALS, ContactForceEstimator, EffortModel,
                                                         effort_excitation, excitation_check, fit_effort_model,
                                                         gravity_torques)
from scout_piper_whole_body_mpc.dynamics.piper import PiperKinematics, rpy_matrix

XACRO = os.path.join(os.path.dirname(__file__), "..", "..", "scout_piper_description", "urdf", "_piper_arm.xacro")
KIN = PiperKinematics()


def _inertial(s, link):
    m = re.search(r'name="\$\{prefix\}' + link + r'">\s*<inertial>\s*<origin\s+xyz="([^"]+)"[^>]*/>\s*<mass\s+value="([^"]+)"',
                  s, re.S)
    return np.array(m.group(1).split(), float), float(m.group(2))


def _joint_origin(s, joint):
    m = re.search(r'name="\$\{prefix\}' + joint + r'"\s+type="\w+">\s*<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)"', s, re.S)
    return np.array(m.group(1).split(), float), np.array(m.group(2).split(), float)


def test_inertials_match_the_xacro():
    s = open(XACRO, encoding="utf-8").read()
    for (fi, m, com), link in zip(INERTIALS[:5], ["link1", "link2", "link3", "link4", "link5"]):
        c, mass = _inertial(s, link)
        assert fi == int(link[-1]) and mass == pytest.approx(m) and np.allclose(c, com)
    # link6 + gripper base (fixed at link6's origin) + both fingers lumped on link6
    parts = [_inertial(s, "link6"), _inertial(s, "gripper_base")]
    for j, link in (("joint7", "link7"), ("joint8", "link8")):
        xyz, rpy = _joint_origin(s, j)
        c, mass = _inertial(s, link)
        parts.append((xyz + rpy_matrix(*rpy) @ c, mass))
    M = sum(m for _, m in parts)
    C = sum(m * c for c, m in parts) / M
    fi, m, com = INERTIALS[5]
    assert fi == 6 and m == pytest.approx(M) and np.allclose(com, C, atol=2e-4), (M, C)


def test_gravity_torques_are_the_gradient_of_potential_energy():
    rng = np.random.default_rng(0)
    q = rng.uniform(KIN.lower, KIN.upper)

    def V(q):
        F = KIN.link_frames(q)
        return sum(m * GRAVITY * (F[fi, :3, :3] @ np.array(c) + F[fi, :3, 3])[2] for fi, m, c in INERTIALS)

    eps = 1e-6
    num = [(V(q + eps * e) - V(q - eps * e)) / (2 * eps) for e in np.eye(6)]
    assert np.allclose(gravity_torques(KIN, q), num, atol=1e-6)
    assert abs(gravity_torques(KIN, q)[0]) < 1e-9           # joint 1 axis is vertical


def _free_motion(n=600, seed=1, a=(1.0, 1.0, 1.1, 0.95, 1.05, 1.0)):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(KIN.lower * 0.8, KIN.upper * 0.8, (n, 6))
    QD = rng.normal(0, 0.2, (n, 6))
    true = EffortModel(a=list(a), b=[0.1, -0.2, 0.05, 0.0, 0.02, -0.01], c=[0.3, 0.5, 0.2, 0.05, 0.05, 0.02],
                       d=[0.4, 0.6, 0.3, 0.1, 0.1, 0.05])
    TAU = true.free_torque(KIN, Q, QD) + rng.normal(0, 0.02, (n, 6))
    return Q, QD, TAU, true


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_fit_recovers_the_effort_model_including_a_reversed_sign(sign):
    a = sign * np.array([1.0, 1.0, 1.1, 0.95, 1.05, 1.0])
    Q, QD, TAU, true = _free_motion(a=tuple(a))
    m = fit_effort_model(KIN, Q, QD, TAU)
    assert m.identified == [False, True, True, True, True, False]   # gravity loads joints 2-5 only
    assert np.allclose(m.a[1:5], a[1:5], atol=0.05)
    assert np.allclose([m.a[0], m.a[5]], sign * 1.025, atol=0.05)   # median gain of joints 2-5
    for k in ("b", "c", "d"):
        assert np.allclose(getattr(m, k), getattr(true, k), atol=0.05), (k, getattr(m, k))
    assert np.all(np.asarray(m.sigma) < 0.03) and m.samples == len(Q)


def test_estimator_recovers_an_applied_force_and_reports_its_noise():
    Q, QD, TAU, true = _free_motion()
    est = ContactForceEstimator(KIN, fit_effort_model(KIN, Q, QD, TAU), filter_s=0.0)
    rng = np.random.default_rng(3)
    q = np.array([0.0, 1.2, -1.0, 0.0, 0.5, 0.0])
    F_true = np.array([1.5, -0.8, 2.0])                      # N on the TCP
    J = KIN.jacobian(q)[:3]
    a = np.asarray(true.a)
    errs = []
    for _ in range(200):
        tau = true.free_torque(KIN, q, np.zeros(6)) - a * (J.T @ F_true) + rng.normal(0, 0.02, 6)
        errs.append(est.update(q, np.zeros(6), tau, 0.01).force - F_true)
    errs = np.array(errs)
    sd = est.force_noise(q)
    assert np.allclose(errs.mean(0), 0.0, atol=3 * sd.max() / np.sqrt(len(errs)) + 0.05)
    assert np.all(errs.std(0) < 2.0 * sd + 1e-3)             # the predicted noise is the real one
    tau0 = true.free_torque(KIN, q, np.zeros(6))
    assert est.update(q, np.zeros(6), tau0, 0.01).magnitude < 4 * sd.max()   # free: ~0


def test_tare_zeroes_a_constant_offset_and_the_model_round_trips(tmp_path):
    est = ContactForceEstimator(KIN, EffortModel(), filter_s=0.0)
    q = np.array([0.0, 1.2, -1.0, 0.0, 0.5, 0.0])
    tau = gravity_torques(KIN, q) + 0.3                     # a bias the default model does not know
    assert est.update(q, np.zeros(6), tau, 0.01).magnitude > 0.5
    est.tare()
    assert est.update(q, np.zeros(6), tau, 0.01).magnitude < 1e-6
    m = EffortModel(a=[1.1] * 6, samples=10)
    m.save(str(tmp_path / "e.json"))
    assert EffortModel.load(str(tmp_path / "e.json")).a == [1.1] * 6


def test_torque_with_a_force_is_the_free_torque_minus_the_jacobian_transpose():
    m = EffortModel(a=[-1.0] * 6, d=[0.2] * 6)
    q, qd, F = np.array([0.1, 1.0, -0.9, 0.2, 0.4, 0.3]), np.full(6, 0.1), np.array([0.5, -1.0, 2.0])
    J = KIN.jacobian(q)[:3]
    assert np.allclose(m.torque(KIN, q, qd, F), m.free_torque(KIN, q, qd) + J.T @ F)
    Q = np.stack([q, q])
    assert m.torque(KIN, Q, np.stack([qd, qd]), F).shape == (2, 6)


def test_excitation_starts_and_ends_at_rest_and_its_velocity_is_the_derivative():
    q0 = np.array([0.0, 0.8, -1.2, 0.0, 0.45, 0.0])
    for t in (0.0, 120.0):
        q, qd = effort_excitation(q0, t, 120.0)
        assert np.allclose(q, q0) and np.allclose(qd, 0.0, atol=1e-12)
    for t in (2.0, 40.0, 117.0):                             # in the ramps and the middle
        num = (effort_excitation(q0, t + 1e-5, 120.0)[0] - effort_excitation(q0, t - 1e-5, 120.0)[0]) / 2e-5
        assert np.allclose(effort_excitation(q0, t, 120.0)[1], num, atol=1e-6)


def test_excitation_check_accepts_the_suggested_centre_and_rejects_the_servo_pose():
    from scout_piper_whole_body_mpc.calibration_nodes import EFFORT_CENTER, SERVO_POSE
    ok = excitation_check(np.array(EFFORT_CENTER), 120.0)
    assert ok["within_limits"] and ok["min_deck_clearance_m"] > 0.1 and ok["peak_speed_rad_s"] < 0.3
    assert ok["min_limit_margin_rad"] > 0.1                  # moveit_servo halts 0.1 rad before a limit
    assert min(ok["gravity_spread_nm"][1:5]) >= 0.1          # joints 2-5 identifiable from it
    low = excitation_check(np.array(SERVO_POSE), 120.0)
    assert low["min_deck_clearance_m"] < 0.05                # the TCP dips to the top plate


def test_calibration_on_the_excitation_and_the_report():
    from scout_piper_whole_body_mpc.calibration_nodes import EFFORT_CENTER, fit_effort_report
    rng = np.random.default_rng(4)
    true = EffortModel(a=[-1.0] * 6, b=[0.05, -0.1, 0.05, 0.0, 0.02, 0.0], c=[0.3, 0.4, 0.2, 0.05, 0.05, 0.02],
                       d=[0.3, 0.5, 0.3, 0.1, 0.1, 0.05])
    ts = np.arange(0.0, 120.0, 0.05)
    Q, QD = (np.array(v) for v in zip(*(effort_excitation(np.array(EFFORT_CENTER), t, 120.0) for t in ts)))
    TAU = true.torque(KIN, Q, QD) + rng.normal(0, 0.02, Q.shape)
    model, rep = fit_effort_report(KIN, Q, QD, TAU)
    assert rep["identified"] == [False, True, True, True, True, False]
    assert np.allclose(model.a, -1.0, atol=0.05)
    assert max(rep["holdout_sigma_nm"]) < 0.05 and max(np.abs(rep["holdout_bias_nm"])) < 0.05
    assert 0.0 < rep["threshold_3sigma_n"] < 1.0             # 0.02 N·m effort noise -> sub-newton
