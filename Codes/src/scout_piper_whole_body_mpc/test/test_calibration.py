"""Calibration maths (P3A.7): slip identification, TCP pivot, hand-eye."""

import numpy as np
import pytest

from scout_piper_whole_body_mpc.calibration import (excitation_plan, hand_eye, identify_slip,
                                                    pivot_calibration, rpy_from_matrix)
from scout_piper_whole_body_mpc.dynamics.piper import rpy_matrix
from scout_piper_whole_body_mpc.dynamics.scout import ScoutParams, rollout_base


def _rand_T(rng, t_scale=0.3):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = rng.normal(0, t_scale, 3)
    return T


def test_excitation_plan_returns_close_to_the_start():
    plan = excitation_plan()
    vw = np.concatenate([np.tile([[v, w]], (int(round(d / 0.1)), 1)) for d, v, w in plan])
    P = rollout_base(np.zeros(3), vw, 0.1)[0]
    assert np.abs(P[:, :2]).max() < 1.0                       # stays within 1 m
    assert np.linalg.norm(P[-1, :2]) < 0.05 and abs(P[-1, 2]) < 0.05


def test_slip_is_identified_from_timed_commands_and_external_poses():
    rng = np.random.default_rng(0)
    truth = ScoutParams(k_v=0.9, k_omega=0.75)
    plan = excitation_plan()
    dt_sim = 0.02
    vw = np.concatenate([np.tile([[v, w]], (int(round(d / dt_sim)), 1)) for d, v, w in plan])
    P = rollout_base(np.zeros(3), vw, dt_sim, truth)[0]
    t = np.arange(len(P)) * dt_sim
    cmd_t = np.arange(len(vw)) * dt_sim                       # one command per sim step
    pose_t = t[::5] + 0.003                                   # 10 Hz external reference, offset clock
    poses = P[::5] + rng.normal(0, [1e-3, 1e-3, 2e-3], (len(pose_t), 3))
    est, rep = identify_slip(cmd_t, vw, pose_t, poses, dt=0.1)
    assert abs(est.k_v - 0.9) < 0.03 and abs(est.k_omega - 0.75) < 0.03
    assert rep["steps_used"] > 100


def test_pivot_calibration_recovers_the_tool_point():
    rng = np.random.default_rng(1)
    tcp = np.array([0.002, -0.001, 0.142])
    c = np.array([0.45, 0.1, 0.2])
    Ts = []
    for _ in range(8):
        T = _rand_T(rng)
        T[:3, 3] = c - T[:3, :3] @ tcp + rng.normal(0, 3e-4, 3)
        Ts.append(T)
    r = pivot_calibration(Ts)
    assert np.allclose(r.tcp_in_flange, tcp, atol=2e-3) and np.allclose(r.pivot_in_base, c, atol=2e-3)
    assert r.rms_m < 2e-3
    with pytest.raises(ValueError):
        pivot_calibration(Ts[:2])
    with pytest.raises(ValueError, match="too similar"):
        pivot_calibration([Ts[0]] * 4)


def test_hand_eye_recovers_the_camera_mount():
    rng = np.random.default_rng(2)
    X = np.eye(4)
    X[:3, :3] = rpy_matrix(0.05, -0.1, 1.57)
    X[:3, 3] = [0.03, -0.06, 0.04]
    board = _rand_T(rng, 0.5)                                  # base -> board, fixed
    E, C = [], []
    for _ in range(10):
        Ei = _rand_T(rng)
        Ci = np.linalg.inv(Ei @ X) @ board                     # camera -> board
        noise = np.eye(4)
        noise[:3, :3] = rpy_matrix(*rng.normal(0, 0.002, 3))
        noise[:3, 3] = rng.normal(0, 5e-4, 3)
        E.append(Ei)
        C.append(Ci @ noise)
    r = hand_eye(E, C)
    assert np.allclose(r.T_flange_camera[:3, 3], X[:3, 3], atol=5e-3)
    assert np.degrees(np.arccos(np.clip((np.trace(r.T_flange_camera[:3, :3].T @ X[:3, :3]) - 1) / 2, -1, 1))) < 0.5
    assert np.allclose(rpy_matrix(*rpy_from_matrix(X[:3, :3])), X[:3, :3], atol=1e-9)
