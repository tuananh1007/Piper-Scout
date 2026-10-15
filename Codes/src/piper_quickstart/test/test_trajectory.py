import numpy as np
import pytest

from piper_quickstart.trajectory import driver_to_urdf, max_abs_error, sample_trajectory


def test_sample_trajectory_interpolates_and_clamps():
    times = np.array([0.0, 1.0, 3.0])
    pos = np.array([[0.0, 10.0], [1.0, 10.0], [3.0, 14.0]])
    assert sample_trajectory(times, pos, -1.0).tolist() == [0.0, 10.0]
    assert sample_trajectory(times, pos, 0.5).tolist() == [0.5, 10.0]
    assert sample_trajectory(times, pos, 2.0).tolist() == [2.0, 12.0]
    assert sample_trajectory(times, pos, 9.0).tolist() == [3.0, 14.0]


def test_sample_trajectory_duplicate_times():
    times = np.array([0.0, 1.0, 1.0, 2.0])
    pos = np.array([[0.0], [1.0], [5.0], [6.0]])
    assert sample_trajectory(times, pos, 1.5).tolist() == [5.5]


def test_sample_trajectory_rejects_bad_shapes():
    with pytest.raises(ValueError):
        sample_trajectory(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5)


def test_driver_to_urdf_splits_gripper_opening():
    names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    out = driver_to_urdf(names, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07], gripper_scale=2.0)
    assert out["joint1"] == 0.1 and out["joint6"] == 0.6
    assert out["joint7"] == pytest.approx(0.035)
    assert out["joint8"] == pytest.approx(-0.035)


def test_max_abs_error():
    assert max_abs_error({"a": 1.0, "b": 2.0}, {"a": 1.5, "b": 1.0}, ["a", "b"]) == 1.0
    assert max_abs_error({"a": 1.0}, {}, ["a"]) == float("inf")
