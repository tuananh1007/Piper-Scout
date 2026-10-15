"""Goal bookkeeping of the controller: repeated goals keep the warm start."""

import numpy as np

from scout_piper_whole_body_mpc.costs.terms import Goal, same_goal


def test_repeated_goal_is_the_same_goal():
    a = Goal(p=np.array([1.0, 0.2, 0.45]), approach_axis=np.array([0.0, 1.0, 0.0]))
    assert same_goal(a, Goal(p=a.p + 5e-4, approach_axis=a.approach_axis.copy()))
    assert not same_goal(None, a)
    assert not same_goal(a, Goal(p=a.p + 0.01, approach_axis=a.approach_axis))
    assert not same_goal(a, Goal(p=a.p, approach_axis=np.array([1.0, 0.0, 0.0])))
    assert not same_goal(a, Goal(p=a.p))                       # axis dropped
    assert same_goal(Goal(p=a.p), Goal(p=a.p))


def test_full_pose_goal_compares_the_rotation():
    R = np.eye(3)
    roll = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # 90 deg about z
    a = Goal(p=np.array([1.0, 0.2, 0.45]), R=R, approach_axis=R[:, 2].copy())
    assert same_goal(a, Goal(p=a.p.copy(), R=R.copy(), approach_axis=R[:, 2].copy()))
    # same approach axis, different roll: a new goal
    assert not same_goal(a, Goal(p=a.p, R=roll, approach_axis=roll[:, 2].copy()))
    assert not same_goal(a, Goal(p=a.p, approach_axis=R[:, 2].copy()))   # rotation dropped


def test_quaternion_matrix():
    pytest = __import__("pytest")
    try:
        from scout_piper_whole_body_mpc.controller_node import _quat_matrix
    except ImportError:                                  # no rclpy in this environment
        pytest.skip("rclpy not available")

    class Q:
        def __init__(self, w, x, y, z):
            self.w, self.x, self.y, self.z = w, x, y, z

    s = np.sqrt(0.5)
    R = _quat_matrix(Q(2 * s, 0.0, 0.0, 2 * s))          # 90 deg about z, unnormalised
    assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert _quat_matrix(Q(0.0, 0.0, 0.0, 0.0)) is None
