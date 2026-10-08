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
