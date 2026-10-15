import numpy as np

from piper_quickstart.export_episode import interp_joints


def test_interp_joints_values_and_gaps():
    stamps = np.array([0, 10, 20], dtype=np.int64)
    pos = np.array([[0.0], [1.0], [3.0]])
    q, gap = interp_joints(stamps, pos, np.array([5, 15, 20, 25, -1], dtype=np.int64))
    assert q[0, 0] == 0.5 and q[1, 0] == 2.0 and q[2, 0] == 3.0
    assert np.isnan(q[3, 0]) and np.isnan(q[4, 0])
    assert gap.tolist() == [5, 5, 0, 5, 1]
