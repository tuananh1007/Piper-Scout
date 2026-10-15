"""The torch leaf evaluation matches the numpy one, and the fitter tracks with it."""

import time

import numpy as np
import pytest

from plant_twin.fitting import FrameObservation, PlantTwinFitter
from plant_twin.leaf import LeafModel
from plant_twin.stem import StemModel

# A module-level importorskip stops ROS's launch_testing pytest plugin from
# collecting the package's other test files, so skip per test instead.
try:
    import torch  # noqa: F401
except ImportError:
    torch = None
else:
    from plant_twin.leaf_torch import TorchLeafEvaluator

pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")


def _leaf():
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    outline = np.column_stack([0.06 * np.cos(th) * (1 + 0.2 * np.cos(2 * th)), 0.06 * np.sin(th)])
    hole = np.column_stack([0.02 + 0.01 * np.cos(th[::3]), 0.01 * np.sin(th[::3])])
    return LeafModel(outline, holes=[hole])


def _sample(leaf, params, n, rng):
    v = leaf.vertices(params)
    f = leaf.faces[rng.integers(0, len(leaf.faces), n)]
    a, b = rng.random((2, n))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    return (1 - a - b)[:, None] * v[f[:, 0]] + a[:, None] * v[f[:, 1]] + b[:, None] * v[f[:, 2]]


@pytest.mark.parametrize("small_rotation", [False, True])
def test_matches_numpy(small_rotation):
    rng = np.random.default_rng(3)
    leaf = _leaf()
    truth = leaf.initial_params([0.3, -0.2, 0.4], [0.5, 0.1, 0.3])
    truth[6:] = rng.normal(0, 0.004, leaf.n_bend)
    obs = _sample(leaf, truth, 500, rng) + rng.normal(0, 0.001, (500, 3))
    x = truth + rng.normal(0, 0.01, len(truth))
    if small_rotation:
        x[:3] = [1e-8, 0.0, 0.0]                     # the small-angle branches
    prev = truth + rng.normal(0, 0.005, len(truth))
    cp = obs[17] + 0.002
    r0, J0 = leaf.evaluate(x, obs, prev, cp)
    r1, J1 = TorchLeafEvaluator(leaf).evaluate(x, obs, prev, cp)
    assert r1.shape == r0.shape and J1.shape == J0.shape
    assert np.allclose(r1, r0, atol=1e-10) and np.allclose(J1, J0, atol=1e-9)
    r2, J2 = TorchLeafEvaluator(leaf).evaluate(x, obs, want_jac=False)
    assert J2 is None and np.allclose(r2, leaf.evaluate(x, obs, want_jac=False)[0], atol=1e-10)


def test_fitter_with_the_torch_backend_tracks_like_numpy():
    rng = np.random.default_rng(5)
    leaf_np, leaf_t = _leaf(), _leaf()
    stem = lambda: StemModel(np.linspace([0, -0.1, -0.2], [0, -0.06, 0], 6), radius_m=0.003)  # noqa: E731
    fits = {"numpy": PlantTwinFitter(leaf_np, stem()), "torch": PlantTwinFitter(leaf_t, stem(), leaf_backend="torch")}
    truth = leaf_np.initial_params([0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    for f in fits.values():
        f.leaf_params = truth.copy()
    times = {k: 0.0 for k in fits}
    for k in range(6):
        truth = truth.copy()
        truth[3] += 0.004                             # 4 mm per frame in-plane
        pts = _sample(leaf_np, truth, 600, rng) + rng.normal(0, 0.001, (600, 3))
        stem_pts = np.linspace([0, -0.1, -0.2], [0, -0.06, 0], 40) + rng.normal(0, 0.001, (40, 3))
        for name, f in fits.items():
            t0 = time.perf_counter()
            f.step(FrameObservation(pts, stem_pts, None, False))
            times[name] += time.perf_counter() - t0
    err = {k: np.abs(leaf_np.vertices(f.leaf_params) - leaf_np.vertices(truth)).max() for k, f in fits.items()}
    assert err["torch"] < 0.004 and abs(err["torch"] - err["numpy"]) < 1e-6
    print(f"6 frames: numpy {1e3 * times['numpy'] / 6:.1f} ms/frame, torch (CPU float64) "
          f"{1e3 * times['torch'] / 6:.1f} ms/frame; max vertex error {1e3 * err['torch']:.2f} mm")
