"""The damped Gauss-Newton solver, on its own and against scipy on the leaf."""

import time

import numpy as np
from scipy.optimize import least_squares

from plant_twin.fitting import FrameObservation, PlantTwinFitter
from plant_twin.solver import levenberg_marquardt
from plant_twin.stem import StemModel
from test_fitting import _make_leaf, _sample_surface


def test_linear_problem_solves_in_one_step():
    rng = np.random.default_rng(0)
    A, b = rng.normal(size=(50, 6)), rng.normal(size=50)
    res = levenberg_marquardt(lambda x: (A @ x - b, A), np.zeros(6), lam=1e-9, ftol=1e-12)
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
    assert res.n_iter <= 2 and res.status == "converged"
    assert np.allclose(res.x, x_ref, atol=1e-6)


def test_stalls_cleanly_when_no_step_helps():
    # Jacobian that points the wrong way: every trial raises the cost.
    def bad(x):
        return np.array([x[0] - 1.0]), np.array([[-1.0]])
    res = levenberg_marquardt(bad, np.array([0.0]), max_iter=5)
    assert res.status == "stalled"
    assert np.allclose(res.x, 0.0)          # stayed at the start point


def test_lm_reaches_trf_optimum_on_leaf():
    leaf = _make_leaf()
    rng = np.random.default_rng(5)
    truth = leaf.initial_params([0.0, 0.3, 0.1], [0.4, 0.05, 0.3])
    obs = _sample_surface(leaf, truth, 600, rng) + rng.normal(0, 5e-4, (600, 3))
    x0 = leaf.initial_params([0.05, 0.35, 0.15], [0.42, 0.07, 0.32])

    lm = levenberg_marquardt(lambda x: leaf.evaluate(x, obs), x0, max_iter=40, ftol=1e-8)
    trf = least_squares(leaf.residuals, x0, jac=leaf.jacobian, args=(obs,), max_nfev=80)
    # ICP re-picks correspondences every step, so the cost keeps creeping by
    # tiny amounts and a 1e-8 tolerance runs to the cap; what matters is that
    # both solvers land on the same surface.
    assert lm.status in ("converged", "max_iter")
    v_lm, v_trf = leaf.vertices(lm.x), leaf.vertices(trf.x)
    assert np.linalg.norm(v_lm - v_trf, axis=1).mean() < 1e-3
    assert abs(lm.cost - trf.cost) < 0.05 * trf.cost


def test_lm_frame_is_faster_than_trf():
    leaf = _make_leaf()
    ctrl = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.25, 6)])
    stem = StemModel(ctrl)

    def run(solver):
        rng = np.random.default_rng(3)
        f = PlantTwinFitter(leaf, stem, max_iter=10, solver=solver)
        f.leaf_params = leaf.initial_params([0.05, 0.35, 0.15], [0.42, 0.07, 0.32])
        times, errs = [], []
        for k in range(12):
            truth = leaf.initial_params([0, 0.3, 0.1], [0.4 + 0.005 * k, 0.05, 0.3])
            lp = _sample_surface(leaf, truth, 600, rng)
            sp = stem.curve(stem.initial_params())[::4]
            t0 = time.perf_counter()
            res = f.step(FrameObservation(lp, sp, pulling=k > 2))
            times.append(time.perf_counter() - t0)
            errs.append(np.linalg.norm(leaf.vertices(res.leaf_params)
                                       - leaf.vertices(truth), axis=1).mean())
        return np.median(times[3:]), np.mean(errs[5:])

    t_lm, e_lm = run("lm")
    t_trf, e_trf = run("trf")
    assert e_lm < 5e-3 and e_trf < 5e-3
    assert t_lm < 0.7 * t_trf, (t_lm, t_trf)   # measured ~0.3-0.4
