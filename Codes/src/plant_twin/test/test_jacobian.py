"""Analytic Jacobians vs finite differences, and the speed-up they buy."""

import time

import numpy as np
from scipy.optimize import least_squares

from plant_twin.fitting import FrameObservation, PlantTwinFitter
from plant_twin.geometry import catmull_rom_basis, rotation_jacobian, rotvec_to_matrix
from plant_twin.leaf import LeafModel
from plant_twin.stem import StemModel
from test_fitting import _hole, _make_leaf, _monstera_outline, _sample_surface


def _fd(fun, x, eps=1e-7):
    f0 = fun(x)
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = eps
        J[:, i] = (fun(x + d) - fun(x - d)) / (2 * eps)
    return J


def test_rotation_jacobian_matches_fd():
    rng = np.random.default_rng(0)
    for rv in (np.zeros(3), rng.normal(size=3), 3.0 * rng.normal(size=3)):
        pts = rng.normal(size=(5, 3))
        J = rotation_jacobian(rv, pts)
        for k, p in enumerate(pts):
            Jfd = _fd(lambda r: rotvec_to_matrix(r) @ p, rv)
            assert np.allclose(J[k], Jfd, atol=1e-6), rv


def test_catmull_rom_basis_rows_sum_to_one():
    M = catmull_rom_basis(6, 8)
    assert np.allclose(M.sum(1), 1.0)
    assert np.allclose(M[0], [1, 0, 0, 0, 0, 0]) and np.allclose(M[-1], [0, 0, 0, 0, 0, 1])


def test_leaf_jacobian_matches_fd():
    leaf = LeafModel(_monstera_outline(), holes=[_hole(0.02, 0.0)], mesh_res=10)
    rng = np.random.default_rng(1)
    params = leaf.initial_params([0.1, -0.2, 0.3], [0.4, 0.05, 0.3])
    params[6:] = rng.normal(0, 0.005, leaf.n_bend)
    obs = _sample_surface(leaf, params, 150, rng) + rng.normal(0, 1e-3, (150, 3))
    prev = params + rng.normal(0, 1e-3, len(params))
    contact = leaf.vertices(params)[7] + [0, 0, 0.01]

    # Correspondences must not flip inside the FD stencil: use a tiny eps and
    # compare only rows whose nn are stable (all, in practice, at 1e-8).
    args = (obs, prev, contact)
    J = leaf.jacobian(params, *args)
    Jfd = _fd(lambda p: leaf.residuals(p, *args), params, eps=1e-8)
    assert J.shape == Jfd.shape
    # Point-to-plane rows hold the normal fixed (Gauss-Newton ICP), so their
    # bending columns differ from FD by the omitted d(normal)/d(bend) term;
    # the rigid columns and every other block must match tightly.
    n_plane = len(obs)
    assert np.abs(J[n_plane:] - Jfd[n_plane:]).max() < 1e-4
    assert np.abs(J[:n_plane, :6] - Jfd[:n_plane, :6]).max() < 1e-2
    assert np.abs(J[:n_plane] - Jfd[:n_plane]).max() < 0.2


def test_stem_jacobian_matches_fd():
    ctrl = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.25, 6)])
    stem = StemModel(ctrl)
    rng = np.random.default_rng(2)
    params = stem.initial_params() + rng.normal(0, 0.01, stem.n_params)
    obs = stem.curve(params)[::3] + rng.normal(0, 2e-3, (len(stem.curve(params)[::3]), 3))
    args = (obs, np.array([0.05, 0.02, 0.24]), params + 0.001, False)
    J = stem.jacobian(params, *args)
    Jfd = _fd(lambda p: stem.residuals(p, *args), params, eps=1e-8)
    assert J.shape == Jfd.shape
    assert np.abs(J - Jfd).max() < 1e-4


def test_analytic_fit_matches_numeric_and_is_faster():
    leaf = _make_leaf()
    rng = np.random.default_rng(3)
    rv, t = np.array([0.0, 0.3, 0.1]), np.array([0.4, 0.05, 0.3])
    truth = leaf.initial_params(rv, t)
    obs = _sample_surface(leaf, truth, 600, rng) + rng.normal(0, 5e-4, (600, 3))
    x0 = leaf.initial_params(rv + 0.05, t + 0.02)

    t0 = time.perf_counter()
    ra = least_squares(leaf.residuals, x0, jac=leaf.jacobian, args=(obs, None, None), max_nfev=60)
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    rn = least_squares(leaf.residuals, x0, args=(obs, None, None), max_nfev=60)
    tn = time.perf_counter() - t0

    ea = np.linalg.norm(leaf.vertices(ra.x) - leaf.vertices(truth), axis=1).mean()
    en = np.linalg.norm(leaf.vertices(rn.x) - leaf.vertices(truth), axis=1).mean()
    assert abs(ea - en) < 1e-3, (ea, en)      # same optimum
    assert ea < 5e-3
    assert ta < 0.5 * tn, (ta, tn)            # measured ~4-5x


def test_fitter_tracks_moving_leaf():
    """5 mm/frame in-plane motion, 10 evaluations per frame: steady-state
    error must stay well below the per-frame motion (measured ~3.7 mm)."""
    leaf = _make_leaf()
    ctrl = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.25, 6)])
    stem = StemModel(ctrl)
    rng = np.random.default_rng(3)
    fitter = PlantTwinFitter(leaf, stem)
    fitter.leaf_params = leaf.initial_params([0.05, 0.35, 0.15], [0.42, 0.07, 0.32])
    errs = []
    for k in range(15):
        truth = leaf.initial_params([0, 0.3, 0.1], [0.4 + 0.005 * k, 0.05, 0.3])
        lp = _sample_surface(leaf, truth, 600, rng)
        sp = stem.curve(stem.initial_params())[::4]
        res = fitter.step(FrameObservation(lp, sp, pulling=k > 2))
        errs.append(np.linalg.norm(leaf.vertices(res.leaf_params) - leaf.vertices(truth), axis=1).mean())
    assert np.mean(errs[5:]) < 5e-3, errs


def test_fitter_analytic_sequence_runs():
    leaf = _make_leaf()
    ctrl = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.25, 6)])
    stem = StemModel(ctrl)
    fitter = PlantTwinFitter(leaf, stem, analytic_jac=True)
    rng = np.random.default_rng(4)
    truth = leaf.initial_params([0, 0.2, 0], [0, 0, 0.25])
    for k in range(4):
        lp = _sample_surface(leaf, truth, 300, rng)
        sp = stem.curve(stem.initial_params())[::4]
        res = fitter.step(FrameObservation(lp, sp, pulling=k >= 2))
        assert np.isfinite(res.leaf_cost) and np.isfinite(res.stem_cost)
