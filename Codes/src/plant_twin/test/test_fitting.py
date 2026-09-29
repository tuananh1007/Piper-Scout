"""Synthetic tests for the leaf/stem twin — run without ROS:

    cd Codes && python -m pytest src/plant_twin/test -q
"""

import numpy as np

from plant_twin.fitting import FrameObservation, PlantTwinFitter
from plant_twin.geometry import polyline_length, rotvec_to_matrix
from plant_twin.leaf import LeafModel
from plant_twin.stem import StemModel


def _monstera_outline(n=40, r=0.06):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # heart-ish outline with the petiole notch at the origin side
    x = r * np.cos(th) * (1 + 0.2 * np.cos(2 * th))
    y = r * np.sin(th) * (1 + 0.1 * np.sin(th))
    return np.column_stack([x, y])


def _hole(cx, cy, r=0.01, n=12):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


def _make_leaf():
    return LeafModel(_monstera_outline(), holes=[_hole(0.02, 0.0), _hole(-0.02, 0.01)])


def _sample_surface(leaf, params, n, rng):
    """Random points on the mesh (uniform barycentric), like a depth cloud."""
    v = leaf.vertices(params)
    f = leaf.faces[rng.integers(0, len(leaf.faces), n)]
    a, b = rng.random((2, n))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    return (1 - a - b)[:, None] * v[f[:, 0]] + a[:, None] * v[f[:, 1]] + b[:, None] * v[f[:, 2]]


def test_rotvec_matrix_is_orthonormal():
    R = rotvec_to_matrix([0.3, -0.2, 0.5])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_leaf_mesh_has_holes_and_outline():
    leaf = _make_leaf()
    assert len(leaf.faces) > 50
    # no vertex sits inside a hole
    for h in leaf.holes:
        c = h.mean(0)
        d = np.linalg.norm(leaf.rest_xy - c, axis=1)
        assert d.min() > 0.005
    # bending lifts vertices along local z only
    bend = np.full(leaf.n_bend, 0.01)
    v = leaf.local_vertices(bend)
    assert np.allclose(v[:, :2], leaf.rest_xy)
    assert v[:, 2].max() > 0.005


def test_leaf_recovers_rigid_pose():
    leaf = _make_leaf()
    rv, t = np.array([0.0, 0.3, 0.1]), np.array([0.4, 0.05, 0.3])
    truth = leaf.initial_params(rv, t)
    rng = np.random.default_rng(0)
    obs = _sample_surface(leaf, truth, 600, rng)
    obs = obs + rng.normal(0, 5e-4, obs.shape)

    fit = leaf.initial_params(rv + 0.05, t + 0.02)
    from scipy.optimize import least_squares
    r = least_squares(leaf.residuals, fit, args=(obs, None, None), max_nfev=60)
    err = np.linalg.norm(leaf.vertices(r.x) - leaf.vertices(truth), axis=1)
    assert err.mean() < 3e-3


def test_leaf_bends_toward_contact_without_stretching():
    leaf = _make_leaf()
    p0 = leaf.initial_params()
    v0 = leaf.vertices(p0)
    tip_i = int(np.argmax(v0[:, 0]))
    contact = v0[tip_i] + np.array([0.0, 0.0, 0.02])   # finger pushes tip up 2 cm
    from scipy.optimize import least_squares
    r = least_squares(leaf.residuals, p0, args=(np.zeros((0, 3)), None, contact),
                      max_nfev=80)
    v = leaf.vertices(r.x)
    assert np.linalg.norm(v[tip_i] - contact) < 5e-3
    el = np.linalg.norm(v[leaf.edges[:, 0]] - v[leaf.edges[:, 1]], axis=1)
    assert np.abs(el / leaf.rest_edge_len - 1).max() < 0.15


def test_stem_keeps_base_and_length_while_following_tip():
    ctrl = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.25, 6)])
    stem = StemModel(ctrl)
    p0 = stem.initial_params()
    tip = np.array([0.05, 0.03, 0.23])
    from scipy.optimize import least_squares
    r = least_squares(stem.residuals, p0,
                      args=(np.zeros((0, 3)), tip, None, True), max_nfev=100)
    c = stem.ctrl(r.x)
    assert np.linalg.norm(c[0] - stem.anchor) < 2e-3
    assert np.linalg.norm(c[-1] - tip) < 5e-3
    L = polyline_length(stem.curve(r.x))
    assert abs(L - stem.rest_length) / stem.rest_length < 0.05


def test_stem_stationary_before_pull():
    ctrl = np.column_stack([np.zeros(5), np.zeros(5), np.linspace(0, 0.2, 5)])
    stem = StemModel(ctrl)
    noisy = stem.curve(stem.initial_params()) + np.random.default_rng(1).normal(0, 0.01, (len(stem.curve(stem.initial_params())), 3))
    from scipy.optimize import least_squares
    r = least_squares(stem.residuals, stem.initial_params(),
                      args=(noisy, None, None, False), max_nfev=60)
    assert np.abs(r.x - stem.initial_params()).max() < 5e-3


def test_full_sequence_tracks_pull():
    leaf = _make_leaf()
    ctrl = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.25, 6)])
    stem = StemModel(ctrl)
    leaf_rv, leaf_t = np.array([0.0, 0.2, 0.0]), np.array([0.0, 0.0, 0.25])
    fitter = PlantTwinFitter(leaf, stem)
    rng = np.random.default_rng(2)

    prev_tip = None
    for k in range(6):
        pull = 0.0 if k < 2 else 0.01 * (k - 1)
        truth = leaf.initial_params(leaf_rv, leaf_t + [pull, 0, 0])
        lp = _sample_surface(leaf, truth, 300, rng)
        lp = lp + rng.normal(0, 5e-4, lp.shape)
        sp = stem.curve(stem.initial_params())[::4]
        sp = sp + rng.normal(0, 1e-3, sp.shape)
        res = fitter.step(FrameObservation(lp, sp, pulling=k >= 2))
        tip = leaf.tip_point(res.leaf_params)
        if prev_tip is not None:
            assert np.linalg.norm(tip - prev_tip) < 0.03  # smooth motion
        prev_tip = tip
        stem_tip = stem.ctrl(res.stem_params)[-1]
        assert np.linalg.norm(stem_tip - tip) < 0.01

    out = fitter.export()
    assert out["leaf_vertices"].shape[1] == 3 and out["stem_curve"].shape[1] == 3
