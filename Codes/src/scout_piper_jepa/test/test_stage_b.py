"""Stage B / C pieces on the synthetic world (no ROS; torch tests skip without torch).

    cd Codes/src/scout_piper_jepa && python -m pytest test -q
"""

import numpy as np
import pytest

from scout_piper_jepa.action import action_embedding, action_from_states
from scout_piper_jepa.prediction_metrics import score_predictions
from scout_piper_jepa.predictive_cost import JepaVisibilityCost
from scout_piper_jepa.predictor import (OracleStatePredictor, PersistencePredictor,
                                        StateConditionedPredictor)
from scout_piper_jepa.readout import ReadoutConfig, readout, readout_sequence
from scout_piper_jepa.synthetic import (DISTRACTOR, LEAF, TARGET, make_flower_scene,
                                        synthetic_episodes, target_truth)

WORLD = make_flower_scene(feature_noise=0.0)
R_TARGET = WORLD.features[WORLD.labels == TARGET][0]


def look_along_x(x, y, z, yaw=0.0):
    """Camera at (x, y, z) looking along world +x rotated by yaw (optical z forward)."""
    c, s = np.cos(yaw), np.sin(yaw)
    fwd, right, down = np.array([c, s, 0.0]), np.array([s, -c, 0.0]), np.array([0, 0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, (x, y, z)
    return T


# ------------------------------------------------------------- the world
CLEAR = (0.25, -0.30, 0.45, np.radians(20))       # flower in view, leaf beside it
BEHIND_LEAF = (0.25, -0.10, 0.45, np.radians(-30))  # flower in the field of view, leaf in front
WITH_TWIN = (0.25, -0.25, 0.40, 0.0)               # flower and its twin both in view


def _project(T):
    p = np.linalg.inv(T) @ np.r_[WORLD.target_position(), 1.0]
    K = WORLD.camera.K
    return K[:2, :2] @ (p[:2] / p[2]) + K[:2, 2], p[2]


def test_render_zbuffer_leaf_hides_the_target():
    H, W = WORLD.camera.image_hw
    T_hidden = look_along_x(*BEHIND_LEAF)
    uv, z = _project(T_hidden)
    assert z > 0 and 0 < uv[0] < W and 0 < uv[1] < H              # inside the field of view ...
    hidden = WORLD.render(T_hidden)
    assert (hidden["label"] == LEAF).sum() > 10 and (hidden["label"] == TARGET).sum() == 0   # ... but occluded
    T_clear = look_along_x(*CLEAR)
    clear = WORLD.render(T_clear)
    u, vis = target_truth(clear["label"], WORLD.camera)
    assert vis and np.linalg.norm(u - _project(T_clear)[0]) < 10.0   # where the pinhole model puts it


# ------------------------------------------------------------- read-outs
def test_readout_finds_the_target_and_needs_the_prior_against_the_twin():
    T = look_along_x(*WITH_TWIN)
    r = WORLD.render(T)
    lab = r["label"]
    assert (lab == TARGET).any() and (lab == DISTRACTOR).any()
    u_true, _ = target_truth(lab, WORLD.camera)
    u_twin = target_truth(np.where(lab == DISTRACTOR, TARGET, 0), WORLD.camera)[0]
    with_prior = readout(r["Z"], R_TARGET, WORLD.camera.image_hw, ReadoutConfig(), prior_u=u_true + [6, -4])
    assert np.linalg.norm(with_prior.u - u_true) < 8 and with_prior.visible and with_prior.identity > 0.9
    no_prior = readout(r["Z"], R_TARGET, WORLD.camera.image_hw, ReadoutConfig(prior_sigma_px=None))
    # without a spatial prior the identical twin pulls the estimate off the target
    assert np.linalg.norm(no_prior.u - u_true) > 0.25 * np.linalg.norm(u_twin - u_true)
    seq = readout_sequence(np.stack([r["Z"]] * 3)[None], R_TARGET, u_true[None], WORLD.camera.image_hw)
    assert seq.u.shape == (1, 3, 2) and seq.visible.all()


# ---------------------------------------------------------------- actions
def _fk(q):
    """Toy batched FK: translation q[:3], rotation vector q[3:]."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, float)
    T = np.zeros(q.shape[:-1] + (4, 4))
    T[..., :3, :3] = Rotation.from_rotvec(q[..., 3:].reshape(-1, 3)).as_matrix().reshape(q.shape[:-1] + (3, 3))
    T[..., :3, 3] = q[..., :3]
    T[..., 3, 3] = 1.0
    return T


def test_action_from_states_matches_the_command_embedding():
    rng = np.random.default_rng(1)
    dt = 0.1
    for _ in range(5):
        x0 = np.r_[rng.normal(size=2), rng.uniform(-3, 3), rng.normal(0, 0.3, 6)]
        u = np.r_[rng.uniform(-0.3, 0.3), rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.5, 6)]
        th_mid = x0[2] + 0.5 * u[1] * dt
        x1 = x0 + np.r_[u[0] * dt * np.cos(th_mid), u[0] * dt * np.sin(th_mid), u[1] * dt, u[2:] * dt]
        a_cmd = action_embedding(x0[3:], u, dt, _fk)
        a_st = action_from_states(x0, x1, _fk)
        # the state version measures the chord of the base arc, the command
        # version its length: equal to second order in the turn per step
        assert np.allclose(a_cmd, a_st, atol=1e-4)
    X = rng.normal(size=(4, 3, 9))
    assert action_from_states(X[:, :-1], X[:, 1:], _fk).shape == (4, 2, 9)


def test_state_predictor_uses_strided_actions():
    class Rec:
        history = 1

        def rollout(self, Z_hist, actions):
            self.actions = actions
            return np.repeat(Z_hist[:, -1:], actions.shape[1], 1)

    rec = Rec()
    sp = StateConditionedPredictor(rec, _fk, stride=2)
    X = np.zeros((3, 9, 9))
    X[:, :, 0] = np.arange(9) * 0.01                          # base moves 1 cm per control step
    Z = np.zeros((1, 12, 16, 4))
    out = sp.predict(Z, X)
    assert out.shape == (3, 4, 12, 16, 4) and np.allclose(rec.actions[..., 0], 0.02)


# ---------------------------------------------------------------- metrics
def _episodes(n=12, frames=8, seed=0):
    """Camera on a unicycle-like 'robot': x = [x, y, θ, q...]; camera 0.45 m up."""
    class Model:
        u_low = np.r_[-0.2, -0.4, np.zeros(6)]
        u_high = -u_low

        def rollout(self, x0, U):
            X = [x0]
            for u in U[0]:
                x = X[-1].copy()
                x[0] += 0.1 * u[0] * np.cos(x[2])
                x[1] += 0.1 * u[0] * np.sin(x[2])
                x[2] += 0.1 * u[1]
                X.append(x)
            return np.array(X)[None]

    def cam(X):
        X = np.asarray(X)
        return np.stack([look_along_x(x[0], x[1], 0.45, x[2]) for x in X.reshape(-1, 9)]).reshape(X.shape[:-1] + (4, 4))

    rng = np.random.default_rng(seed)
    starts = np.zeros((n, 9))
    starts[:, 0] = CLEAR[0]
    starts[:, 1] = CLEAR[1] + rng.uniform(-0.02, 0.02, n)
    starts[:, 2] = CLEAR[3] + rng.uniform(-0.1, 0.1, n)
    return synthetic_episodes(WORLD, Model(), cam, starts, frames, 2, lambda q: _fk(np.zeros(q.shape)), rng), cam


def test_metrics_reward_the_true_future_over_persistence():
    eps, _ = _episodes()
    H, sl = 4, slice(1, 5)
    keep = [e for e in eps if e["visible"][0]]
    assert len(keep) >= 4
    Zt = np.stack([e["Z"][sl] for e in keep])
    args = (Zt, np.stack([e["label"][sl] for e in keep]), np.stack([e["u"][sl] for e in keep]),
            np.stack([e["visible"][sl] for e in keep]), R_TARGET, np.stack([e["u"][0] for e in keep]),
            WORLD.camera.image_hw)
    oracle = score_predictions(Zt, *args)
    persist = PersistencePredictor().rollout(np.stack([e["Z"][:1] for e in keep]), np.zeros((len(keep), H, 9)))
    naive = score_predictions(persist, *args)
    assert oracle.cumulative_target_error(H) < 4.0
    assert oracle.global_latent[-1] == 0.0
    assert naive.cumulative_target_error(H) > oracle.cumulative_target_error(H)


# ---------------------------------------------------------- predictive cost
def test_visibility_cost_prefers_keeping_the_target_in_view():
    _, cam = _episodes(n=1)
    x0 = np.r_[CLEAR[0], CLEAR[1], CLEAR[3], np.zeros(6)]
    oracle = OracleStatePredictor(WORLD, cam, stride=2)
    cost = JepaVisibilityCost(oracle, WORLD.camera.image_hw)
    assert np.all(cost(np.zeros((2, 5, 9)), np.zeros((2, 4, 8))) == 0)       # no context: C2
    r0 = WORLD.render(cam(x0))
    u0, vis0 = target_truth(r0["label"], WORLD.camera)
    assert vis0
    cost.set_context(r0["Z"][None], R_TARGET, u0)
    steps = 9
    keep = np.repeat(x0[None], steps, 0)                                    # hold still
    turn = keep.copy()
    turn[:, 2] = x0[2] - np.linspace(0, 0.9, steps)                         # yaw the target out of view
    J = cost(np.stack([keep, turn]), np.zeros((2, steps - 1, 8)))
    assert J[0] < J[1]
    assert cost.last_terms["J_vis"][1] > cost.last_terms["J_vis"][0]


def test_geometry_anchor_catches_the_identical_twin():
    """Slide from a clear view to one where the leaf hides the flower and its
    twin is in view. Appearance alone reads the twin as "visible, same
    identity"; anchoring on the projected 3-D target reads "not visible"."""
    _, cam = _episodes(n=1)
    x0 = np.r_[CLEAR[0], CLEAR[1], CLEAR[3], np.zeros(6)]
    steps = 9
    stay = np.repeat(x0[None], steps, 0)
    trap = stay.copy()
    trap[:, 1] = np.linspace(CLEAR[1], -0.10, steps)
    trap[:, 2] = np.linspace(CLEAR[3], 0.0, steps)
    end = WORLD.render(cam(trap[-1]))["label"]
    assert (end == TARGET).sum() == 0 and (end == DISTRACTOR).sum() > 0 and (end == LEAF).sum() > 0
    r0 = WORLD.render(cam(x0))
    u0, _ = target_truth(r0["label"], WORLD.camera)
    X, U = np.stack([stay, trap]), np.zeros((2, steps - 1, 8))
    J = {}
    for anchor in (False, True):
        c = JepaVisibilityCost(OracleStatePredictor(WORLD, cam, stride=2), WORLD.camera.image_hw,
                               camera_pose=cam, K=WORLD.camera.K, stride=2)
        c.set_context(r0["Z"][None], R_TARGET, u0, p_world=WORLD.target_position() if anchor else None)
        J[anchor] = c(X, U)
        vis_end = c.terms(X)["visible"][1, -1]
        assert (vis_end < 0.5) if anchor else (vis_end > 0.5)
    assert J[True][1] - J[True][0] > 10 * (J[False][1] - J[False][0])


# ------------------------------------------------------------- learned P*
def _need_torch():
    return pytest.importorskip("torch")


def test_untrained_predictor_is_persistence_and_round_trips(tmp_path):
    _need_torch()
    from scout_piper_jepa.torch_predictor import ACPredictorConfig, TorchACPredictor
    cfg = ACPredictorConfig(grid_hw=(12, 16), feat_dim=16, history=2, d_model=32, layers=1, heads=2)
    model = TorchACPredictor(cfg)
    Zh = np.random.default_rng(0).normal(size=(2, 2, 12, 16, 16)).astype(np.float32)
    A = np.zeros((2, 3, 9), np.float32)
    out = model.rollout(Zh, A)
    assert np.allclose(out, np.repeat(Zh[:, -1:], 3, 1), atol=1e-6)        # zero-initialised head
    model.save(str(tmp_path / "p.pt"))
    again = TorchACPredictor.load(str(tmp_path / "p.pt"))
    assert np.allclose(again.rollout(Zh, A), out)


def _shift_episodes(n, seed, T=8, hw=(6, 8), C=8):
    """A world where the action is a ±1-cell horizontal shift of the feature grid:
    trivially predictable with the action, unpredictable without it."""
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n):
        Z = rng.normal(size=hw + (C,))
        Z /= np.linalg.norm(Z, axis=-1, keepdims=True)
        s = rng.integers(-1, 2, T - 1)
        Zs = [Z]
        for k in range(T - 1):
            Zs.append(np.roll(Zs[-1], s[k], axis=1))
        A = np.zeros((T - 1, 9))
        A[:, 0] = s
        none = np.zeros((T,) + hw, bool)
        eps.append({"Z": np.array(Zs, np.float32), "A": A, "target": none, "plant": none})
    return eps


def test_trained_predictor_uses_the_action():
    """P2 learns the action-dependent shift; P0 (action-free) cannot."""
    _need_torch()
    from scout_piper_jepa.torch_predictor import ACPredictorConfig, train_predictor
    train, test = _shift_episodes(48, 0), _shift_episodes(16, 1)
    Zh = np.stack([e["Z"][t - 1:t + 1] for e in test for t in range(1, 6)])
    A = np.stack([e["A"][t:t + 1] for e in test for t in range(1, 6)])
    Zn = np.stack([e["Z"][t + 1] for e in test for t in range(1, 6)])

    def l1(P):
        return float(np.abs(P - Zn).sum(-1).mean())

    kw = dict(grid_hw=(6, 8), feat_dim=8, history=2, horizon=2, d_model=32, layers=2, heads=2,
              steps=200, batch=16, lr=2e-3)
    p2 = train_predictor(train, ACPredictorConfig.p2(**kw))
    p0 = train_predictor(train, ACPredictorConfig.p0(**kw))
    persistence = l1(Zh[:, -1])
    assert l1(p2.rollout(Zh, A)[:, 0]) < 0.85 * persistence
    assert l1(p2.rollout(Zh, A)[:, 0]) < l1(p0.rollout(Zh, A)[:, 0])
