"""Learned action-conditioned dense predictor (research plan §10–11).

One architecture, three methods (PIPER_JEPA_EXPERIMENTS.md E3):

  P0  action-free temporal predictor      use_actions=False
  P2  generic action-conditioned          lambda_target = lambda_plant = 0
  P3  Piper-JEPA target-weighted          lambda_target > lambda_plant > 0

Model: each patch of the last K feature grids becomes a token (linear
projection + learned position); the action embedding Γ (§9, normalised) plus,
with ``use_state``, the proprioceptive state s (the six joint angles at the
start of the step, normalised; P_φ(Z, a, s) of §10) form one extra token; a
pre-norm transformer encoder mixes them and a linear head predicts a
per-patch residual, so an untrained model is the persistence predictor
(zero-initialised head). Rollouts feed predictions back autoregressively.
P0 uses neither actions nor state (the planned states would carry the motion).

Feature projection (``input_dim`` > 0): encoder features of ``input_dim``
channels (1024 for V-JEPA 2 ViT-L) are projected onto the top ``feat_dim``
principal directions of the training features (uncentred, so cosine
similarities inside the subspace are kept) and re-normalised before the
predictor; the projection is stored in the checkpoint and applied in
``rollout`` and ``descriptor``. Without it a control step of 256 samples ×
10 steps × 256 cells × 1024 channels is 2.7 GB of predicted features.

Loss (§11): weights w(p) = 1 + λ_T M_target(p) + λ_P M_plant(p) on the
*future* frame; L_TF = teacher-forced one-step L1 at every window position,
L_roll = Σ_k γ^{k-1} L1 of the autoregressive rollout, L = L_TF + λ_R L_roll.

torch is imported lazily (as in encoder.py) so the rest of the package and its
tests run without it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _torch():
    import torch  # noqa: PLC0415 — optional heavy dependency
    return torch


@dataclass
class ACPredictorConfig:
    grid_hw: Tuple[int, int] = (12, 16)
    feat_dim: int = 16
    action_dim: int = 9
    history: int = 2
    d_model: int = 96
    layers: int = 3
    heads: int = 4
    use_actions: bool = True              # False = P0 (action-free)
    lambda_target: float = 0.0            # λ_T  (P3: > λ_P > 0)
    lambda_plant: float = 0.0             # λ_P
    lambda_rollout: float = 1.0           # λ_R
    gamma: float = 0.9
    horizon: int = 4                      # training rollout length
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 32
    steps: int = 1500
    seed: int = 0
    action_scale: List[float] = field(default_factory=lambda: [1.0] * 9)
    device: str = "cpu"                   # "cuda" for GPU training / inference
    step_s: float = 0.0                   # time between the training frames (0 = unknown); the
                                          # MPC must step the predictor at the same interval
    use_state: bool = False               # condition on the joint angles (P_φ(Z, a, s), §10)
    state_dim: int = 6
    state_mean: List[float] = field(default_factory=lambda: [0.0] * 6)
    state_std: List[float] = field(default_factory=lambda: [1.0] * 6)
    input_dim: int = 0                    # encoder channels before the projection (0 = none)

    @classmethod
    def p0(cls, **kw) -> "ACPredictorConfig":
        kw["use_state"] = False
        return cls(use_actions=False, **kw)

    @classmethod
    def p2(cls, **kw) -> "ACPredictorConfig":
        return cls(lambda_target=0.0, lambda_plant=0.0, **kw)

    @classmethod
    def p3(cls, lambda_target: float = 20.0, lambda_plant: float = 2.0, **kw) -> "ACPredictorConfig":
        return cls(lambda_target=lambda_target, lambda_plant=lambda_plant, **kw)


def _build_net(cfg: ACPredictorConfig):
    torch = _torch()
    nn = torch.nn
    M = cfg.grid_hw[0] * cfg.grid_hw[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.inp = nn.Linear(cfg.history * cfg.feat_dim, d)
            self.pos = nn.Parameter(torch.zeros(1, M, d))
            nn.init.normal_(self.pos, std=0.02)
            self.act = nn.Sequential(nn.Linear(cfg.action_dim, d), nn.GELU(), nn.Linear(d, d))
            self.act_free = nn.Parameter(torch.zeros(1, 1, d))
            if cfg.use_state:
                self.state = nn.Sequential(nn.Linear(cfg.state_dim, d), nn.GELU(), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, cfg.feat_dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, Zh, a, s=None):
            """Zh (B, K, M, C), a (B, A), s (B, state_dim) normalised -> Ẑ (B, M, C)."""
            B = Zh.shape[0]
            x = self.inp(Zh.permute(0, 2, 1, 3).reshape(B, M, -1)) + self.pos
            tok = self.act(a)[:, None] if cfg.use_actions else self.act_free.expand(B, 1, -1)
            if cfg.use_state:
                tok = tok + self.state(s)[:, None]
            h = self.enc(torch.cat([tok, x], 1))
            return Zh[:, -1] + self.out(self.norm(h[:, 1:]))

    return Net()


class TorchACPredictor:
    """DensePredictor over numpy arrays; wraps the trained network."""

    def __init__(self, cfg: ACPredictorConfig, net=None, proj: Optional[np.ndarray] = None):
        torch = _torch()
        self.cfg = cfg
        self.history = cfg.history
        self.device = cfg.device
        self.net = (net if net is not None else _build_net(cfg)).to(self.device)
        self.net.eval()
        self._scale = torch.tensor(cfg.action_scale, dtype=torch.float32, device=self.device)
        self._s_mean = torch.tensor(cfg.state_mean, dtype=torch.float32, device=self.device)
        self._s_std = torch.tensor(cfg.state_std, dtype=torch.float32, device=self.device)
        self.proj = None if proj is None else np.asarray(proj, np.float32)          # (input_dim, feat_dim)
        self._proj_t = None if proj is None else torch.from_numpy(self.proj).to(self.device)
        self.loss_history: List[float] = []

    # ----------------------------------------------------------- inference
    def _state_norm(self, S):
        return None if S is None else (S - self._s_mean) / self._s_std

    def _rollout_t(self, Zh, A, S=None):
        """Zh (B, K, M, C), A (B, H, 9), S (B, H, state_dim) or None, torch -> (B, H, M, C)."""
        torch = _torch()
        Sn = self._state_norm(S)
        outs = []
        for k in range(A.shape[1]):
            Zn = self.net(Zh, A[:, k] / self._scale, None if Sn is None else Sn[:, k])
            outs.append(Zn)
            Zh = torch.cat([Zh[:, 1:], Zn[:, None]], 1)
        return torch.stack(outs, 1)

    def descriptor(self, r: np.ndarray) -> np.ndarray:
        """The target descriptor in the predictor's feature space."""
        return np.asarray(r) if self.proj is None else project_features(r, self.proj)

    def rollout(self, Z_hist: np.ndarray, actions: np.ndarray,
                states: Optional[np.ndarray] = None, projected: bool = False) -> np.ndarray:
        """Z_hist (B, K, Hf, Wf, C) encoder features (C = input_dim with a
        projection; ``projected``: already projected, C = feat_dim), actions
        (B, H, 9), states (B, H, ≥6) joint angles at the start of each step
        (needed with use_state) -> (B, H, Hf, Wf, feat_dim). A context shared
        by all samples (a broadcast view, as the predictive cost passes it) is
        copied to the device once."""
        torch = _torch()
        Z_in = np.asarray(Z_hist)
        B, K, hf, wf, C = Z_in.shape
        project = self.proj is not None and not projected
        want = self.cfg.input_dim if project else self.cfg.feat_dim
        if C != want:
            raise ValueError(f"features have {C} channels, the predictor expects {want}")
        shared = B > 1 and Z_in.strides[0] == 0
        src = np.asarray(Z_in[:1] if shared else Z_in, np.float32)
        if K < self.history:                     # pad by repeating the oldest frame
            src = np.concatenate([np.repeat(src[:, :1], self.history - K, 1), src], 1)
        src = src[:, -self.history:]
        if self.cfg.use_state and states is None:
            raise ValueError("this predictor conditions on the joint state: pass states (B, H, 6)")
        with torch.no_grad():
            Zh = torch.from_numpy(src.reshape(len(src), self.history, hf * wf, C).copy())   # one context if shared
            Zh = Zh.to(self.device)
            if project:
                Zh = torch.nn.functional.normalize(Zh @ self._proj_t, dim=-1)
            if shared:
                Zh = Zh.expand(B, -1, -1, -1)
            A = torch.from_numpy(np.asarray(actions, np.float32).copy()).to(self.device)
            S = None
            if self.cfg.use_state:
                S = torch.from_numpy(np.ascontiguousarray(
                    np.asarray(states, np.float32)[..., :self.cfg.state_dim])).to(self.device)
            out = self._rollout_t(Zh, A, S)
        return out.cpu().numpy().reshape(B, A.shape[1], hf, wf, self.cfg.feat_dim)

    # ------------------------------------------------------------ storage
    def save(self, path: str) -> None:
        torch = _torch()
        torch.save({"config": asdict(self.cfg), "state_dict": self.net.state_dict(), "proj": self.proj}, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TorchACPredictor":
        """Load a checkpoint; ``device`` overrides the one it was trained on
        (e.g. a GPU-trained model on a CPU-only machine)."""
        torch = _torch()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        known = set(ACPredictorConfig.__dataclass_fields__)
        cfg = ACPredictorConfig(**{k: tuple(v) if k == "grid_hw" else v for k, v in ck["config"].items()
                                   if k in known})
        if device is not None:
            cfg.device = device
        elif cfg.device.startswith("cuda") and not torch.cuda.is_available():
            cfg.device = "cpu"
        net = _build_net(cfg)
        net.load_state_dict(ck["state_dict"])
        return cls(cfg, net, proj=ck.get("proj"))


def fit_projection(Z_list: Sequence[np.ndarray], dim: int, max_rows: int = 200_000, seed: int = 0):
    """Top ``dim`` uncentred principal directions of the feature vectors in
    ``Z_list`` (arrays (..., C)): (W (C, dim) float32, fraction of the energy kept)."""
    X = np.concatenate([np.asarray(Z, np.float32).reshape(-1, np.shape(Z)[-1]) for Z in Z_list])
    if len(X) > max_rows:
        X = X[np.random.default_rng(seed).choice(len(X), max_rows, replace=False)]
    G = X.T.astype(np.float64) @ X
    ev, V = np.linalg.eigh(G)                                   # ascending
    W = V[:, ::-1][:, :dim]
    return W.astype(np.float32), float(ev[::-1][:dim].sum() / max(ev.sum(), 1e-12))


def project_features(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(..., C) -> (..., dim), re-normalised (cosine similarity stays a dot product)."""
    P = np.asarray(Z, np.float32) @ W
    return P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-8)


def predictor_stride(step_s: float, dt: float, default: int = 2):
    """Controller steps per predictor step for a model trained at ``step_s``:
    (stride, warning or None). Unknown ``step_s`` (0, checkpoints from before it
    was recorded) gives ``default``: 2, the step of the synthetic training."""
    if step_s <= 0:
        return default, None
    stride = max(1, int(round(step_s / dt)))
    if abs(stride * dt - step_s) > 0.25 * step_s:
        return stride, (f"the predictor was trained on frames {step_s:.3f} s apart, the controller steps "
                        f"{dt:.3f} s: no stride matches (export the episodes with a frame interval that is "
                        f"a multiple of {dt:.3f} s)")
    return stride, None


# ---------------------------------------------------------------- training
def episode_windows(episodes: Sequence[Dict[str, np.ndarray]], history: int, horizon: int):
    """(episode, t) pairs with K frames of context and H future frames."""
    out = []
    for e, ep in enumerate(episodes):
        T = len(ep["Z"])
        out += [(e, t) for t in range(history - 1, T - horizon)]
    return out


def action_scale(episodes: Sequence[Dict[str, np.ndarray]]) -> List[float]:
    A = np.concatenate([ep["A"] for ep in episodes])
    s = A.std(0)
    return [float(v) if v > 1e-6 else 1.0 for v in s]


def train_predictor(episodes: Sequence[Dict[str, np.ndarray]], cfg: ACPredictorConfig,
                    log_every: int = 0, proj: Optional[np.ndarray] = None) -> TorchACPredictor:
    """episodes: dicts with Z (T, Hf, Wf, feat_dim) features (already projected
    when ``proj`` is given), A (T-1, 9) actions (A[t] moves frame t to t+1),
    target (T, Hf, Wf) and plant (T, Hf, Wf) masks and, with use_state, the
    joint angles (``S`` (T, 6), or the whole-body ``states`` (T, 9)).
    ``proj`` (input_dim, feat_dim) is stored with the model for inference."""
    from .predictor import joint_states  # noqa: PLC0415
    torch = _torch()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if cfg.action_scale == [1.0] * 9:
        cfg.action_scale = action_scale(episodes)
    if cfg.use_state:
        Sall = np.concatenate([joint_states(ep)[:, :cfg.state_dim] for ep in episodes])
        cfg.state_mean = [float(v) for v in Sall.mean(0)]
        cfg.state_std = [float(v) if v > 1e-6 else 1.0 for v in Sall.std(0)]
    model = TorchACPredictor(cfg, proj=proj)
    net = model.net
    net.train()
    K, H = cfg.history, cfg.horizon
    hf, wf = cfg.grid_hw
    M = hf * wf
    win = episode_windows(episodes, K, H)
    if not win:
        raise ValueError("episodes too short for history + horizon")
    dev = model.device
    Zs = [torch.from_numpy(ep["Z"].reshape(len(ep["Z"]), M, -1).astype(np.float32)).to(dev) for ep in episodes]
    As = [torch.from_numpy(ep["A"].astype(np.float32)).to(dev) for ep in episodes]
    Ws = [torch.from_numpy((1.0 + cfg.lambda_target * ep["target"].reshape(len(ep["Z"]), M)
                            + cfg.lambda_plant * ep["plant"].reshape(len(ep["Z"]), M)).astype(np.float32)).to(dev)
          for ep in episodes]
    Ss = [torch.from_numpy(np.ascontiguousarray(joint_states(ep)[:, :cfg.state_dim], np.float32)).to(dev)
          if cfg.use_state else None for ep in episodes]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    disc = torch.tensor([cfg.gamma ** k for k in range(H)], device=dev)
    for step in range(cfg.steps):
        pick = rng.integers(len(win), size=cfg.batch)
        Zh = torch.stack([Zs[e][t - K + 1:t + 1] for e, t in (win[i] for i in pick)])          # B,K,M,C
        Zf = torch.stack([Zs[e][t + 1:t + H + 1] for e, t in (win[i] for i in pick)])          # B,H,M,C
        Af = torch.stack([As[e][t:t + H] for e, t in (win[i] for i in pick)])                  # B,H,9
        Wf = torch.stack([Ws[e][t + 1:t + H + 1] for e, t in (win[i] for i in pick)])          # B,H,M
        Sf = torch.stack([Ss[e][t:t + H] for e, t in (win[i] for i in pick)]) if cfg.use_state else None
        Sn = model._state_norm(Sf)                                                             # B,H,6
        # teacher forcing: true context for every one-step prediction in the window
        ctx = torch.cat([Zh, Zf[:, :-1]], 1)                                                   # B,K+H-1,M,C
        tf = torch.stack([net(ctx[:, k:k + K], Af[:, k] / model._scale, None if Sn is None else Sn[:, k])
                          for k in range(H)], 1)
        err_tf = (tf - Zf).abs().sum(-1)                                                       # B,H,M
        l_tf = (Wf * err_tf).sum(-1).mean() / M
        roll = model._rollout_t(Zh, Af, Sf)
        err_roll = (roll - Zf).abs().sum(-1)
        l_roll = ((Wf * err_roll).sum(-1) * disc).sum(-1).mean() / M
        loss = l_tf + cfg.lambda_rollout * l_roll
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        model.loss_history.append(loss.item())
        if log_every and step % log_every == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 4),
                              "l_tf": round(float(l_tf), 4), "l_roll": round(float(l_roll), 4)}))
    net.eval()
    return model
