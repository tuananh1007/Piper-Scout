"""Learned action-conditioned dense predictor (research plan §10–11).

One architecture, three methods (PIPER_JEPA_EXPERIMENTS.md E3):

  P0  action-free temporal predictor      use_actions=False
  P2  generic action-conditioned          lambda_target = lambda_plant = 0
  P3  Piper-JEPA target-weighted          lambda_target > lambda_plant > 0

Model: each patch of the last K feature grids becomes a token (linear
projection + learned position); the action embedding Γ (§9, normalised) and
the proprioceptive state are one extra token; a pre-norm transformer encoder
mixes them and a linear head predicts a per-patch residual, so an untrained
model is the persistence predictor (zero-initialised head). Rollouts feed
predictions back autoregressively.

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

    @classmethod
    def p0(cls, **kw) -> "ACPredictorConfig":
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
            layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, cfg.feat_dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, Zh, a):
            """Zh (B, K, M, C), a (B, A) normalised -> Ẑ (B, M, C)."""
            B = Zh.shape[0]
            x = self.inp(Zh.permute(0, 2, 1, 3).reshape(B, M, -1)) + self.pos
            tok = self.act(a)[:, None] if cfg.use_actions else self.act_free.expand(B, 1, -1)
            h = self.enc(torch.cat([tok, x], 1))
            return Zh[:, -1] + self.out(self.norm(h[:, 1:]))

    return Net()


class TorchACPredictor:
    """DensePredictor over numpy arrays; wraps the trained network."""

    def __init__(self, cfg: ACPredictorConfig, net=None):
        torch = _torch()
        self.cfg = cfg
        self.history = cfg.history
        self.net = net if net is not None else _build_net(cfg)
        self.net.eval()
        self._scale = torch.tensor(cfg.action_scale, dtype=torch.float32)
        self.loss_history: List[float] = []

    # ----------------------------------------------------------- inference
    def _rollout_t(self, Zh, A):
        """Zh (B, K, M, C), A (B, H, 9) torch -> (B, H, M, C)."""
        torch = _torch()
        outs = []
        for k in range(A.shape[1]):
            Zn = self.net(Zh, A[:, k] / self._scale)
            outs.append(Zn)
            Zh = torch.cat([Zh[:, 1:], Zn[:, None]], 1)
        return torch.stack(outs, 1)

    def rollout(self, Z_hist: np.ndarray, actions: np.ndarray) -> np.ndarray:
        torch = _torch()
        Z_hist = np.asarray(Z_hist, np.float32)
        B, K, hf, wf, C = Z_hist.shape
        if K < self.history:                     # pad by repeating the oldest frame
            Z_hist = np.concatenate([np.repeat(Z_hist[:, :1], self.history - K, 1), Z_hist], 1)
        Z_hist = Z_hist[:, -self.history:]
        with torch.no_grad():
            Zh = torch.from_numpy(Z_hist.reshape(B, self.history, hf * wf, C).copy())
            A = torch.from_numpy(np.asarray(actions, np.float32).copy())
            out = self._rollout_t(Zh, A)
        return out.numpy().reshape(B, A.shape[1], hf, wf, C)

    # ------------------------------------------------------------ storage
    def save(self, path: str) -> None:
        torch = _torch()
        torch.save({"config": asdict(self.cfg), "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> "TorchACPredictor":
        torch = _torch()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ACPredictorConfig(**{k: tuple(v) if k == "grid_hw" else v for k, v in ck["config"].items()})
        net = _build_net(cfg)
        net.load_state_dict(ck["state_dict"])
        return cls(cfg, net)


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
                    log_every: int = 0) -> TorchACPredictor:
    """episodes: dicts with Z (T, Hf, Wf, C) features, A (T-1, 9) actions
    (A[t] moves frame t to t+1), target (T, Hf, Wf) and plant (T, Hf, Wf) masks."""
    torch = _torch()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if cfg.action_scale == [1.0] * 9:
        cfg.action_scale = action_scale(episodes)
    model = TorchACPredictor(cfg)
    net = model.net
    net.train()
    K, H = cfg.history, cfg.horizon
    hf, wf = cfg.grid_hw
    M = hf * wf
    win = episode_windows(episodes, K, H)
    if not win:
        raise ValueError("episodes too short for history + horizon")
    Zs = [torch.from_numpy(ep["Z"].reshape(len(ep["Z"]), M, -1).astype(np.float32)) for ep in episodes]
    As = [torch.from_numpy(ep["A"].astype(np.float32)) for ep in episodes]
    Ws = [torch.from_numpy((1.0 + cfg.lambda_target * ep["target"].reshape(len(ep["Z"]), M)
                            + cfg.lambda_plant * ep["plant"].reshape(len(ep["Z"]), M)).astype(np.float32))
          for ep in episodes]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    disc = torch.tensor([cfg.gamma ** k for k in range(H)])
    for step in range(cfg.steps):
        pick = rng.integers(len(win), size=cfg.batch)
        Zh = torch.stack([Zs[e][t - K + 1:t + 1] for e, t in (win[i] for i in pick)])          # B,K,M,C
        Zf = torch.stack([Zs[e][t + 1:t + H + 1] for e, t in (win[i] for i in pick)])          # B,H,M,C
        Af = torch.stack([As[e][t:t + H] for e, t in (win[i] for i in pick)])                  # B,H,9
        Wf = torch.stack([Ws[e][t + 1:t + H + 1] for e, t in (win[i] for i in pick)])          # B,H,M
        # teacher forcing: true context for every one-step prediction in the window
        ctx = torch.cat([Zh, Zf[:, :-1]], 1)                                                   # B,K+H-1,M,C
        tf = torch.stack([net(ctx[:, k:k + K], Af[:, k] / model._scale) for k in range(H)], 1)
        err_tf = (tf - Zf).abs().sum(-1)                                                       # B,H,M
        l_tf = (Wf * err_tf).sum(-1).mean() / M
        roll = model._rollout_t(Zh, Af)
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
