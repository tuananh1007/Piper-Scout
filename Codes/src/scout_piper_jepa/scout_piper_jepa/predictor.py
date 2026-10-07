"""Stage B: action-conditioned prediction of dense features (research plan §10).

    Ẑ_{t+1:t+H} = P_φ(Z_{t-K+1:t}, a_{t:t+H-1})

``DensePredictor`` is the interface every method implements:

  * ``PersistencePredictor`` — "nothing changes"; the floor every predictor
    must clear (target error grows with motion, never with the model).
  * ``torch_predictor.TorchACPredictor`` — the learned predictor: P0
    (action-free), P2 (action-conditioned, unweighted loss) and P3 (Piper-JEPA,
    target-weighted loss, §11) differ only in configuration.
  * ``OracleStatePredictor`` — synthetic world only: renders the true future
    from the planned states; the upper bound for the predictive cost.

The MPC plans in whole-body states, the predictor consumes action embeddings;
``StateConditionedPredictor`` converts state rollouts into actions with
``action_from_states`` at the predictor's step (``stride`` control steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .action import FK, action_from_states


class DensePredictor(Protocol):
    history: int        # frames of context K

    def rollout(self, Z_hist: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Z_hist (B, K, Hf, Wf, C), actions (B, H, 9) -> (B, H, Hf, Wf, C)."""


@dataclass
class PersistencePredictor:
    history: int = 1

    def rollout(self, Z_hist: np.ndarray, actions: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z_hist)[:, -1]
        return np.repeat(Z[:, None], np.shape(actions)[1], axis=1)


class StatePredictor(Protocol):
    def predict(self, Z_hist: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Z_hist (K, Hf, Wf, C) current context, X (B, H+1, 9) planned states
        (X[:, 0] = now) -> predicted features (B, H // stride, Hf, Wf, C) at
        X[:, stride], X[:, 2·stride], ..."""


def _strided(X: np.ndarray, stride: int) -> np.ndarray:
    idx = np.arange(0, X.shape[1], stride)
    return X[:, idx]


@dataclass
class StateConditionedPredictor:
    predictor: DensePredictor
    fk: FK                       # batched (..., 6) -> (..., 4, 4) EE pose in the base frame
    stride: int = 2              # controller steps per predictor step

    def predict(self, Z_hist: np.ndarray, X: np.ndarray) -> np.ndarray:
        Xs = _strided(np.asarray(X, float), self.stride)
        A = action_from_states(Xs[:, :-1], Xs[:, 1:], self.fk)
        Zh = np.broadcast_to(np.asarray(Z_hist)[None], (len(Xs),) + np.shape(Z_hist))
        return self.predictor.rollout(Zh, A)


@dataclass
class OracleStatePredictor:
    world: object                             # synthetic.SyntheticWorld
    camera_pose: Callable[[np.ndarray], np.ndarray]   # states (..., 9) -> T_world_cam (..., 4, 4)
    stride: int = 2

    def predict(self, Z_hist: np.ndarray, X: np.ndarray) -> np.ndarray:
        Xs = _strided(np.asarray(X, float), self.stride)[:, 1:]
        return self.world.render(self.camera_pose(Xs))["Z"]
