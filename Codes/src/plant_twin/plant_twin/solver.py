"""Small damped Gauss-Newton (Levenberg-Marquardt) solver on the normal equations.

The twin's problems are tall and thin (thousands of residuals, ~20 unknowns),
so forming JᵀJ and solving a P×P system per iteration is far cheaper than
scipy's TRF, which does an SVD of the full Jacobian each step. The
``evaluate`` callable must return ``(residual, jacobian)`` at a point — see
``LeafModel.evaluate`` / ``StemModel.evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

Evaluate = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class SolveResult:
    x: np.ndarray
    cost: float           # 0.5 * ||r||²
    n_eval: int           # evaluate() calls
    n_iter: int           # accepted steps
    status: str           # 'converged' | 'max_iter' | 'stalled'


def levenberg_marquardt(
    evaluate: Evaluate,
    x0: np.ndarray,
    max_iter: int = 10,
    lam: float = 1e-3,
    lam_up: float = 10.0,
    lam_down: float = 0.2,
    lam_max: float = 1e6,
    ftol: float = 1e-3,
    xtol: float = 1e-9,
) -> SolveResult:
    """Marquardt-scaled LM: (JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr, accept when the
    cost drops, otherwise raise λ and retry from the same point.

    ``ftol`` is the relative cost drop below which we stop. The default 1e-3
    is a *tracking* tolerance: with a warm start most frames converge in 3–5
    steps and the remaining improvement is below sensor noise. Use 1e-6 for
    a from-scratch fit.

    Each trial step evaluates residual *and* Jacobian in one call (one
    KD-tree pass), so a rejected step costs one evaluation like an accepted
    one. Rejections are rare with a warm start.
    """
    x = np.asarray(x0, dtype=float).copy()
    r, J = evaluate(x)
    cost = 0.5 * float(r @ r)
    n_eval, n_iter = 1, 0
    status = "max_iter"

    for _ in range(max_iter):
        JtJ = J.T @ J
        g = J.T @ r
        diag = np.diag(JtJ).copy()
        diag[diag < 1e-12] = 1e-12
        accepted = False
        while lam <= lam_max:
            A = JtJ + lam * np.diag(diag)
            try:
                dx = cho_solve(cho_factor(A), -g)
            except np.linalg.LinAlgError:
                lam *= lam_up
                continue
            x_new = x + dx
            r_new, J_new = evaluate(x_new)
            n_eval += 1
            cost_new = 0.5 * float(r_new @ r_new)
            if cost_new < cost:
                rel_drop = (cost - cost_new) / max(cost, 1e-300)
                x, r, J, cost = x_new, r_new, J_new, cost_new
                lam = max(lam * lam_down, 1e-12)
                n_iter += 1
                accepted = True
                if rel_drop < ftol or np.linalg.norm(dx) < xtol:
                    status = "converged"
                break
            lam *= lam_up
        if not accepted:
            status = "stalled"
            break
        if status == "converged":
            break

    return SolveResult(x, cost, n_eval, n_iter, status)
