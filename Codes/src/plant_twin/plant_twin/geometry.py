"""Small geometry helpers shared by the leaf and stem models."""

from __future__ import annotations

import numpy as np


def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle vector (3,) -> rotation matrix (3, 3)."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def apply_rigid(points: np.ndarray, rotvec: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """R(rotvec) @ p + t for each row of ``points`` (N, 3)."""
    return points @ rotvec_to_matrix(rotvec).T + np.asarray(trans, dtype=float)


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd rule for a 2D point against a closed polygon (M, 2)."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def bump_basis(uv: np.ndarray, grid: int) -> np.ndarray:
    """Smooth radial-basis matrix for a ``grid``×``grid`` lattice on [0,1]².

    Rows = query points (N, 2); columns = lattice nodes. Used to turn a small
    set of per-node offsets into a C² height field (the leaf bending model).
    """
    uv = np.asarray(uv, dtype=float)
    g = np.linspace(0.0, 1.0, grid)
    cu, cv = np.meshgrid(g, g, indexing="ij")
    centres = np.stack([cu.ravel(), cv.ravel()], axis=1)
    sigma = 1.0 / max(grid - 1, 1)
    d2 = ((uv[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / sigma**2)


def catmull_rom(ctrl: np.ndarray, samples_per_seg: int = 8) -> np.ndarray:
    """Uniform Catmull-Rom spline through control points (K, 3) -> (S, 3)."""
    ctrl = np.asarray(ctrl, dtype=float)
    if len(ctrl) < 2:
        return ctrl.copy()
    padded = np.vstack([ctrl[0], ctrl, ctrl[-1]])
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        t2, t3 = t**2, t**3
        seg = 0.5 * (
            (2 * p1)[None]
            + np.outer(t, -p0 + p2)
            + np.outer(t2, 2 * p0 - 5 * p1 + 4 * p2 - p3)
            + np.outer(t3, -p0 + 3 * p1 - 3 * p2 + p3)
        )
        out.append(seg)
    out.append(ctrl[-1][None])
    return np.vstack(out)


def polyline_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
