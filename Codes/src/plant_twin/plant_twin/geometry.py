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


def rotation_jacobian(rv: np.ndarray, points: np.ndarray) -> np.ndarray:
    """d(R(rv) p)/d rv for each row p of ``points`` (N, 3) -> (N, 3, 3).

    Uses the SO(3) right Jacobian: d(R p)/d rv = -R [p]_x J_r(rv).
    """
    rv = np.asarray(rv, dtype=float)
    R = rotvec_to_matrix(rv)
    theta = float(np.linalg.norm(rv))
    K = np.array([[0.0, -rv[2], rv[1]], [rv[2], 0.0, -rv[0]], [-rv[1], rv[0], 0.0]])
    if theta < 1e-6:
        Jr = np.eye(3) - 0.5 * K
    else:
        Jr = (np.eye(3) - (1 - np.cos(theta)) / theta**2 * K
              + (theta - np.sin(theta)) / theta**3 * (K @ K))
    P = np.asarray(points, dtype=float)
    px = np.zeros((len(P), 3, 3))
    px[:, 0, 1], px[:, 0, 2] = -P[:, 2], P[:, 1]
    px[:, 1, 0], px[:, 1, 2] = P[:, 2], -P[:, 0]
    px[:, 2, 0], px[:, 2, 1] = -P[:, 1], P[:, 0]
    return -np.einsum("ij,njk,kl->nil", R, px, Jr)


def catmull_rom_basis(n_ctrl: int, samples_per_seg: int = 8) -> np.ndarray:
    """Basis matrix M (S, K) so that curve = M @ ctrl for a uniform
    Catmull-Rom spline with end control points duplicated."""
    if n_ctrl < 2:
        return np.eye(n_ctrl)
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)
    t2, t3 = t**2, t**3
    w = 0.5 * np.stack([-t + 2 * t2 - t3, 2 - 5 * t2 + 3 * t3,
                        t + 4 * t2 - 3 * t3, -t2 + t3], axis=1)   # (s, 4)
    rows = []
    for i in range(n_ctrl - 1):
        idx = np.clip([i - 1, i, i + 1, i + 2], 0, n_ctrl - 1)
        seg = np.zeros((samples_per_seg, n_ctrl))
        for k in range(4):
            seg[:, idx[k]] += w[:, k]
        rows.append(seg)
    end = np.zeros((1, n_ctrl))
    end[0, -1] = 1.0
    rows.append(end)
    return np.vstack(rows)


def catmull_rom(ctrl: np.ndarray, samples_per_seg: int = 8) -> np.ndarray:
    """Uniform Catmull-Rom spline through control points (K, 3) -> (S, 3)."""
    ctrl = np.asarray(ctrl, dtype=float)
    return catmull_rom_basis(len(ctrl), samples_per_seg) @ ctrl


def polyline_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
