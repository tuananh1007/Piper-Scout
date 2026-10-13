"""Torch port of ``LeafModel.evaluate`` (residual + analytic Jacobian), for the GPU.

The numpy evaluation costs ~3.7 ms per call on a laptop CPU, mostly the
(V, 3, P) per-vertex Jacobian and its contractions, plus two KD-tree queries;
an LM frame calls it 3-5 times. Here every step is a dense tensor operation
and the nearest neighbours come from ``torch.cdist`` (V and N are a few
hundred to a few thousand, so the distance matrix is small), so the whole
evaluation runs on the device; the LM solver keeps working on the returned
numpy arrays (a P × P system).

Same blocks, order and weights as ``LeafModel.evaluate`` (see there): with
float64 on the CPU the two agree to rounding (test_leaf_torch.py). float32 on
a GPU is the intended use on the Orin; the solver only needs the normal
equations to a few digits.

    from plant_twin.leaf_torch import TorchLeafEvaluator
    ev = TorchLeafEvaluator(leaf, device="cuda", dtype=torch.float32)
    r, J = ev.evaluate(params, observed, prev_params, contact_point)
    PlantTwinFitter(leaf, stem, leaf_backend="torch", device="cuda")   # the same, inside the fitter
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def _skew(v: torch.Tensor) -> torch.Tensor:
    """[v]_x for v (..., 3) -> (..., 3, 3)."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rotation_and_right_jacobian(rv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """R(rv) and the SO(3) right Jacobian J_r(rv), as geometry.py."""
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device)
    theta = torch.linalg.norm(rv)
    K = _skew(rv)
    if float(theta) < 1e-12:
        R = eye.clone()
    else:
        k = _skew(rv / theta)
        R = eye + torch.sin(theta) * k + (1 - torch.cos(theta)) * (k @ k)
    if float(theta) < 1e-6:
        Jr = eye - 0.5 * K
    else:
        Jr = eye - (1 - torch.cos(theta)) / theta ** 2 * K + (theta - torch.sin(theta)) / theta ** 3 * (K @ K)
    return R, Jr


class TorchLeafEvaluator:
    """``evaluate`` of one ``LeafModel`` on a torch device (mesh and basis copied once)."""

    def __init__(self, model, device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.m, self.dev, self.dt = model, torch.device(device), dtype
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.dev)  # noqa: E731
        self.rest_xy = T(model.rest_xy)
        self.basis = T(model._basis)
        self.faces = torch.as_tensor(model.faces, dtype=torch.long, device=self.dev)
        self.edges = torch.as_tensor(model.edges, dtype=torch.long, device=self.dev)
        self.rest_edge_len = T(model.rest_edge_len)
        self._obs_key, self._obs = None, None

    def _t(self, a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, dtype=float), dtype=self.dt, device=self.dev)

    def _observed(self, observed) -> torch.Tensor:
        if self._obs_key is not observed:              # fixed within a frame: copy once
            self._obs_key = observed
            self._obs = self._t(np.asarray(observed, dtype=float).reshape(-1, 3))
        return self._obs

    def vertices_and_jacobian(self, p: torch.Tensor, want_jac: bool = True):
        m = self.m
        rv, t, bend = p[:3], p[3:6], p[6:]
        local = torch.cat([self.rest_xy, (self.basis @ bend)[:, None]], 1)
        R, Jr = rotation_and_right_jacobian(rv)
        v = local @ R.T + t
        if not want_jac:
            return v, None
        V, P = len(local), m.n_params
        D = torch.zeros(V, 3, P, dtype=self.dt, device=self.dev)
        D[:, :, 0:3] = -torch.einsum("ij,njk,kl->nil", R, _skew(local), Jr)
        D[:, :, 3:6] = torch.eye(3, dtype=self.dt, device=self.dev)
        D[:, :, 6:] = R[:, 2][None, :, None] * self.basis[:, None, :]
        return v, D

    def normals(self, v: torch.Tensor) -> torch.Tensor:
        f = self.faces
        fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1)
        vn = torch.zeros_like(v)
        for k in range(3):
            vn.index_add_(0, f[:, k], fn)
        return vn / torch.clamp(torch.linalg.norm(vn, dim=1, keepdim=True), min=1e-12)

    def evaluate(self, params, observed, prev_params=None, contact_point=None,
                 want_jac: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        w, m = self.m.weights, self.m
        p = self._t(params)
        v, D = self.vertices_and_jacobian(p, want_jac)
        P = m.n_params
        r_blocks, J_blocks = [], []

        obs = self._observed(observed)
        if len(obs):
            nrm = self.normals(v)
            dist = torch.cdist(obs, v)                  # (N, V)
            nn = dist.argmin(1)
            mn = dist.argmin(0)
            diff = obs - v[nn]
            r_blocks += [w.data * (diff * nrm[nn]).sum(1), w.p2p * w.data * diff.reshape(-1),
                         w.reverse * w.data * (v - obs[mn]).reshape(-1)]
            if want_jac:
                Dnn = D[nn]
                J_blocks += [-w.data * torch.einsum("ij,ijk->ik", nrm[nn], Dnn),
                             -w.p2p * w.data * Dnn.reshape(-1, P), w.reverse * w.data * D.reshape(-1, P)]

        a, b = self.edges[:, 0], self.edges[:, 1]
        d = v[a] - v[b]
        el = torch.linalg.norm(d, dim=1)
        r_blocks.append(w.stretch * (el - self.rest_edge_len))
        if want_jac:
            u = d / torch.clamp(el, min=1e-12)[:, None]
            J_blocks.append(w.stretch * torch.einsum("ij,ijk->ik", u, D[a] - D[b]))

        bend = p[6:]
        r_blocks.append(w.prior * bend)
        if want_jac:
            Jp = torch.zeros(m.n_bend, P, dtype=self.dt, device=self.dev)
            Jp[:, 6:] = w.prior * torch.eye(m.n_bend, dtype=self.dt, device=self.dev)
            J_blocks.append(Jp)

        if prev_params is not None:
            r_blocks.append(w.temporal * (p - self._t(prev_params)))
            if want_jac:
                J_blocks.append(w.temporal * torch.eye(P, dtype=self.dt, device=self.dev))

        if contact_point is not None:
            cp = self._t(contact_point).reshape(3)
            k = torch.linalg.norm(v - cp, dim=1).argmin()
            r_blocks.append(w.contact * (cp - v[k]))
            if want_jac:
                J_blocks.append(-w.contact * D[k])

        r = torch.cat(r_blocks).detach().cpu().numpy().astype(float)
        J = torch.cat(J_blocks).detach().cpu().numpy().astype(float) if want_jac else None
        return r, J
