"""Torch backend of the whole-body MPC (P3.1): the same model, cost and MPPI
on a GPU (CUDA on the RTX 3060 workstation and the Jetson AGX Orin), or on
the CPU through torch.

Decision (P3.1.1, 2026-10-09): the base enters as two non-holonomic inputs
(v, ω) of our own batched rollout, not as planar virtual joints in a forked
cuRobo kernel. The unicycle model is exact (no holonomic relaxation and no
penalty to tune, P3.1.3), the cost terms stay one implementation (numpy for
tests and the CPU path, this file for the GPU), and nothing outside PyTorch
has to be built for the Orin. cuRobo's collision checking against nvblox is
replaced by the distance-field snapshot that the semantic scene already
publishes (``TorchGridField``).

Pieces, each a line-by-line port of its numpy counterpart:

  TorchModel      dynamics/whole_body.py + dynamics/piper.py: rollout (closed
                  form with cumulative sums), FK, collision spheres,
                  manipulability, wrist extension
  TorchCost       costs/terms.py WholeBodyCost (all terms; ``extra`` / ``secondary`` numpy
                  terms such as Piper-JEPA's are evaluated on the CPU)
  TorchGridField  a dense distance grid on the device, trilinear lookups;
                  from a ``DistanceFieldSnapshot`` (semantic scene / nvblox
                  bridge) or a ``scene_adapter.GridFn``
  TorchMPPI       solvers/mppi.py MPPI: same sampling, elitism, warm start;
                  the gradient refinement uses autograd instead of finite
                  differences

``test_torch_backend.py`` holds the two implementations to each other.
torch is imported lazily so the package imports without it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .costs.terms import WholeBodyCost
from .dynamics.whole_body import WholeBodyModel
from .solvers.mppi import MPPIConfig


def _torch():
    import torch  # noqa: PLC0415 — optional heavy dependency
    return torch


def default_device() -> str:
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- geometry
class TorchGridField:
    """Distance grid on the device with optional validity and leaf-cost grids.

    ``centre_offset`` is 0.5 for voxel grids whose values sit at voxel centres
    (``origin`` = corner of voxel 0, as in ``DistanceFieldSnapshot``) and 0 for
    point-sampled grids (``GridFn``). Out of bounds: ``outside_valid`` False
    reports the point as invalid with distance ≤ 0 (the semantic query's
    "unknown is not free"), True clamps to the border value."""

    def __init__(self, origin, voxel_size: float, distance: np.ndarray, valid: Optional[np.ndarray] = None,
                 leaf: Optional[np.ndarray] = None, centre_offset: float = 0.5, outside_valid: bool = False,
                 device: str = "cpu", dtype=None):
        torch = _torch()
        self.torch, self.device = torch, device
        self.dtype = dtype or torch.float32
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=self.dtype, device=device)  # noqa: E731
        self.origin = f(origin)
        self.vs = float(voxel_size)
        d = np.asarray(distance, np.float64)
        self.dist = f(np.where(np.isfinite(d), d, 1e3))            # +inf -> far (INF_SUBSTITUTE)
        self.valid = None if valid is None else torch.as_tensor(np.asarray(valid, bool), device=device)
        self.leaf = None if leaf is None else f(leaf)
        self.n = torch.as_tensor(self.dist.shape, device=device)
        self.off = centre_offset
        self.outside_valid = outside_valid

    @classmethod
    def from_grid_fn(cls, dist_fn, leaf_fn=None, device: str = "cpu", dtype=None) -> "TorchGridField":
        """From ``scene_adapter.GridFn`` objects (synthetic scenes)."""
        if leaf_fn is not None and (leaf_fn.grid.shape != dist_fn.grid.shape
                                    or not np.allclose(leaf_fn.lo, dist_fn.lo)):
            raise ValueError("leaf and distance grids must share their layout")
        return cls(dist_fn.lo, dist_fn.voxel_size, dist_fn.grid, None,
                   None if leaf_fn is None else leaf_fn.grid, centre_offset=0.0, outside_valid=True,
                   device=device, dtype=dtype)

    @classmethod
    def from_snapshot(cls, snap, now: float, max_age_s: float, leaf_weight: float = 50.0,
                      d_soft: float = 0.02, device: str = "cpu", dtype=None) -> "TorchGridField":
        """From ``scout_piper_scene_repr_py.field.DistanceFieldSnapshot``: hard
        distance (padding applied), validity = known and fresh at ``now``, and
        the leaf cost w·(d_soft − d)² of the soft class."""
        from scout_piper_scene_repr_py.field import AGE_PER_SECOND, AGE_UNKNOWN  # noqa: PLC0415

        age = snap.age_ds
        valid = (age != AGE_UNKNOWN) & (age / AGE_PER_SECOND + (now - snap.stamp) <= max_age_s)
        leaf = None
        if snap.soft_distance is not None:
            sd = np.where(np.isfinite(snap.soft_distance), snap.soft_distance, 1e3)
            leaf = np.where(sd < d_soft, leaf_weight * (sd - d_soft) ** 2, 0.0)
        return cls(snap.origin, snap.voxel_size, snap.hard_distance, valid, leaf,
                   centre_offset=0.5, outside_valid=False, device=device, dtype=dtype)

    def _sample(self, grid, P):
        torch = self.torch
        g = (P - self.origin) / self.vs - self.off
        n = self.n.to(P.dtype)
        gc = torch.minimum(torch.clamp(g, min=0.0), n - 1)
        i0 = torch.minimum(torch.floor(gc), torch.clamp(n - 2, min=0)).long()
        t = gc - i0
        i1 = torch.minimum(i0 + 1, self.n - 1)
        out = 0.0
        for dx in (0, 1):
            wx = t[:, 0] if dx else 1 - t[:, 0]
            ix = i1[:, 0] if dx else i0[:, 0]
            for dy in (0, 1):
                wy = t[:, 1] if dy else 1 - t[:, 1]
                iy = i1[:, 1] if dy else i0[:, 1]
                for dz in (0, 1):
                    wz = t[:, 2] if dz else 1 - t[:, 2]
                    iz = i1[:, 2] if dz else i0[:, 2]
                    out = out + wx * wy * wz * grid[ix, iy, iz]
        return out

    def distance(self, P):
        """P (N, 3) -> hard distance (N,) and valid (N,) bool."""
        torch = self.torch
        d = self._sample(self.dist, P)
        idx = torch.floor((P - self.origin) / self.vs + (0.5 - self.off)).long()
        inb = torch.all((idx >= 0) & (idx < self.n), dim=1)
        if self.valid is not None:
            idc = torch.minimum(torch.clamp(idx, min=0), self.n - 1)
            valid = inb & self.valid[idc[:, 0], idc[:, 1], idc[:, 2]]
        else:
            valid = inb if not self.outside_valid else torch.ones_like(inb)
        d = torch.where(valid, d, torch.clamp(d, max=0.0))            # unknown is not free
        return d, valid

    def leaf_cost(self, P):
        if self.leaf is None:
            return P.new_zeros(len(P))
        return self._sample(self.leaf, P)


# -------------------------------------------------------------------- model
class TorchModel:
    """Batched whole-body model on the device (port of WholeBodyModel)."""

    def __init__(self, model: WholeBodyModel, device: str = "cpu", dtype=None):
        torch = _torch()
        self.torch, self.m, self.device = torch, model, device
        self.dtype = dtype or torch.float32
        f = lambda a: torch.as_tensor(np.asarray(a, float), dtype=self.dtype, device=device)  # noqa: E731
        kin = model.kin
        self.origins = f(kin._origins)
        self.T_base_arm = f(kin.T_base_arm)
        self.T_tcp = f(kin.T_tcp)
        self.lower, self.upper = f(kin.lower), f(kin.upper)
        self.arm_idx = [i for i, _ in model.arm_spheres]
        self.base_p = f([s[:3] for s in model.base_spheres])
        self.radii = f(np.r_[[r for _, r in model.arm_spheres], [s[3] for s in model.base_spheres]])
        self.dt = float(model.dt)
        self.k_v, self.k_w = float(model.scout.k_v), float(model.scout.k_omega)
        self.u_low, self.u_high = f(model.u_low), f(model.u_high)

    def tensor(self, a):
        return self.torch.as_tensor(np.asarray(a, float), dtype=self.dtype, device=self.device)

    def rollout(self, x0, U):
        """x0 (9,), U (B, H, 8) -> X (B, H+1, 9); closed-form unicycle."""
        torch, dt = self.torch, self.dt
        B, H, _ = U.shape
        v, w = self.k_v * U[..., 0], self.k_w * U[..., 1]
        th = x0[2] + dt * torch.cumsum(w, 1)                            # θ_1..θ_H
        th_prev = torch.cat([x0[2].expand(B, 1), th[:, :-1]], 1)       # θ_0..θ_{H-1}
        x = x0[0] + dt * torch.cumsum(v * torch.cos(th_prev), 1)
        y = x0[1] + dt * torch.cumsum(v * torch.sin(th_prev), 1)
        q = torch.minimum(torch.maximum(x0[3:] + dt * torch.cumsum(U[..., 2:], 1), self.lower), self.upper)
        Xn = torch.cat([torch.stack([x, y, th], -1), q], -1)
        return torch.cat([x0.expand(B, 1, 9), Xn], 1)

    def link_frames(self, q):
        """q (..., 6) -> (..., 8, 4, 4) in the Scout base_link frame."""
        torch = self.torch
        T = self.T_base_arm.expand(q.shape[:-1] + (4, 4))
        frames = [T]
        c, s = torch.cos(q), torch.sin(q)
        for i in range(6):
            A = T @ self.origins[i]
            a0, a1 = A[..., :, 0], A[..., :, 1]
            ci, si = c[..., i, None], s[..., i, None]
            T = torch.stack([ci * a0 + si * a1, ci * a1 - si * a0, A[..., :, 2], A[..., :, 3]], -1)
            frames.append(T)
        frames.append(T @ self.T_tcp)
        return torch.stack(frames, -3)

    def planar_T(self, b):
        torch = self.torch
        c, s = torch.cos(b[..., 2]), torch.sin(b[..., 2])
        z, o = torch.zeros_like(c), torch.ones_like(c)
        return torch.stack([torch.stack([c, -s, z, b[..., 0]], -1), torch.stack([s, c, z, b[..., 1]], -1),
                            torch.stack([z, z, o, z], -1), torch.stack([z, z, z, o], -1)], -2)

    def manipulability(self, F):
        """Yoshikawa measure of the translational Jacobian from arm frames F (..., 8, 4, 4)."""
        torch = self.torch
        p = F[..., -1, :3, 3]
        cols = [torch.linalg.cross(F[..., i + 1, :3, 2], p - F[..., i + 1, :3, 3], dim=-1) for i in range(6)]
        Jv = torch.stack(cols, -1)                                       # (..., 3, 6)
        return torch.sqrt(torch.clamp(torch.linalg.det(Jv @ Jv.transpose(-1, -2)), min=0.0) + 1e-12)

    def wrist_extension(self, q):
        torch = self.torch
        t3, R3 = self.origins[2][:3, 3], self.origins[2][:3, :3]
        w = self.origins[3][:3, 3]
        c, s = torch.cos(q[..., 2]), torch.sin(q[..., 2])
        rw = torch.stack([c * w[0] - s * w[1], s * w[0] + c * w[1], torch.full_like(c, float(w[2]))], -1)
        return torch.linalg.norm(t3 + rw @ R3.T, dim=-1)


# --------------------------------------------------------------------- cost
class TorchCost:
    """WholeBodyCost on tensors. Geometry: a TorchGridField on the device, or
    the numpy ``distance_fn`` / ``leaf_fn`` of the cost evaluated on the CPU
    (no gradient through it)."""

    def __init__(self, tm: TorchModel, cost: WholeBodyCost, field: Optional[TorchGridField] = None):
        self.tm, self.cost, self.field = tm, cost, field
        self.w = cost.w
        g = cost.goal
        self.goal_p = tm.tensor(g.p)
        self.goal_R = None if g.R is None else tm.tensor(g.R)
        self.axis = None if g.approach_axis is None else tm.tensor(g.approach_axis)
        self.advance = float(g.advance_m)

    def _geometry(self, C):
        torch, tm = self.tm.torch, self.tm
        P = C.reshape(-1, 3)
        if self.field is not None:
            d, valid = self.field.distance(P)
            leaf = self.field.leaf_cost(P) if self.cost.leaf_fn is not None else None
        else:
            if self.cost.distance_fn is None and self.cost.leaf_fn is None:
                return None, None, None
            Pn = P.detach().cpu().numpy().astype(float)
            d = valid = leaf = None
            if self.cost.distance_fn is not None:
                dn, vn = self.cost.distance_fn(Pn)
                d = tm.tensor(dn)
                valid = torch.as_tensor(np.asarray(vn, bool), device=tm.device)
            if self.cost.leaf_fn is not None:
                leaf = tm.tensor(self.cost.leaf_fn(Pn))
        shp = C.shape[:-1]
        return (None if d is None else d.reshape(shp), None if valid is None else valid.reshape(shp),
                None if leaf is None else leaf.reshape(shp))

    def __call__(self, X, U, u_prev=None):
        torch, tm, w = self.tm.torch, self.tm, self.w
        Xk = X[:, 1:]
        Tb = tm.planar_T(Xk[..., :3])                                     # (B, H, 4, 4)
        F = tm.link_frames(Xk[..., 3:])                                   # (B, H, 8, 4, 4)
        T = Tb @ F[..., -1, :, :]
        e = T[..., :3, 3] - self.goal_p
        dist2 = (e ** 2).sum(-1)
        J = w.goal * dist2.mean(1) + w.goal_terminal * dist2[:, -1]
        if w.goal_terminal_linear:
            J = J + w.goal_terminal_linear * torch.sqrt(dist2[:, -1] + 1e-12)
        if self.goal_R is not None:
            Rr = T[:, -1, :3, :3]
            c = ((Rr * self.goal_R).sum((-1, -2)) - 1) / 2
            J = J + w.orient * torch.arccos(torch.clamp(c, -1 + 1e-7, 1 - 1e-7)) ** 2
        elif self.axis is not None:
            J = J + w.orient * (1 - T[:, -1, :3, 2] @ self.axis)

        geometric = self.field is not None or self.cost.distance_fn is not None or self.cost.leaf_fn is not None
        if geometric:
            Fw = Tb[..., None, :, :] @ F[..., tm.arm_idx, :, :]
            arm_c = Fw[..., :3, 3]                                        # (B, H, Sa, 3)
            base_c = (Tb[..., None, :3, :3] @ tm.base_p[..., None])[..., 0] + Tb[..., None, :3, 3]
            C = torch.cat([arm_c, base_c], -2)                            # (B, H, S, 3)
            d, valid, leaf = self._geometry(C)
            if d is not None and (self.field is not None or self.cost.distance_fn is not None):
                d = d - tm.radii
                viol = torch.clamp(w.d_safe - d, min=0.0)
                J = J + w.collision * (viol ** 2).sum((1, 2))
                J = J + w.unknown * (~valid).sum((1, 2)).to(J.dtype) / valid.shape[2]
            if leaf is not None:
                J = J + w.leaf * leaf.sum((1, 2))

        q = Xk[..., 3:]
        J = J + w.manip * (1.0 / (tm.manipulability(F) + 1e-3)).mean(1)
        margin = 0.05
        lim = torch.clamp(tm.lower + margin - q, min=0) + torch.clamp(q - (tm.upper - margin), min=0)
        J = J + w.joint_limit * (lim ** 2).sum((1, 2))
        if w.reach > 0:
            over = torch.clamp(tm.wrist_extension(q) - w.reach_max_m, min=0)
            J = J + w.reach * (over ** 2).mean(1)
        if w.reach_advanced > 0 and self.advance > 0 and self.axis is not None:
            Fl = F[:, -1]
            ww = Fl[:, 4, :3, 3] - Fl[:, 2, :3, 3] + self.advance * Fl[:, -1, :3, 2]
            ext = torch.linalg.norm(ww, dim=-1)
            J = J + w.reach_advanced * torch.clamp(ext - w.reach_max_advanced_m, min=0) ** 2

        J = J + w.base * (U[..., 0] ** 2 + w.omega * U[..., 1] ** 2).mean(1)
        dU = U[:, 1:] - U[:, :-1]
        J = J + w.smooth * (dU ** 2).sum((1, 2))
        if u_prev is not None:
            J = J + w.smooth * ((U[:, 0] - u_prev) ** 2).sum(-1)
        if self.cost.extra or self.cost.secondary:
            Xn, Un = X.detach().cpu().numpy().astype(float), U.detach().cpu().numpy().astype(float)
            for term in self.cost.extra:
                J = J + tm.tensor(term(Xn, Un))
            if self.cost.secondary:
                e_T = torch.sqrt(dist2[:, -1]).detach().cpu().numpy().astype(float)
                J = J + tm.tensor(self.cost.secondary_cost(Xn, Un, e_T))
        return J


# ------------------------------------------------------------------- solver
class TorchMPPI:
    """Drop-in for ``solvers.mppi.MPPI`` (same ``solve``/``shift``/``reset``,
    ``last_cost``, ``last_X``); ``field`` is the optional device geometry."""

    def __init__(self, model: WholeBodyModel, cfg: MPPIConfig = MPPIConfig(), device: Optional[str] = None,
                 dtype=None, field: Optional[TorchGridField] = None):
        torch = _torch()
        self.torch, self.m, self.cfg = torch, model, cfg
        self.device = device or default_device()
        self.tm = TorchModel(model, self.device, dtype)
        self.field = field
        self.U = torch.zeros(cfg.horizon, 8, dtype=self.tm.dtype, device=self.device)
        sigma = np.r_[cfg.noise_base, [cfg.noise_arm] * 6]
        if cfg.arm_only:
            sigma[:2] = 0.0
        self.sigma = self.tm.tensor(sigma)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(cfg.seed if cfg.seed is not None else int(np.random.SeedSequence().entropy % 2**63))
        self.last_cost = np.inf
        self.last_X: Optional[np.ndarray] = None
        H, k = cfg.horizon, cfg.noise_knots
        self.A = None
        if k >= 2:
            t = np.linspace(0, k - 1, H)
            A = np.zeros((H, k))
            i = np.minimum(np.floor(t).astype(int), k - 2)
            f = t - i
            A[np.arange(H), i] = 1 - f
            A[np.arange(H), i + 1] = f
            self.A = self.tm.tensor(A)

    def _noise(self):
        torch, cfg = self.torch, self.cfg
        if self.A is None:
            z = torch.randn(cfg.samples, cfg.horizon, 8, generator=self.gen, device=self.device, dtype=self.tm.dtype)
            return z * self.sigma
        z = torch.randn(cfg.samples, self.A.shape[1], 8, generator=self.gen, device=self.device,
                        dtype=self.tm.dtype) * self.sigma
        return torch.einsum("hk,skd->shd", self.A, z)

    def reset(self) -> None:
        self.U.zero_()

    def shift(self) -> None:
        self.U[:-1] = self.U[1:].clone()
        self.U[-1] = self.U[-2]

    def solve(self, x0: np.ndarray, cost: WholeBodyCost, u_prev: Optional[np.ndarray] = None) -> np.ndarray:
        torch, cfg, tm = self.torch, self.cfg, self.tm
        x = tm.tensor(x0)
        up = None if u_prev is None else tm.tensor(u_prev)
        # a cost with its own torch twin (visual_servo.VisualServoCost) brings it along
        tc = cost.torch_cost(tm, self.field) if hasattr(cost, "torch_cost") else TorchCost(tm, cost, self.field)
        lo, hi = tm.u_low.clone(), tm.u_high.clone()
        if cfg.arm_only:
            lo[:2] = 0.0
            hi[:2] = 0.0
        clip = lambda u: torch.minimum(torch.maximum(u, lo), hi)  # noqa: E731
        with torch.no_grad():
            for _ in range(cfg.iterations):
                Us = clip(self.U[None] + self._noise())
                if cfg.include_zero:
                    Us[0] = 0.0
                    Us[1] = clip(self.U)
                J = tc(tm.rollout(x, Us), Us, up)
                Jn = (J - J.min()) / torch.clamp(J.std(correction=0), min=1e-9)
                wts = torch.softmax(-Jn / cfg.temperature, 0)
                self.U = clip((wts[:, None, None] * Us).sum(0))
                if cfg.keep_best:
                    b = int(torch.argmin(J))
                    j_avg = tc(tm.rollout(x, self.U[None]), self.U[None], up)[0]
                    if J[b] < j_avg:
                        self.U = Us[b].clone()
        if cfg.refine_iters > 0 and self.A is not None:
            self._refine(x, tc, up, clip)
        with torch.no_grad():
            X = tm.rollout(x, self.U[None])
            self.last_cost = float(tc(X, self.U[None], up)[0])
            self.last_X = X[0].cpu().numpy().astype(float)
        return self.U[0].cpu().numpy().astype(float)

    def _refine(self, x, tc, up, clip) -> None:
        """Gradient steps on knot offsets Z (knots, 8) with autograd, batched line search."""
        torch, cfg, tm, A = self.torch, self.cfg, self.tm, self.A
        k = A.shape[1]
        mask = torch.ones(8, dtype=tm.dtype, device=self.device)
        if cfg.arm_only:
            mask[:2] = 0.0
        steps = tm.tensor(np.geomspace(1e-3, 1.0, 10) * cfg.refine_max_step)
        with torch.no_grad():
            j0 = float(tc(tm.rollout(x, self.U[None]), self.U[None], up)[0])
        for _ in range(cfg.refine_iters):
            Z = torch.zeros(k, 8, dtype=tm.dtype, device=self.device, requires_grad=True)
            Uz = clip(self.U + A @ Z)
            j = tc(tm.rollout(x, Uz[None]), Uz[None], up)[0]
            (g,) = torch.autograd.grad(j, Z)
            g = g * mask
            gmax = g.abs().max()
            if not torch.isfinite(gmax) or float(gmax) < 1e-12:
                return
            with torch.no_grad():
                cand = -steps[:, None, None] * (g / gmax)[None]           # (10, k, 8)
                Uc = clip(self.U[None] + torch.einsum("hk,bkd->bhd", A, cand))
                Jc = tc(tm.rollout(x, Uc), Uc, up)
                b = int(torch.argmin(Jc))
                if not float(Jc[b]) < j0:
                    return
                self.U = Uc[b].clone()
                j0 = float(Jc[b])
