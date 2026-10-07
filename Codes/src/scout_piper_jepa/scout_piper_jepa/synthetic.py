"""Synthetic dense-feature world for Stage B / C development (no camera, no GPU).

A scene is a set of *surfels* (3-D points with a radius, a feature vector and
a label). ``render(T_world_cam)`` produces what an encoder would: a feature
grid (Hf, Wf, C) for a pinhole camera, z-buffered so that nearer surfels hide
farther ones, plus per-cell labels and depth. Cells that see nothing get the
background feature.

``make_flower_scene`` builds the visibility-sensitive case the research plan
cares about (R4): the selected flower, an identical twin, a leaf that occludes
the flower from some viewpoints, stems and a textured wall. Because the twin
looks exactly like the target, appearance alone cannot keep identity; the
read-outs need a spatial prior (as the Stage A memory does).

This is a test bed for the predictor and the predictive cost, **not** a
substitute for V-JEPA features or robot data: E3 / E5 results must come from
recorded episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

BACKGROUND, TARGET, DISTRACTOR, LEAF, STEM, WALL = 0, 1, 2, 3, 4, 5
PLANT_LABELS = (TARGET, DISTRACTOR, LEAF, STEM)


@dataclass
class SyntheticCamera:
    image_hw: Tuple[int, int] = (96, 128)
    grid_hw: Tuple[int, int] = (12, 16)       # 8 px patches
    fov_x_deg: float = 87.0                   # RealSense-like horizontal FoV
    near_m: float = 0.04
    far_m: float = 3.0

    @property
    def K(self) -> np.ndarray:
        H, W = self.image_hw
        f = 0.5 * W / np.tan(np.radians(self.fov_x_deg) / 2)
        return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])

    def cell_centers(self) -> np.ndarray:
        """(Hf*Wf, 2) pixel centres of the grid cells, row-major."""
        hf, wf = self.grid_hw
        H, W = self.image_hw
        u = (np.arange(wf) + 0.5) * W / wf
        v = (np.arange(hf) + 0.5) * H / hf
        uu, vv = np.meshgrid(u, v)
        return np.stack([uu.ravel(), vv.ravel()], -1)


@dataclass
class SyntheticWorld:
    positions: np.ndarray            # (N, 3) world
    radii: np.ndarray                # (N,)
    features: np.ndarray             # (N, C), unit norm
    labels: np.ndarray               # (N,) int
    background: np.ndarray           # (C,) unit norm
    camera: SyntheticCamera = field(default_factory=SyntheticCamera)
    feature_noise: float = 0.0       # std of per-cell noise added before normalising

    @property
    def feat_dim(self) -> int:
        return self.features.shape[1]

    def target_position(self) -> np.ndarray:
        return self.positions[self.labels == TARGET].mean(0)

    def render(self, T_world_cam: np.ndarray, rng: Optional[np.random.Generator] = None,
               chunk: int = 256) -> dict:
        """T_world_cam (..., 4, 4) -> dict with
        Z (..., Hf, Wf, C) float32 unit-norm features, label (..., Hf, Wf) int,
        depth (..., Hf, Wf) float (inf where only background is seen)."""
        T = np.asarray(T_world_cam, float)
        lead = T.shape[:-2]
        T = T.reshape(-1, 4, 4)
        out = [self._render_batch(T[i:i + chunk]) for i in range(0, len(T), chunk)]
        Z = np.concatenate([o[0] for o in out])
        lab = np.concatenate([o[1] for o in out])
        dep = np.concatenate([o[2] for o in out])
        if self.feature_noise > 0:
            rng = rng or np.random.default_rng()
            Z = Z + rng.normal(0, self.feature_noise, Z.shape)
            Z /= np.linalg.norm(Z, axis=-1, keepdims=True)
        hf, wf = self.camera.grid_hw
        return {"Z": Z.reshape(lead + (hf, wf, -1)).astype(np.float32),
                "label": lab.reshape(lead + (hf, wf)),
                "depth": dep.reshape(lead + (hf, wf))}

    def _render_batch(self, T: np.ndarray):
        cam = self.camera
        K = cam.K
        R = np.swapaxes(T[:, :3, :3], 1, 2)                         # cam_R_world
        t = -(R @ T[:, :3, 3:])[..., 0]                             # cam_t_world
        R, t = R.astype(np.float32), t.astype(np.float32)          # float32: render cost dominates the MPC cost
        Pc = np.einsum("bij,nj->bni", R, self.positions.astype(np.float32)) + t[:, None]   # (B, N, 3)
        z = Pc[..., 2]
        ok = (z > cam.near_m) & (z < cam.far_m)
        zs = np.where(ok, z, 1.0)
        uv = np.stack([K[0, 0] * Pc[..., 0] / zs + K[0, 2], K[1, 1] * Pc[..., 1] / zs + K[1, 2]], -1)
        rad = np.float32(K[0, 0]) * self.radii[None].astype(np.float32) / zs   # projected radius, px
        C = cam.cell_centers().astype(np.float32)                    # (M, 2)
        # a surfel covers a cell if the cell centre lies in its projected disk
        # (at least half a cell, so distant surfels do not fall between centres)
        half_cell = 0.5 * cam.image_hw[1] / cam.grid_hw[1]
        r_eff = np.maximum(rad, half_cell)
        d2 = ((uv[:, :, None, :] - C[None, None]) ** 2).sum(-1)      # (B, N, M)
        cover = ok[..., None] & (d2 <= r_eff[..., None] ** 2)
        zc = np.where(cover, z[..., None], np.inf)                   # (B, N, M)
        win = np.argmin(zc, axis=1)                                  # (B, M)
        depth = np.take_along_axis(zc, win[:, None], 1)[:, 0]
        hit = np.isfinite(depth)
        Z = np.where(hit[..., None], self.features[win], self.background[None, None])
        lab = np.where(hit, self.labels[win], BACKGROUND)
        return Z, lab, depth


def target_truth(label: np.ndarray, camera: SyntheticCamera, min_cells: int = 1):
    """Ground-truth target centre (..., 2) px and visibility (...) from labels."""
    hf, wf = camera.grid_hw
    lab = np.asarray(label).reshape(-1, hf * wf)
    m = lab == TARGET
    n = m.sum(1)
    C = camera.cell_centers()
    u = (m[..., None] * C[None]).sum(1) / np.maximum(n, 1)[:, None]
    shp = np.shape(label)[:-2]
    return u.reshape(shp + (2,)), (n >= min_cells).reshape(shp)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _disk_points(center, normal, radius, n, rng):
    normal = _unit(np.asarray(normal, float))
    a = _unit(np.cross(normal, [0, 0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0, 0]))
    b = np.cross(normal, a)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return np.asarray(center, float) + r[:, None] * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)


def make_flower_scene(seed: int = 0, feat_dim: int = 16,
                      target=(0.75, -0.10, 0.45), twin=(0.75, 0.10, 0.45),
                      leaf_center=(0.62, -0.05, 0.47), leaf_normal=(-1.0, 0.0, 0.0),
                      leaf_radius=0.06, wall_x=1.10, feature_noise=0.02,
                      camera: Optional[SyntheticCamera] = None) -> SyntheticWorld:
    """Flower + identical twin + occluding leaf + stems + textured wall."""
    rng = np.random.default_rng(seed)
    flower = _unit(rng.normal(size=feat_dim))
    leaf_f = _unit(rng.normal(size=feat_dim))
    stem_f = _unit(rng.normal(size=feat_dim))
    bg = _unit(rng.normal(size=feat_dim))
    P, Rr, F, L = [], [], [], []

    def add(pts, r, f, lab):
        pts = np.atleast_2d(pts)
        P.append(pts)
        Rr.append(np.full(len(pts), r))
        F.append(np.broadcast_to(f, (len(pts), feat_dim)))
        L.append(np.full(len(pts), lab))

    for c, lab in ((target, TARGET), (twin, DISTRACTOR)):
        add(np.asarray(c) + rng.normal(0, 0.006, (12, 3)), 0.008, flower, lab)
        z = np.linspace(0.0, c[2] - 0.02, 14)
        add(np.column_stack([np.full_like(z, c[0]), np.full_like(z, c[1]), z]), 0.004, stem_f, STEM)
    add(_disk_points(leaf_center, leaf_normal, leaf_radius, 120, rng), 0.008, leaf_f, LEAF)
    # wall: a grid of textured patches (each patch its own feature)
    ys, zs = np.meshgrid(np.linspace(-0.9, 0.9, 13), np.linspace(0.0, 1.2, 9))
    wall = np.column_stack([np.full(ys.size, wall_x), ys.ravel(), zs.ravel()])
    for p in wall:
        add(p, 0.085, _unit(0.6 * bg + rng.normal(0, 0.5, feat_dim)), WALL)
    return SyntheticWorld(positions=np.concatenate(P), radii=np.concatenate(Rr),
                          features=_unit(np.concatenate(F)), labels=np.concatenate(L),
                          background=bg, camera=camera or SyntheticCamera(),
                          feature_noise=feature_noise)


# ------------------------------------------------------------ episode data
def smooth_controls(rng: np.random.Generator, n: int, steps: int, u_low: np.ndarray, u_high: np.ndarray,
                    knots: int = 4, scale: float = 0.6) -> np.ndarray:
    """(n, steps, 8) controls: Gaussian knots, linearly interpolated, clipped."""
    t = np.linspace(0, knots - 1, steps)
    i = np.minimum(np.floor(t).astype(int), knots - 2)
    f = (t - i)[None, :, None]
    z = rng.normal(0, 1, (n, knots, len(u_low))) * scale * np.asarray(u_high)
    U = (1 - f) * z[:, i] + f * z[:, i + 1]
    return np.clip(U, u_low, u_high)


def synthetic_episodes(world: SyntheticWorld, model, camera_pose, starts: np.ndarray, frames: int,
                       stride: int, fk, rng: np.random.Generator, scale: float = 0.6,
                       knots: int = 4) -> list:
    """Random smooth whole-body motions from ``starts`` (N, 9), rendered every
    ``stride`` control steps. ``model`` needs ``rollout(x0, U)``, ``u_low``,
    ``u_high`` (scout_piper_whole_body_mpc.WholeBodyModel). Each episode dict
    holds Z, label, A (Γ between consecutive frames), target / plant masks,
    true target centre u and visibility, and the states."""
    from .action import action_from_states  # local: keeps synthetic.py importable alone

    eps = []
    U = smooth_controls(rng, len(starts), (frames - 1) * stride, model.u_low, model.u_high, knots, scale)
    for x0, Ui in zip(starts, U):
        X = model.rollout(np.asarray(x0, float), Ui[None])[0][::stride]            # (frames, 9)
        r = world.render(camera_pose(X), rng=rng)
        u, vis = target_truth(r["label"], world.camera)
        eps.append({"Z": r["Z"], "label": r["label"], "states": X,
                    "A": action_from_states(X[:-1], X[1:], fk),
                    "target": r["label"] == TARGET,
                    "plant": np.isin(r["label"], PLANT_LABELS),
                    "u": u, "visible": vis})
    return eps


def eye_in_hand_pose(model, link_index: int = 6, offset=(0.0, 0.0, 0.03)):
    """Camera pose function for a WholeBodyModel: optical frame = link6 frame
    shifted by ``offset`` (link6 z is the viewing direction). Hand-eye
    calibration (P0.2.6) replaces this on the robot."""
    T_link_cam = np.eye(4)
    T_link_cam[:3, 3] = offset

    def pose(X: np.ndarray) -> np.ndarray:
        return model.world_frames(np.asarray(X, float))[..., link_index, :, :] @ T_link_cam
    return pose
