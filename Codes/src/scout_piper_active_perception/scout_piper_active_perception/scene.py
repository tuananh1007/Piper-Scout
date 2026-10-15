"""Simulated plant scene for active perception: exact ray casting and a depth
sensor model with the failure that motivates Phase 4.

The geometry is scout_piper_whole_body_mpc's ``make_plant_scene`` (stems,
stakes and branches as capsules, pots as spheres, leaves as discs) plus a
floor. ``make_ap_scene`` moves the target onto the middle of a capsule (the
grasp point on a peduncle-like branch or stem, at least ``target_half_len_m``
from its ends) and places leaves between a standoff camera and the target, so
the first view is partly or fully occluded.

``render`` casts one ray per pixel and returns z-depth and a class label per
pixel (``CLASSES``: hits on the target capsule within ``target_half_len_m`` of
the target are ``target``). ``DepthSensor`` turns that into what an active
stereo camera at short range (the wrist D405) returns: depth noise growing
with z², and dropout on structures that project narrower than a few pixels
(thin stems against a far background are where stereo matching fails, the
risk "thin peduncle depth failure" in ROADMAP.md). The dropout is the reason a
closer or less oblique view can be worth more than one more view from far.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

from scout_piper_whole_body_mpc.scenes import PlantScene, make_plant_scene

CLASSES = ("stem", "branch", "leaf", "target", "other")
NONE = -1                                    # label of a pixel without a hit


@dataclass
class Camera:
    image_hw: Tuple[int, int] = (240, 424)   # D405 848 x 480 at half resolution
    fov_x_deg: float = 87.0
    near_m: float = 0.07                     # D405 minimum range
    far_m: float = 1.0

    @property
    def K(self) -> np.ndarray:
        H, W = self.image_hw
        f = 0.5 * W / np.tan(np.radians(self.fov_x_deg) / 2)
        return np.array([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]])

    @property
    def f(self) -> float:
        return float(self.K[0, 0])


@dataclass
class APScene:
    plant: PlantScene
    target_capsule: int                      # index into plant.cap_a / cap_b
    target_half_len_m: float = 0.03

    @property
    def target(self) -> np.ndarray:
        return self.plant.target

    def joint_axis(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """(point at the target, unit direction, radius) of the true joint."""
        a, b = self.plant.cap_a[self.target_capsule], self.plant.cap_b[self.target_capsule]
        d = (b - a) / np.linalg.norm(b - a)
        return self.plant.target.copy(), d, float(self.plant.cap_r[self.target_capsule])

    def capsule_class(self) -> np.ndarray:
        kinds = self.plant.kinds
        return np.array([CLASSES.index("branch") if k == "branch" else
                         CLASSES.index("stem") if k == "stem" else CLASSES.index("other") for k in kinds])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def make_ap_scene(seed: int, standoff_m: float = 0.45, occluders: Tuple[int, int] = (1, 3),
                  target_half_len_m: float = 0.03, prefer_branch: bool = True) -> APScene:
    """A plant scene with the target in the middle of a capsule and 1-3
    leaves (discs) 4-9 cm in front of it toward the standoff camera on the
    robot's approach line."""
    rng = np.random.default_rng(10_000 + seed)
    for k in range(50):
        ps = make_plant_scene(seed + 1000 * k)
        L = np.linalg.norm(ps.cap_b - ps.cap_a, axis=1)
        ok = (L > 2 * target_half_len_m + 0.02) & np.array([kd in ("stem", "branch") for kd in ps.kinds])
        idx = np.flatnonzero(ok)
        if prefer_branch and any(ps.kinds[i] == "branch" for i in idx):
            idx = np.array([i for i in idx if ps.kinds[i] == "branch"])
        rng.shuffle(idx)
        for i in idx:
            a, b = ps.cap_a[i], ps.cap_b[i]
            margin = (target_half_len_m + 0.01) / L[i]
            t = rng.uniform(margin, 1 - margin)
            target = a + t * (b - a)
            if not 0.25 < target[2] < 0.65:
                continue
            axis = _unit(np.r_[target[:2], 0.0])
            cam = target - standoff_m * axis
            lc, ln, lr = [list(x) for x in (ps.leaf_c, ps.leaf_n, ps.leaf_r)]
            for _ in range(int(rng.integers(occluders[0], occluders[1] + 1))):
                s = rng.uniform(0.04, 0.09)
                side = np.cross(axis, [0.0, 0.0, 1.0])
                c = target - s * axis + rng.normal(0, 0.02) * side + rng.normal(0, 0.015) * np.r_[0, 0, 1.0]
                n = _unit(axis + rng.normal(0, 0.4, 3))
                lc.append(c); ln.append(n); lr.append(rng.uniform(0.025, 0.045))
            plant = replace(ps, target=target, axis=axis, goal=target - 0.12 * axis,
                            leaf_c=np.array(lc), leaf_n=np.array(ln), leaf_r=np.array(lr))
            # the leaves must leave the target capsule and the standoff camera free
            from scout_piper_whole_body_mpc.scene_adapter import disc_distance  # noqa: PLC0415
            if disc_distance(cam[None], plant.leaf_c, plant.leaf_n, plant.leaf_r).min() < 0.05:
                continue
            seg = a[None] + np.linspace(0, 1, 40)[:, None] * (b - a)[None]
            if disc_distance(seg, plant.leaf_c, plant.leaf_n, plant.leaf_r).min() < ps.cap_r[i] + 0.003:
                continue
            return APScene(plant=plant, target_capsule=int(i), target_half_len_m=target_half_len_m)
    raise RuntimeError(f"no target for seed {seed}")


# ------------------------------------------------------------------ rendering
def _capsule_hits(o, u, a, b, r):
    """Ray (o, u) (N, 3) against capsules (M,): distance t (N, M), inf on a miss."""
    ba = (b - a)[None]                                      # (1, M, 3)
    oa = o[:, None] - a[None]                               # (N, M, 3)
    baba = (ba * ba).sum(-1)
    bard = (ba * u[:, None]).sum(-1)
    baoa = (ba * oa).sum(-1)
    rdoa = (u[:, None] * oa).sum(-1)
    oaoa = (oa * oa).sum(-1)
    A = baba - bard * bard
    B = baba * rdoa - baoa * bard
    C = baba * oaoa - baoa * baoa - (r * r)[None] * baba
    h = B * B - A * C
    t = np.full(h.shape, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        tb = (-B - np.sqrt(np.maximum(h, 0.0))) / A
        y = baoa + tb * bard
        body = (h >= 0) & (A > 1e-12) & (y > 0) & (y < baba) & (tb > 0)
        t[body] = tb[body]
        # end caps (spheres at a and b)
        for end in (a, b):
            oc = o[:, None] - end[None]
            bb = (u[:, None] * oc).sum(-1)
            cc = (oc * oc).sum(-1) - (r * r)[None]
            hh = bb * bb - cc
            ts = -bb - np.sqrt(np.maximum(hh, 0.0))
            cap = (hh >= 0) & (ts > 0) & ~body
            t = np.where(cap, np.minimum(t, ts), t)
    return t


def _sphere_hits(o, u, c, r):
    oc = o[:, None] - c[None]
    b = (u[:, None] * oc).sum(-1)
    h = b * b - ((oc * oc).sum(-1) - (r * r)[None])
    t = -b - np.sqrt(np.maximum(h, 0.0))
    return np.where((h >= 0) & (t > 0), t, np.inf)


def _disc_hits(o, u, c, n, r):
    with np.errstate(divide="ignore", invalid="ignore"):
        den = (u[:, None] * n[None]).sum(-1)
        t = ((c[None] - o[:, None]) * n[None]).sum(-1) / den
        p = o[:, None] + t[..., None] * u[:, None]
        inside = np.linalg.norm(p - c[None], axis=-1) <= r[None]
    return np.where((np.abs(den) > 1e-9) & (t > 0) & inside, t, np.inf)


def render(scene: APScene, T_world_cam: np.ndarray, camera: Camera, chunk: int = 16384) -> Dict[str, np.ndarray]:
    """z-depth (H, W) in metres (inf without a hit), label (H, W) index into
    CLASSES (NONE without a hit), and ``radius`` (H, W): the radius of the
    hit capsule (0 for other surfaces), for the sensor's thin-structure model."""
    H, W = camera.image_hw
    K = camera.K
    vv, uu = np.mgrid[0:H, 0:W]
    rays_c = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3)
    zdir = rays_c[:, 2] / np.linalg.norm(rays_c, axis=1)                  # cos to the optical axis
    R, c = T_world_cam[:3, :3], T_world_cam[:3, 3]
    u_w = (rays_c / np.linalg.norm(rays_c, axis=1, keepdims=True)) @ R.T
    p = scene.plant
    cls_cap = scene.capsule_class()
    leaf, other = CLASSES.index("leaf"), CLASSES.index("other")
    depth = np.full(H * W, np.inf)
    label = np.full(H * W, NONE, np.int64)
    radius = np.zeros(H * W)
    a_t, b_t = p.cap_a[scene.target_capsule], p.cap_b[scene.target_capsule]
    d_t = (b_t - a_t) / np.linalg.norm(b_t - a_t)
    for s in range(0, H * W, chunk):
        o = np.broadcast_to(c, (min(chunk, H * W - s), 3))
        u = u_w[s:s + chunk]
        groups = [_capsule_hits(o, u, p.cap_a, p.cap_b, p.cap_r)]
        groups.append(_sphere_hits(o, u, p.pots, p.pot_r) if len(p.pots) else np.full((len(u), 0), np.inf))
        groups.append(_disc_hits(o, u, p.leaf_c, p.leaf_n, p.leaf_r) if len(p.leaf_c) else np.full((len(u), 0), np.inf))
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = np.where(u[:, 2] < -1e-9, -c[2] / u[:, 2], np.inf)        # floor z = 0
        groups.append(tf[:, None])
        T = np.concatenate(groups, 1)
        k = np.argmin(T, 1)
        t = T[np.arange(len(u)), k]
        nc, nps, nl = p.cap_a.shape[0], len(p.pots), len(p.leaf_c)
        lab = np.where(k < nc, cls_cap[np.minimum(k, nc - 1)], np.where(k < nc + nps, other,
                       np.where(k < nc + nps + nl, leaf, other)))
        rad = np.where(k < nc, p.cap_r[np.minimum(k, nc - 1)], 0.0)
        hit = np.isfinite(t)
        x = c + np.where(hit, t, 0.0)[:, None] * u
        on_t = (k == scene.target_capsule) & hit & (np.abs((x - p.target) @ d_t) <= scene.target_half_len_m)
        lab = np.where(on_t, CLASSES.index("target"), lab)
        depth[s:s + chunk] = np.where(hit, t * zdir[s:s + chunk], np.inf)
        label[s:s + chunk] = np.where(hit, lab, NONE)
        radius[s:s + chunk] = np.where(hit, rad, 0.0)
    return {"depth": depth.reshape(H, W), "label": label.reshape(H, W), "radius": radius.reshape(H, W)}


@dataclass
class DepthSensor:
    """Depth noise σ_z = sigma0 + k2·z² and thin-structure dropout: a pixel on a
    capsule whose projected width w = 2r·f/z is below ``thin_hi_px`` drops with
    probability rising linearly to 1 at ``thin_lo_px``."""
    sigma0: float = 0.0005
    k2: float = 0.002
    thin_lo_px: float = 2.0
    thin_hi_px: float = 6.0

    def sigma_z(self, z):
        return self.sigma0 + self.k2 * np.asarray(z) ** 2

    def drop_probability(self, radius_m, z, f):
        w = 2.0 * np.asarray(radius_m) * f / np.maximum(np.asarray(z), 1e-6)
        p = (self.thin_hi_px - w) / (self.thin_hi_px - self.thin_lo_px)
        return np.where(np.asarray(radius_m) > 0, np.clip(p, 0.0, 1.0), 0.0)

    def sense(self, frame: Dict[str, np.ndarray], camera: Camera,
              rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        rng = rng or np.random.default_rng()
        z = frame["depth"]
        ok = np.isfinite(z) & (z > camera.near_m) & (z < camera.far_m)
        drop = rng.random(z.shape) < self.drop_probability(frame["radius"], np.where(ok, z, 1.0), camera.f)
        ok &= ~drop
        zn = np.where(ok, z + rng.normal(0.0, 1.0, z.shape) * self.sigma_z(np.where(ok, z, 0.0)), np.nan)
        return {"depth": zn, "label": np.where(ok, frame["label"], NONE)}


def backproject(depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, mask: Optional[np.ndarray] = None):
    """World points (N, 3) of the valid pixels (and ``mask``)."""
    ok = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        ok &= mask
    v, u = np.nonzero(ok)
    z = depth[v, u]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    return pc @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
