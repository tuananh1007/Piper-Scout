"""Synthetic cluttered-plant scenes for the whole-body benchmark (P3.3, WE3).

A scene is a row of 1–3 potted plants in front of the robot (base at the
origin facing +x): stems, stakes and branches are capsules (hard), pots are
spheres (hard), leaves are discs (soft, ``discs_leaf_fn``). The target is a
point on a branch or stem (a peduncle) 0.25–0.65 m above the floor; the goal
is the pre-grasp point ``offset_m`` (0.12, stem_grasp's
``target_position_offset_m``) in front of it along the horizontal line from the
robot, which is also the approach axis.

Analytic geometry, not a replacement for recorded scenes: it exercises
reachability, base motion and clearance around thin structures, with the
same distance-function interface the semantic scene provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .scene_adapter import capsules_distance_fn, discs_leaf_fn, gridded


@dataclass
class PlantScene:
    name: str
    cap_a: np.ndarray                 # (M, 3) capsule start
    cap_b: np.ndarray                 # (M, 3) capsule end
    cap_r: np.ndarray                 # (M,)
    pots: np.ndarray                  # (P, 3) pot sphere centres
    pot_r: np.ndarray                 # (P,)
    leaf_c: np.ndarray                # (L, 3)
    leaf_n: np.ndarray                # (L, 3)
    leaf_r: np.ndarray                # (L,)
    target: np.ndarray                # (3,) peduncle point
    axis: np.ndarray                  # (3,) horizontal approach direction (unit)
    goal: np.ndarray                  # (3,) pre-grasp point
    kinds: List[str] = field(default_factory=list)   # per capsule: stem / stake / branch

    def distance_fn(self):
        return capsules_distance_fn(self.cap_a, self.cap_b, self.cap_r, (self.pots, self.pot_r))

    def leaf_fn(self, weight: float = 50.0, d_soft: float = 0.02):
        return discs_leaf_fn(self.leaf_c, self.leaf_n, self.leaf_r, weight, d_soft)

    def bounds(self, margin: float = 0.7):
        """Box around the plants and the robot's start area."""
        pts = np.concatenate([self.cap_a, self.cap_b, self.pots, [[0.0, 0.0, 0.0]]])
        lo = pts.min(0) - margin
        hi = pts.max(0) + margin
        return np.maximum(lo, [-0.8, -1.2, -0.1]), np.minimum(hi, [2.6, 1.2, 1.1])

    def grid_fields(self, voxel_size: float = 0.01):
        """(distance_fn, leaf_fn) on a voxel grid: fast, for the planners. Keep
        the exact ``distance_fn()`` for reported clearances."""
        lo, hi = self.bounds()
        return (gridded(self.distance_fn(), lo, hi, voxel_size),
                gridded(self.leaf_fn(), lo, hi, voxel_size, values=True))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def make_plant_scene(seed: int, offset_m: float = 0.12, x_range=(0.75, 1.7),
                     y_range=(-0.35, 0.35), max_plants: int = 3,
                     min_goal_clearance_m: float = 0.06) -> PlantScene:
    rng = np.random.default_rng(seed)
    for _ in range(200):                                   # resample until the goal is clear
        A, B, R, kinds = [], [], [], []
        pots, pot_r, lc, ln, lr = [], [], [], [], []
        candidates = []                                    # possible targets
        n = int(rng.integers(1, max_plants + 1))
        xs = np.sort(rng.uniform(*x_range, n))
        for i in range(n):
            base = np.r_[xs[i], rng.uniform(*y_range), 0.0]
            h = rng.uniform(0.45, 0.8)
            top = base + np.r_[rng.normal(0, 0.03), rng.normal(0, 0.03), h]
            A.append(base); B.append(top); R.append(rng.uniform(0.006, 0.012)); kinds.append("stem")
            if rng.random() < 0.5:                         # support stake beside the stem
                off = np.r_[rng.normal(0, 0.03), rng.normal(0, 0.03), 0.0]
                A.append(base + off); B.append(base + off + np.r_[0, 0, h + 0.05])
                R.append(0.006); kinds.append("stake")
            pots.append(base + np.r_[0, 0, 0.04]); pot_r.append(rng.uniform(0.09, 0.12))
            for _b in range(int(rng.integers(2, 5))):
                hb = rng.uniform(0.2, h - 0.05)
                s = base + (top - base) * hb / h
                yaw = rng.uniform(-np.pi, np.pi)
                slope = np.radians(rng.uniform(20, 50))
                d = np.r_[np.cos(yaw) * np.cos(slope), np.sin(yaw) * np.cos(slope), np.sin(slope)]
                tip = s + rng.uniform(0.08, 0.18) * d
                A.append(s); B.append(tip); R.append(rng.uniform(0.004, 0.006)); kinds.append("branch")
                candidates.append(tip)
                for _l in range(int(rng.integers(1, 3))):  # leaves near the branch
                    lc.append(tip + rng.normal(0, 0.03, 3)); ln.append(_unit(rng.normal(size=3)))
                    lr.append(rng.uniform(0.03, 0.06))
            for _s in range(2):                            # peduncle-like points on the stem
                candidates.append(base + (top - base) * rng.uniform(0.45, 0.9))
        cand = np.array(candidates)
        cand = cand[(cand[:, 2] > 0.25) & (cand[:, 2] < 0.65)]
        if not len(cand):
            continue
        target = cand[int(rng.integers(len(cand)))]
        axis = _unit(np.r_[target[:2], 0.0])
        goal = target - offset_m * axis
        scene = PlantScene(name=f"P{seed:02d}", cap_a=np.array(A), cap_b=np.array(B), cap_r=np.array(R),
                           pots=np.array(pots), pot_r=np.array(pot_r), leaf_c=np.array(lc),
                           leaf_n=np.array(ln), leaf_r=np.array(lr), target=target, axis=axis,
                           goal=goal, kinds=kinds)
        d, _ = scene.distance_fn()(goal[None])
        if d[0] >= min_goal_clearance_m:
            return scene
    raise RuntimeError(f"no clear goal found for seed {seed}")
