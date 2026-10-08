"""Batched Piper forward kinematics from the project URDF.

The chain base_link(Scout) → piper_mount_link → piper_base_link → link1..link6
uses the joint origins of ``scout_piper_description/urdf/_piper_arm.xacro``
and the mount in ``scout_piper.urdf.xacro``. All six arm joints rotate about
their local z axis, so each joint is ``T_origin · Rz(q)``.

``JOINTS`` is the table parsed from the xacro; ``test_kinematics`` re-parses
the file to keep the two in sync. The TCP (grasp point between the fingers)
is an offset along link6 z — calibrate ``tcp_offset_m``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# name, xyz, rpy, (lower, upper) — from _piper_arm.xacro
JOINTS: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float]]] = [
    ("joint1", (0.0, 0.0, 0.123), (0.0, 0.0, 0.0), (-2.618, 2.618)),
    ("joint2", (0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416), (0.0, 3.14)),
    ("joint3", (0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939), (-2.967, 0.0)),
    ("joint4", (-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0), (-1.745, 1.745)),
    ("joint5", (0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0), (-1.22, 1.22)),
    ("joint6", (8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0), (-2.0944, 2.0944)),
]
# base_link (Scout) → piper_mount_link → piper_base_link, from scout_piper.urdf.xacro
MOUNT_XYZ = (0.15, 0.0, 0.18 + 0.005)


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_T(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def parse_xacro_joints(path: str):
    """(name, xyz, rpy, limits) for joint1..joint6 of the forked arm xacro."""
    s = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r'<joint\s+name="\$\{prefix\}(joint[1-6])"\s+type="revolute">(.*?)</joint>', s, re.S):
        body = m.group(2)
        o = re.search(r'<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)"', body)
        lim = re.search(r'<limit\s+lower="([^"]+)"\s+upper="([^"]+)"', body)
        ax = re.search(r'<axis\s+xyz="([^"]+)"', body)
        if tuple(map(float, ax.group(1).split())) != (0.0, 0.0, 1.0):
            raise ValueError(f"{m.group(1)}: only z-axis joints are supported")
        out.append((m.group(1), tuple(map(float, o.group(1).split())),
                    tuple(map(float, o.group(2).split())), tuple(map(float, lim.groups()))))
    return out


@dataclass
class PiperKinematics:
    tcp_offset_m: float = 0.14            # link6 → grasp point along link6 z; calibrate
    joints: list = field(default_factory=lambda: list(JOINTS))
    mount_xyz: Tuple[float, float, float] = MOUNT_XYZ

    def __post_init__(self) -> None:
        self._origins = np.stack([origin_T(xyz, rpy) for _, xyz, rpy, _ in self.joints])
        self.lower = np.array([j[3][0] for j in self.joints])
        self.upper = np.array([j[3][1] for j in self.joints])
        self.T_base_arm = origin_T(self.mount_xyz, (0, 0, 0))
        self.T_tcp = origin_T((0, 0, self.tcp_offset_m), (0, 0, 0))

    def link_frames(self, q: np.ndarray) -> np.ndarray:
        """q (..., 6) -> frames (..., 8, 4, 4) in the Scout base_link frame:
        [arm base, link1..link6, tcp]."""
        q = np.asarray(q, float)
        shp = q.shape[:-1]
        T = np.broadcast_to(self.T_base_arm, shp + (4, 4)).copy()
        frames = [T]
        c, s = np.cos(q), np.sin(q)
        for i in range(6):
            Rz = np.zeros(shp + (4, 4))
            Rz[..., 0, 0], Rz[..., 0, 1], Rz[..., 1, 0], Rz[..., 1, 1] = c[..., i], -s[..., i], s[..., i], c[..., i]
            Rz[..., 2, 2] = Rz[..., 3, 3] = 1.0
            T = T @ self._origins[i] @ Rz
            frames.append(T)
        frames.append(T @ self.T_tcp)
        return np.stack(frames, axis=-3)

    def tcp(self, q: np.ndarray) -> np.ndarray:
        return self.link_frames(q)[..., -1, :, :]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric 6×6 TCP Jacobian [v; ω] in the base frame, q (..., 6)."""
        F = self.link_frames(q)
        p = F[..., -1, :3, 3]
        J = np.zeros(np.shape(q)[:-1] + (6, 6))
        for i in range(6):
            z = F[..., i + 1, :3, 2]                 # joint i axis after its origin
            o = F[..., i + 1, :3, 3]
            J[..., :3, i] = np.cross(z, p - o)
            J[..., 3:, i] = z
        return J

    def wrist_extension(self, q: np.ndarray) -> np.ndarray:
        """Shoulder (joint2) to wrist-centre (joint4/5) distance, q (..., 6).

        Rotations about joint2 keep the distance from a point on its axis, so
        it depends on the elbow (q3) only: |t3 + R3 Rz(q3) w| with joint3's
        origin (t3, R3) in link2 and the joint4 origin w in link3. 0.09 m
        folded, at most 0.537 m (upper arm 0.285 + forearm 0.252)."""
        t3, R3 = self._origins[2][:3, 3], self._origins[2][:3, :3]
        w = self._origins[3][:3, 3]
        q3 = np.asarray(q, float)[..., 2]
        c, s = np.cos(q3), np.sin(q3)
        rw = np.stack([c * w[0] - s * w[1], s * w[0] + c * w[1], np.full_like(c, w[2])], axis=-1)
        return np.linalg.norm(t3 + rw @ R3.T, axis=-1)

    def manipulability(self, q: np.ndarray) -> np.ndarray:
        """Yoshikawa measure sqrt(det(J Jᵀ)) of the translational Jacobian
        (scale-consistent; the full 6×6 mixes metres and radians)."""
        Jv = self.jacobian(q)[..., :3, :]
        return np.sqrt(np.clip(np.linalg.det(Jv @ np.swapaxes(Jv, -1, -2)), 0, None))
