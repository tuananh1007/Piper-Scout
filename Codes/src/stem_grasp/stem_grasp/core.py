"""Servo math + skeleton extraction + grasp candidate selection.

Ported verbatim from ROS 1 stem_grasp_ros1/src/stem_grasp_ros1/core.py.
None of these classes/functions depend on rospy — the port is a
mechanical copy with no semantic changes.

Phase 2 of the ROADMAP replaces FullAdaptiveServoController with MPPI-VS;
preserve the .step() signature so the pipeline node doesn't need to change.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R_scipy
from skimage.morphology import skeletonize
from sklearn.neighbors import KDTree


# ---------------------------------------------------------------------------
# Servo: Kalman filter on stem pixel velocity (sway estimator)
# ---------------------------------------------------------------------------


class StemVelocityObserver:
    def __init__(self, dt: float = 0.01, process_noise: float = 1.0,
                 measurement_noise: float = 5.0) -> None:
        self.dt = dt
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise

    def update(self, observed_uv):
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = np.array(observed_uv) - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        return self.x[:2], self.x[2:]


# ---------------------------------------------------------------------------
# Servo: online image-Jacobian estimator (EMA-blended)
# ---------------------------------------------------------------------------


class OnlineJacobianEstimator:
    def __init__(self, n_features: int = 2, n_dof: int = 3) -> None:
        self.J_hat = np.zeros((n_features, n_dof))
        self.initialized = False
        self.prev_vel = None
        self.prev_feat = None
        self.alpha = 0.7

    def initialize(self, J_analytical) -> None:
        self.J_hat = J_analytical.copy()
        self.initialized = True

    def update(self, curr_feature, curr_vel):
        if self.prev_feat is None:
            self.prev_feat, self.prev_vel = curr_feature.copy(), curr_vel.copy()
            return self.J_hat
        df = curr_feature - self.prev_feat
        dv = curr_vel - self.prev_vel
        norm2 = np.dot(dv, dv)
        if norm2 > 1e-8:
            res = df - self.J_hat @ dv
            upd = np.outer(res, dv) / norm2
            self.J_hat = (1 - self.alpha) * self.J_hat + self.alpha * (
                self.J_hat + upd
            )
        self.prev_feat, self.prev_vel = curr_feature.copy(), curr_vel.copy()
        return self.J_hat


# ---------------------------------------------------------------------------
# Servo: adaptive-gain IBVS with proximity & sway damping
# ---------------------------------------------------------------------------


class FullAdaptiveServoController:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float = 320,
        cy: float = 240,
        lambda_0: float = 0.8,
        lambda_inf: float = 0.07,
        rho: float = 0.3,
        sway_threshold: float = 3.0,
        sway_damping: float = 0.4,
    ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.lambda_0 = lambda_0
        self.lambda_inf = lambda_inf
        self.rho = rho
        self.sway_thresh = sway_threshold
        self.sway_damp = sway_damping
        self.sway_observer = StemVelocityObserver(dt=0.01)
        self.jacobian_est = OnlineJacobianEstimator()

    def _adaptive_gain(self, error_norm):
        return self.lambda_inf + (self.lambda_0 - self.lambda_inf) * np.exp(
            -self.rho * error_norm
        )

    def _depth_compensated_gain(self, error_norm, depth_z,
                                z_safe: float = 0.25, z_min: float = 0.05):
        lam_base = self._adaptive_gain(error_norm)
        if depth_z < z_safe:
            depth_scale = np.clip((depth_z - z_min) / (z_safe - z_min), 0.1, 1.0)
        else:
            depth_scale = 1.0
        return lam_base * depth_scale

    def step(self, raw_uv, desired_uv, depth_z, force_n, commanded_vel=None):
        filt_uv, sway_vel = self.sway_observer.update(np.array(raw_uv))
        sway_speed = float(np.linalg.norm(sway_vel))
        e = filt_uv - np.array(desired_uv)
        error_norm = float(np.linalg.norm(e))

        lam = self._depth_compensated_gain(error_norm, depth_z)
        if sway_speed > self.sway_thresh:
            lam *= self.sway_damp
        lam = float(np.clip(lam, 0.01, 1.2))

        J_analytical = np.array(
            [
                [-self.fx / depth_z, 0, filt_uv[0] / depth_z],
                [0, -self.fy / depth_z, filt_uv[1] / depth_z],
            ]
        )
        if not self.jacobian_est.initialized:
            self.jacobian_est.initialize(J_analytical)
        if commanded_vel is not None:
            self.jacobian_est.update(filt_uv, np.array(commanded_vel))

        vel = -lam * np.linalg.pinv(self.jacobian_est.J_hat) @ e
        vel = np.clip(vel, -0.05, 0.05)
        diag = {
            "error_norm": error_norm,
            "lambda": lam,
            "sway_speed": sway_speed,
            "filtered_uv": filt_uv.tolist(),
            "force_n": force_n,
        }
        return vel, diag


# ---------------------------------------------------------------------------
# Skeleton extraction from a 3D point cloud
# ---------------------------------------------------------------------------


def skeletonize_plant_points(points_xyz, voxel_size: float = 0.003):
    pts = np.asarray(points_xyz)
    if len(pts) == 0:
        return nx.Graph(), np.zeros((0, 3))

    min_b = pts.min(axis=0)
    grid_pts = ((pts - min_b) / voxel_size).astype(int)
    grid_size = grid_pts.max(axis=0) + 2
    grid = np.zeros(grid_size, dtype=bool)
    grid[grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2]] = True

    skeleton = skeletonize(grid)
    edt = distance_transform_edt(grid)
    sk_idx = np.argwhere(skeleton)
    if len(sk_idx) == 0:
        return nx.Graph(), np.zeros((0, 3))

    sk_pts = sk_idx * voxel_size + min_b
    G = nx.Graph()
    for i, p in enumerate(sk_pts):
        G.add_node(i, pos=p, thickness=float(edt[tuple(sk_idx[i])]))

    k = min(6, len(sk_pts) - 1)
    if k >= 1:
        tree = KDTree(sk_pts)
        distances, indices = tree.query(sk_pts, k=k + 1)
        for i in range(len(sk_pts)):
            for j_local in range(1, k + 1):
                j = indices[i, j_local]
                d = distances[i, j_local]
                t_avg = (G.nodes[i]["thickness"] + G.nodes[j]["thickness"]) / 2
                weight = d / (t_avg + 1e-6)
                G.add_edge(i, j, weight=weight)

    return G, sk_pts


def extract_main_stem(
    G,
    sk_pts,
    vertical_axis: int = 1,
    connectivity_weight: float = 1.0,
    thickness_weight: float = 0.8,
    verticality_weight: float = 1.2,
):
    if len(sk_pts) == 0 or G.number_of_nodes() == 0:
        return np.zeros((0, 3))

    nodes = list(G.nodes)
    positions = np.array([G.nodes[n]["pos"] for n in nodes])
    thickness = np.array(
        [G.nodes[n].get("thickness", 0.0) for n in nodes], dtype=np.float32
    )

    root_idx = int(np.argmin(positions[:, vertical_axis]))
    root_node = nodes[root_idx]

    # Monotonic upward search: only allow transitions to nodes with larger
    # vertical coordinate.
    order = np.argsort(positions[:, vertical_axis])
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    best_score = {root_node: 0.0}
    best_prev = {root_node: None}

    for idx in order:
        u = nodes[idx]
        if u not in best_score:
            continue

        pos_u = positions[idx]
        y_u = pos_u[vertical_axis]

        for v in G.neighbors(u):
            v_idx = node_to_idx[v]
            pos_v = positions[v_idx]
            y_v = pos_v[vertical_axis]

            delta_y = y_v - y_u
            if delta_y <= 1e-9:
                continue

            edge_vec = pos_v - pos_u
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len <= 1e-9:
                continue

            avg_thickness = 0.5 * (thickness[idx] + thickness[v_idx])
            verticality = delta_y / edge_len
            connectivity = 1.0 / edge_len

            step_score = (
                connectivity_weight * connectivity
                + thickness_weight * avg_thickness
                + verticality_weight * verticality
            )

            cand_score = best_score[u] + step_score
            if cand_score > best_score.get(v, -np.inf):
                best_score[v] = cand_score
                best_prev[v] = u

    if len(best_score) == 1:
        return sk_pts[[root_idx]]

    terminal_node = max(
        best_score.keys(),
        key=lambda n: (positions[node_to_idx[n], vertical_axis], best_score[n]),
    )

    path_nodes = []
    cur = terminal_node
    while cur is not None:
        path_nodes.append(cur)
        cur = best_prev.get(cur)

    path_nodes.reverse()
    path_idx = [node_to_idx[n] for n in path_nodes]
    return sk_pts[path_idx]


# ---------------------------------------------------------------------------
# Grasp candidate selection along the extracted stem skeleton
# ---------------------------------------------------------------------------


def select_grasp_candidates(
    stem_points,
    num_candidates: int = 5,
    gripper_offset: float = 0.05,
    target_offset=None,
):
    n = len(stem_points)
    if n < 2:
        return []

    offset = max(float(gripper_offset if target_offset is None else target_offset), 0.0)

    candidates = []
    for ratio in np.linspace(0.3, 0.7, num_candidates):
        grasp_idx = int(n * ratio)
        grasp_idx = int(np.clip(grasp_idx, 1, n - 2))
        grasp_pos = stem_points[grasp_idx]

        p0 = stem_points[max(0, grasp_idx - 2)]
        p1 = stem_points[min(n - 1, grasp_idx + 2)]
        stem_dir = p1 - p0
        stem_dir /= np.linalg.norm(stem_dir) + 1e-9

        world_x = np.array([1.0, 0.0, 0.0])
        approach = np.cross(stem_dir, world_x)
        if np.linalg.norm(approach) < 1e-3:
            approach = np.cross(stem_dir, np.array([0.0, 0.0, 1.0]))
        approach /= np.linalg.norm(approach)

        z_axis = -approach
        y_axis = stem_dir
        x_axis = np.cross(y_axis, z_axis)
        norm = np.linalg.norm(x_axis)
        x_axis = np.array([1.0, 0.0, 0.0]) if norm < 1e-9 else x_axis / norm

        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        quat = R_scipy.from_matrix(rot_matrix).as_quat()
        pre_pos = grasp_pos + approach * offset
        score = 1.0 - abs(ratio - 0.45)
        candidates.append(
            {
                "pos": grasp_pos,
                "pre_pos": pre_pos,
                "quat": quat,
                "score": float(score),
                "ratio": float(ratio),
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
