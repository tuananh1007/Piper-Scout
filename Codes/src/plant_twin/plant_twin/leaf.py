"""Leaf model: textured, curved triangle mesh with explicit outline and holes.

Parameterisation (fixed per leaf after the first frame):
  * rest shape  — 2D outline polygon + hole polygons in the leaf's own (u, v)
                  plane, triangulated once on a regular lattice (triangles
                  outside the outline or inside a hole are dropped).
  * rigid pose  — rotation vector (3) + translation (3).
  * bending     — ``grid``² scalar heights on a lattice, turned into a smooth
                  height field by ``geometry.bump_basis``; applied along the
                  leaf normal before the rigid transform.

Residuals (all returned by ``LeafModel.residuals`` and stacked by the
fitter):
  data      — tracked 3D points to nearest mesh vertex.
  stretch   — mesh edge lengths vs rest lengths (leaves bend, hardly stretch).
  prior     — bending heights pulled toward zero (rest shape).
  temporal  — parameters pulled toward previous frame.
  contact   — gripper contact point must lie on the surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (apply_rigid, bump_basis, point_in_polygon,
                       rotation_jacobian, rotvec_to_matrix)


@dataclass
class LeafWeights:
    data: float = 1.0          # point-to-plane (obs -> mesh)
    p2p: float = 0.3           # point-to-point obs -> mesh, relative to data
    reverse: float = 0.5       # point-to-point mesh -> obs, relative to data
    stretch: float = 20.0
    prior: float = 0.5
    temporal: float = 0.5      # > ~1 makes in-plane motion lag the data
    contact: float = 10.0


@dataclass
class LeafModel:
    outline: np.ndarray                       # (M, 2) in metres, leaf plane
    holes: Sequence[np.ndarray] = field(default_factory=list)
    grid: int = 4                              # bending lattice per axis
    mesh_res: int = 20                         # lattice used to triangulate;
                                               # cell must be < outline detail
                                               # you want the pose to resolve
    weights: LeafWeights = field(default_factory=LeafWeights)

    def __post_init__(self) -> None:
        self.outline = np.asarray(self.outline, dtype=float)
        self.holes = [np.asarray(h, dtype=float) for h in self.holes]
        self._build_mesh()
        self.n_bend = self.grid * self.grid
        self.n_params = 6 + self.n_bend
        self._basis = bump_basis(self.uv, self.grid)
        # Where the stem attaches (index into rest_xy). Defaults to the vertex
        # nearest the rest-plane origin; ``set_petiole`` overrides it from the
        # mask/stem-base geometry.
        self.petiole_idx = int(np.argmin(np.linalg.norm(self.rest_xy, axis=1)))
        self.colors: Optional[np.ndarray] = None   # (V, 3) RGB in [0, 1]

    # ------------------------------------------------------------------ mesh
    def _build_mesh(self) -> None:
        lo, hi = self.outline.min(0), self.outline.max(0)
        n = self.mesh_res
        gx = np.linspace(lo[0], hi[0], n)
        gy = np.linspace(lo[1], hi[1], n)
        xx, yy = np.meshgrid(gx, gy, indexing="ij")
        verts = np.stack([xx.ravel(), yy.ravel()], axis=1)

        in_outline = np.array([point_in_polygon(p, self.outline) for p in verts])
        in_hole = np.array([any(point_in_polygon(p, h) for h in self.holes) for p in verts])
        # A triangle is kept when at least two vertices are inside the outline
        # (so the boundary stays closed rather than pixelated) and none of its
        # vertices sits in a hole (holes are explicit and must stay open).
        idx = lambda i, j: i * n + j  # noqa: E731
        tris = []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, d = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
                for tri in ((a, b, c), (b, d, c)):
                    t = list(tri)
                    if in_outline[t].sum() >= 2 and not in_hole[t].any():
                        tris.append(tri)
        tris = np.array(tris, dtype=int).reshape(-1, 3)
        used = np.unique(tris)
        remap = -np.ones(len(verts), dtype=int)
        remap[used] = np.arange(len(used))
        self.rest_xy = verts[used]
        self.faces = remap[tris]
        span = np.maximum(hi - lo, 1e-9)
        self.uv = (self.rest_xy - lo) / span            # normalised [0,1]²
        edges = set()
        for t in self.faces:
            for k in range(3):
                e = tuple(sorted((int(t[k]), int(t[(k + 1) % 3]))))
                edges.add(e)
        self.edges = np.array(sorted(edges), dtype=int).reshape(-1, 2)
        self.rest_edge_len = np.linalg.norm(
            self.rest_xy[self.edges[:, 0]] - self.rest_xy[self.edges[:, 1]], axis=1
        )

    # ------------------------------------------------------------ evaluation
    def initial_params(self, rotvec=(0, 0, 0), trans=(0, 0, 0)) -> np.ndarray:
        p = np.zeros(self.n_params)
        p[:3], p[3:6] = rotvec, trans
        return p

    def split(self, params: np.ndarray):
        return params[:3], params[3:6], params[6:]

    def local_vertices(self, bend: np.ndarray) -> np.ndarray:
        """Rest-plane vertices lifted along +z by the bending height field."""
        z = self._basis @ bend
        return np.column_stack([self.rest_xy, z])

    def vertices(self, params: np.ndarray) -> np.ndarray:
        rv, t, bend = self.split(params)
        return apply_rigid(self.local_vertices(bend), rv, t)

    def normals(self, params: np.ndarray) -> np.ndarray:
        v = self.vertices(params)
        fn = np.cross(v[self.faces[:, 1]] - v[self.faces[:, 0]],
                      v[self.faces[:, 2]] - v[self.faces[:, 0]])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    # ---------------------------------------------------- residuals + jacobian
    def vertex_jacobian(self, params: np.ndarray) -> np.ndarray:
        """d v_i / d params for every vertex -> (V, 3, n_params).

        v_i = R(rv) l_i(bend) + t with l_i = [x_i, y_i, (Φ bend)_i]:
          d/dt    = I
          d/drv   = rotation_jacobian(rv, l)
          d/dbend = R e_z ⊗ Φ_i
        """
        rv, _, bend = self.split(params)
        local = self.local_vertices(bend)
        R = rotvec_to_matrix(rv)
        D = np.zeros((len(local), 3, self.n_params))
        D[:, :, 0:3] = rotation_jacobian(rv, local)
        D[:, :, 3:6] = np.eye(3)
        D[:, :, 6:] = R[:, 2][None, :, None] * self._basis[:, None, :]
        return D

    def evaluate(
        self,
        params: np.ndarray,
        observed: np.ndarray,
        prev_params: Optional[np.ndarray] = None,
        contact_point: Optional[np.ndarray] = None,
        want_jac: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Stacked residual vector and (optionally) its analytic Jacobian.

        One pass shares the KD-tree queries and normals between the two.
        Correspondences and normals are held fixed in the Jacobian
        (Gauss-Newton ICP linearisation).

        Blocks, in order:
          data      point-to-plane (obs -> nearest vertex, along its normal),
                    point-to-point obs -> mesh (weight p2p), and mesh -> obs
                    (weight reverse). Pure point-to-point on a lattice stalls
                    half a cell from the true pose because in-plane offsets
                    give no gradient; the plane term lets interior points
                    slide and leaves in-plane alignment to the outline. The
                    reverse term stops the mesh drifting off the observed
                    region.
          stretch   edge lengths vs rest lengths.
          prior     bending heights -> 0.
          temporal  params -> previous frame.
          contact   fingertip must lie on the surface.
        """
        w = self.weights
        _, _, bend = self.split(params)
        v = self.vertices(params)
        P = self.n_params
        D = self.vertex_jacobian(params) if want_jac else None
        r_blocks, J_blocks = [], []

        observed, obs_tree = self._observed(observed)
        if len(observed):
            nrm = self.normals(params)
            _, nn = cKDTree(v).query(observed)
            diff = observed - v[nn]
            _, mn = obs_tree.query(v)
            r_blocks.append(w.data * np.einsum("ij,ij->i", diff, nrm[nn]))
            r_blocks.append(w.p2p * w.data * diff.ravel())
            r_blocks.append(w.reverse * w.data * (v - observed[mn]).ravel())
            if want_jac:
                Dnn = D[nn]
                J_blocks.append(-w.data * np.einsum("ij,ijk->ik", nrm[nn], Dnn))
                J_blocks.append(-w.p2p * w.data * Dnn.reshape(-1, P))
                J_blocks.append(w.reverse * w.data * D.reshape(-1, P))

        a, b = self.edges[:, 0], self.edges[:, 1]
        d = v[a] - v[b]
        el = np.linalg.norm(d, axis=1)
        r_blocks.append(w.stretch * (el - self.rest_edge_len))
        if want_jac:
            u = d / np.maximum(el, 1e-12)[:, None]
            J_blocks.append(w.stretch * np.einsum("ij,ijk->ik", u, D[a] - D[b]))

        r_blocks.append(w.prior * bend)
        if want_jac:
            Jp = np.zeros((self.n_bend, P))
            Jp[:, 6:] = w.prior * np.eye(self.n_bend)
            J_blocks.append(Jp)

        if prev_params is not None:
            r_blocks.append(w.temporal * (params - prev_params))
            if want_jac:
                J_blocks.append(w.temporal * np.eye(P))

        if contact_point is not None:
            cp = np.asarray(contact_point, dtype=float)
            _, nn = cKDTree(v).query(cp)
            r_blocks.append(w.contact * (cp - v[nn]))
            if want_jac:
                J_blocks.append(-w.contact * D[nn])

        r = np.concatenate(r_blocks)
        return r, (np.vstack(J_blocks) if want_jac else None)

    def _observed(self, observed):
        """Observations are fixed within a frame but ``evaluate`` runs many
        times per frame, so the KD-tree over them is cached by identity."""
        cache = getattr(self, "_obs_cache", None)
        if cache is not None and cache[0] is observed:
            return cache[1], cache[2]
        arr = np.asarray(observed, dtype=float).reshape(-1, 3)
        tree = cKDTree(arr) if len(arr) else None
        self._obs_cache = (observed, arr, tree)
        return arr, tree

    def residuals(self, params, observed, prev_params=None, contact_point=None):
        return self.evaluate(params, observed, prev_params, contact_point, want_jac=False)[0]

    def jacobian(self, params, observed, prev_params=None, contact_point=None):
        return self.evaluate(params, observed, prev_params, contact_point, want_jac=True)[1]

    # --------------------------------------------------------------- exports
    def set_petiole(self, xy: np.ndarray) -> None:
        """Pick the mesh vertex nearest a rest-plane point as the attachment."""
        self.petiole_idx = int(np.argmin(np.linalg.norm(self.rest_xy - xy, axis=1)))

    def tip_point(self, params: np.ndarray) -> np.ndarray:
        """World-space point where the stem attaches."""
        return self.vertices(params)[self.petiole_idx]

    def texture_from_cloud(self, params: np.ndarray, points: np.ndarray,
                           colors: np.ndarray, max_dist_m: float = 0.01) -> None:
        """Per-vertex colour = colour of the nearest cloud point (first frame).

        Vertices with no cloud point within ``max_dist_m`` take the mean colour
        so outline-fill triangles don't show up black.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        colors = np.asarray(colors, dtype=float).reshape(-1, 3)
        if colors.max() > 1.0:
            colors = colors / 255.0
        v = self.vertices(params)
        d, nn = cKDTree(points).query(v)
        col = colors[nn]
        far = d > max_dist_m
        col[far] = colors.mean(0)
        self.colors = col
