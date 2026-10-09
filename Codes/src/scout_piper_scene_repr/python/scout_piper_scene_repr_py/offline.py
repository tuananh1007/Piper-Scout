"""Build the semantic voxel map offline from an exported episode (P1.4.2 on recorded scenes).

An episode exported with ``scout_piper_jepa.episode export`` from a bag
recorded with ``scripts/record_bag.sh scene`` holds aligned depth, the colour
intrinsics, the camera pose of every frame and the merged semantic labels
(/stem_grasp/semantic_label). ``map_from_episode`` integrates them exactly as
``scene_query_node`` would live, so planners can be compared on the same
recorded scene (whole_body_mpc benchmarks/semantic_vs_occupancy.py --episode).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .voxel_map import SemanticVoxelMap, grid_around

LABEL_IDS = {"stem": 1, "branch": 2, "leaf": 3, "target": 4}


def map_from_episode(ep: Dict[str, np.ndarray], center: Optional[Sequence[float]] = None,
                     half_extent_m: float = 0.5, voxel_size_m: float = 0.01, stride: int = 4,
                     every: int = 1, max_range_m: float = 1.0) -> SemanticVoxelMap:
    """Integrate every ``every``-th frame with depth, pose and labels.

    ``center`` defaults to the mean of the valid depth points in the world frame."""
    for k in ("depth", "K", "T_world_cam", "labels"):
        if k not in ep:
            raise KeyError(f"episode has no {k!r}: record with record_bag.sh scene and export with --labels")
    depth = np.asarray(ep["depth"], np.float32)
    K, T, labels = np.asarray(ep["K"], float), np.asarray(ep["T_world_cam"], float), ep["labels"]
    ok = [i for i in range(0, len(depth), every) if np.isfinite(T[i]).all()]
    if not ok:
        raise ValueError("no frame with a camera pose")
    if center is None:
        pts = []
        for i in ok[:: max(len(ok) // 10, 1)]:
            d = depth[i]
            vv, uu = np.nonzero(np.isfinite(d) & (d > 0.05) & (d < max_range_m))
            if not len(vv):
                continue
            z = d[vv, uu]
            pc = np.column_stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z])
            pts.append(pc[::50] @ T[i, :3, :3].T + T[i, :3, 3])
        center = np.concatenate(pts).mean(0) if pts else T[ok[0], :3, 3]
    vmap = SemanticVoxelMap(grid_around(center, half_extent_m, voxel_size_m), max_range_m=max_range_m)
    stamps = ep.get("stamps", np.arange(len(depth), dtype=float))
    for i in ok:
        masks = {c: labels[i] == lid for c, lid in LABEL_IDS.items()}
        d = depth[i].copy()
        d[~np.isfinite(d)] = np.nan
        vmap.integrate(d, masks, K, T[i], float(stamps[i]), stride)
    return vmap
