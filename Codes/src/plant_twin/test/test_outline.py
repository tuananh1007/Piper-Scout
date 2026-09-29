"""Mask → outline/holes → LeafModel, on a synthetic rendered leaf."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from plant_twin.leaf import LeafModel  # noqa: E402
from plant_twin.outline import fit_plane, mask_contours, outline_from_mask  # noqa: E402

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])


def _render_leaf(z=0.4):
    """Flat elliptical leaf with two holes at depth z, facing the camera."""
    mask = np.zeros((480, 640), np.uint8)
    cv2.ellipse(mask, (320, 240), (110, 70), 20, 0, 360, 255, -1)
    cv2.circle(mask, (350, 250), 14, 0, -1)
    cv2.circle(mask, (290, 225), 10, 0, -1)
    depth = np.full(mask.shape, np.nan, np.float32)
    depth[mask > 0] = z
    v, u = np.nonzero(mask)
    pts = np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1],
                           np.full(len(u), z)])
    return mask, depth, pts


def test_mask_contours_finds_holes():
    mask, _, _ = _render_leaf()
    outer, holes = mask_contours(mask)
    assert len(outer) >= 8
    assert len(holes) == 2


def test_fit_plane_orients_normal_toward_camera():
    _, _, pts = _render_leaf()
    plane = fit_plane(pts, toward=np.zeros(3))
    assert np.dot(plane.axes[:, 2], -plane.origin) > 0
    assert np.isclose(np.linalg.det(plane.axes), 1.0)


def test_outline_from_mask_builds_leaf_with_holes():
    mask, depth, pts = _render_leaf()
    R, t = np.eye(3), np.zeros(3)         # camera == world
    outline, holes, plane = outline_from_mask(mask, depth, K, R, t, pts)
    # ellipse semi-axes 110/70 px at 0.4 m, f=600 → 7.3 cm / 4.7 cm
    ext = outline.max(0) - outline.min(0)
    assert 0.12 < ext.max() < 0.16 and 0.08 < ext.min() < 0.11
    assert len(holes) == 2

    leaf = LeafModel(outline, holes=holes)
    params = leaf.initial_params(plane.rotvec, plane.origin)
    v = leaf.vertices(params)
    # mesh should lie on the observed surface
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(v)
    assert np.median(d) < 2e-3
    for h in holes:
        c = h.mean(0)
        assert np.linalg.norm(leaf.rest_xy - c, axis=1).min() > 0.004


def test_texture_from_cloud_and_petiole():
    mask, depth, pts = _render_leaf()
    outline, holes, plane = outline_from_mask(mask, depth, K, np.eye(3), np.zeros(3), pts)
    leaf = LeafModel(outline, holes=holes)
    params = leaf.initial_params(plane.rotvec, plane.origin)
    colors = np.tile([[0.2, 0.8, 0.3]], (len(pts), 1))
    leaf.texture_from_cloud(params, pts, colors)
    assert leaf.colors.shape == (len(leaf.rest_xy), 3)
    assert np.allclose(leaf.colors, [0.2, 0.8, 0.3])

    base_2d = plane.to_plane(np.array([[0.0, 0.08, 0.4]]))[0]  # below the leaf
    leaf.set_petiole(base_2d)
    tip = leaf.tip_point(params)
    assert tip[1] > plane.origin[1] + 0.03                    # lowest edge
