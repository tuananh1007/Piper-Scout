"""UV texture baking, silhouette IoU / re-initialisation and OBJ export (no ROS)."""

import os

import numpy as np
import pytest

from plant_twin.leaf import LeafModel
from plant_twin.texture import ReinitMonitor, UVTexture, export_obj, silhouette, silhouette_iou

try:                                   # per-test skip (see test_leaf_torch.py)
    import cv2
except ImportError:
    cv2 = None

pytestmark = pytest.mark.skipif(cv2 is None, reason="OpenCV not installed")

K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
HW = (240, 320)


def _leaf():
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    outline = np.column_stack([0.06 * np.cos(th), 0.04 * np.sin(th)])
    hole = np.column_stack([0.02 + 0.008 * np.cos(th[::3]), 0.008 * np.sin(th[::3])])
    return LeafModel(outline, holes=[hole])


def _camera(dist=0.3):
    """Camera on the leaf's +z side looking down -z (world = R p_cam + t)."""
    R = np.diag([1.0, -1.0, -1.0])               # camera z along world -z
    return R, np.array([0.0, 0.0, dist])


def _render_leaf_image(dist=0.3):
    """Synthetic camera image: the leaf's world x < 0 half red, x >= 0 half green."""
    R, t = _camera(dist)
    v, u = np.mgrid[0:HW[0], 0:HW[1]]
    # pixel ray hits the plane z = 0 at x = (u - cx) z / fx with the camera x along world x
    x = (u - K[0, 2]) * dist / K[0, 0]
    img = np.zeros(HW + (3,), np.uint8)
    img[x < 0] = (220, 30, 30)
    img[x >= 0] = (30, 200, 30)
    depth = np.full(HW, dist, np.float32)
    return img, depth, R, t


def test_texture_bakes_the_image_into_uv():
    leaf = _leaf()
    params = leaf.initial_params()
    img, depth, R, t = _render_leaf_image()
    tex = UVTexture(leaf, size=64)
    n = tex.accumulate(params, img, K, R, t, depth=depth)
    assert n > 0.9 * len(tex.uv) and tex.coverage() > 0.9
    out = tex.image()
    assert out.shape == (64, 64, 4)
    row = out[32]                                  # middle row, v = 0.5: through the hole
    seen = row[:, 3] > 0
    left, right = row[seen & (np.arange(64) < 24)], row[seen & (np.arange(64) > 40)]
    assert (left[:, 0] > 180).all() and (left[:, 1] < 60).all()       # red on the x < 0 side
    assert (right[:, 1] > 160).all() and (right[:, 0] < 60).all()     # green on the x > 0 side
    assert out[0, 0, 3] == 0                        # corner: outside the outline
    hole_col = int(round((0.02 + 0.06) / 0.12 * 64 - 0.5))
    assert out[32, hole_col, 3] == 0                # the hole stays transparent


def test_full_size_texture():
    """texture_size 256: more texels than cv2.remap takes in one map (32767)."""
    leaf = _leaf()
    img, depth, R, t = _render_leaf_image()
    tex = UVTexture(leaf, size=256)
    assert len(tex.uv) > 32767
    assert tex.accumulate(leaf.initial_params(), img, K, R, t, depth=depth) > 0.9 * len(tex.uv)


def test_texture_skips_occluded_texels_and_blends_by_view_angle():
    leaf = _leaf()
    params = leaf.initial_params()
    img, depth, R, t = _render_leaf_image()
    tex = UVTexture(leaf, size=48)
    assert tex.accumulate(params, img, K, R, t, depth=depth - 0.05) == 0     # something 5 cm in front
    img2 = np.zeros_like(img)
    img2[...] = (0, 0, 250)                          # a blue frame seen at a grazing 70 deg
    tex.accumulate(params, img, K, R, t, depth=depth)
    a = np.deg2rad(70.0)
    Rg = R @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    tg = Rg[:, 2] * -0.3                             # camera 0.3 m back along its own axis from the leaf
    assert tex.accumulate(params, img2, K, Rg, tg, depth=None, min_cos=0.1) > 0
    out = tex.image()
    seen = out[..., 3] > 0
    # cos(70°)² ≈ 0.12 against 1 for the frontal frame: blue stays a tint (30 -> about 53)
    assert 35 < out[seen][:, 2].mean() < 60


def test_silhouette_iou_and_reinit_monitor():
    leaf = _leaf()
    params = leaf.initial_params()
    R, t = _camera()
    sil = silhouette(leaf.vertices(params), leaf.faces, K, R, t, HW)
    assert sil.sum() > 1000
    assert silhouette_iou(sil, sil.astype(np.uint8) * 255) == 1.0
    moved = params.copy()
    moved[3] += 0.05                                 # the mask now sits 5 cm off
    sil2 = silhouette(leaf.vertices(moved), leaf.faces, K, R, t, HW)
    iou = silhouette_iou(sil, sil2)
    assert iou < 0.6

    mon = ReinitMonitor(min_iou=0.6, frames=3)
    assert not mon.update(0.4, touching=False) and not mon.update(0.4, touching=False)
    assert not mon.update(0.3, touching=True)        # fingers on the leaf: not counted
    assert mon.update(0.4, touching=False)           # third low frame without contact
    assert not mon.update(0.4, touching=False)       # counts again from zero
    assert not mon.update(0.9, touching=False) and mon.low == 0


def test_export_obj(tmp_path):
    leaf = _leaf()
    params = leaf.initial_params()
    img, depth, R, t = _render_leaf_image()
    tex = UVTexture(leaf, size=32)
    tex.accumulate(params, img, K, R, t, depth=depth)
    obj = export_obj(str(tmp_path / "leaf"), leaf.vertices(params), leaf.faces, leaf.uv, tex.image())
    text = open(obj).read()
    assert text.count("\nv ") + text.startswith("v ") == len(leaf.rest_xy)
    assert text.count("\nvt ") == len(leaf.uv) and text.count("\nf ") == len(leaf.faces)
    assert os.path.exists(tmp_path / "leaf.mtl") and os.path.exists(tmp_path / "leaf_texture.png")
    png = cv2.imread(str(tmp_path / "leaf_texture.png"), cv2.IMREAD_UNCHANGED)
    assert png.shape == (32, 32, 4)
