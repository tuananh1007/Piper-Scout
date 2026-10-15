"""Scene rendering, occupancy grid, stem joint estimate (no ROS)."""

import numpy as np
import pytest

from scout_piper_active_perception.grid import OccupancyGrid, corridor_mask
from scout_piper_active_perception.scene import (CLASSES, Camera, DepthSensor, backproject, make_ap_scene,
                                                 render)
from scout_piper_active_perception.stem_joint import (Cylinder, NoiseModel, StemJointEstimator, axis_error,
                                                      information, pose_covariance, view_information)


def look_at(c, t):
    z = (t - c) / np.linalg.norm(t - c)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, c
    return T


CAM = Camera(image_hw=(120, 212))


def side_view(sc, ang_deg=70.0, dist=0.3):
    side = np.cross(sc.plant.axis, [0, 0, 1.0])
    a = np.radians(ang_deg)
    return look_at(sc.target + dist * (np.cos(a) * -sc.plant.axis + np.sin(a) * side), sc.target)


def test_render_depth_and_labels():
    sc = make_ap_scene(0)
    T = side_view(sc)
    fr = render(sc, T, CAM)
    tgt = fr["label"] == CLASSES.index("target")
    assert tgt.sum() > 20
    P = backproject(fr["depth"], CAM.K, T, tgt)
    p, d, r = sc.joint_axis()
    rel = P - p
    rho = np.linalg.norm(rel - (rel @ d)[:, None] * d, axis=1)
    assert np.allclose(rho, r, atol=1e-6)                       # target pixels lie on the capsule surface
    assert np.abs(rel @ d).max() <= sc.target_half_len_m + 1e-6


def test_sensor_drops_thin_structures_far_away():
    s = DepthSensor()
    f = CAM.f
    assert s.drop_probability(0.003, 0.1, f) == 0.0              # 6 mm at 10 cm: many pixels
    assert s.drop_probability(0.003, 2.0, f) == 1.0              # at 2 m: under two pixels
    assert s.drop_probability(0.0, 2.0, f) == 0.0                # leaves, floor: no thin-structure dropout


def test_transmittance_and_voxel_gain():
    n, vs = 20, 0.01
    occ = np.zeros((n, n, n), bool)
    unk = np.zeros((n, n, n), bool)
    occ[10, :, :] = True                                        # a wall at x = 0.105
    g = OccupancyGrid(np.zeros(3), vs, occ, unk)
    assert g.transmittance([0.02, 0.1, 0.1], [0.18, 0.1, 0.1])[()] == 0.0
    assert g.transmittance([0.02, 0.1, 0.1], [0.08, 0.1, 0.1])[()] == 1.0
    unk2 = np.zeros_like(unk)
    unk2[5:8, :, :] = True
    g2 = OccupancyGrid(np.zeros(3), vs, np.zeros_like(occ), unk2)
    assert g2.transmittance([0.02, 0.1, 0.1], [0.18, 0.1, 0.1], p_unknown=0.1)[()] == pytest.approx(0.9 ** 3)
    # voxel gain: a camera at x = 0 looking along +x clears the unknown voxels in front of the wall
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]               # optical z = world x
    T[:3, 3] = [-0.1, 0.1, 0.1]
    vox = np.zeros_like(unk)
    vox[6, 9:11, 9:11] = True
    g3 = OccupancyGrid(np.zeros(3), vs, occ, vox.copy())
    assert g3.voxel_gain(T, K, (100, 100), vox, p_unknown=0.0) == pytest.approx(4.0)
    g4 = OccupancyGrid(np.zeros(3), vs, np.zeros_like(occ), vox.copy())     # nothing behind: no return
    assert g4.voxel_gain(T, K, (100, 100), vox, p_unknown=0.0, p_return_outside=0.0) == 0.0


def test_unknown_fraction_skips_the_skin_of_surfaces():
    n = 10
    occ = np.zeros((n, n, n), bool)
    occ[5] = True
    unk = np.zeros((n, n, n), bool)
    unk[4] = True                                               # the layer in front of the surface
    g = OccupancyGrid(np.zeros(3), 0.01, occ, unk)
    m = np.ones((n, n, n), bool)
    assert g.unknown_fraction(m, skip_skin=False) == pytest.approx(0.1)
    assert g.unknown_fraction(m) == 0.0
    cm = corridor_mask(g, np.array([0.0, 0.05, 0.05]), np.array([0.1, 0.05, 0.05]), 0.02)
    assert cm.sum() > 0 and not cm[:, 0, 0].any()


def _cylinder_points(cyl, cam, n=400, rng=None, sigma=0.0003):
    rng = rng or np.random.default_rng(0)
    B = np.linalg.svd(np.eye(3) - np.outer(cyl.d, cyl.d))[0][:, :2]
    phi = rng.uniform(0, 2 * np.pi, 4 * n)
    s = rng.uniform(-0.03, 0.03, 4 * n)
    nrm = np.cos(phi)[:, None] * B[:, 0] + np.sin(phi)[:, None] * B[:, 1]
    X = cyl.p + s[:, None] * cyl.d + cyl.r * nrm
    vis = ((X - cam) * nrm).sum(1) < 0
    X = X[vis][:n]
    u = (X - cam) / np.linalg.norm(X - cam, axis=1, keepdims=True)
    return X + rng.normal(0, sigma, (len(X), 1)) * u


def test_joint_fit_recovers_axis_and_sigma_shrinks_with_views():
    true = Cylinder(np.array([0.9, 0.1, 0.45]), np.array([0.3, 0.0, 0.95]) / np.linalg.norm([0.3, 0.0, 0.95]), 0.005)
    est = StemJointEstimator(true.p + [0.006, -0.004, 0.003], noise=NoiseModel(f_px=CAM.f))
    cams = [true.p + [-0.3, 0.0, 0.0], true.p + [0.0, -0.3, 0.05], true.p + [-0.2, 0.2, 0.0]]
    sig = []
    for i, c in enumerate(cams):
        est.add_view(_cylinder_points(true, c, rng=np.random.default_rng(i)), c)
        est.fit()
        sig.append(est.sigma())
    assert axis_error(est.cyl, true.p, true.d, 0.03) < 0.001
    assert abs(est.cyl.r - true.r) < 0.001
    assert sig[2] < sig[0]


def test_pose_error_caps_the_information_of_one_view():
    cyl = Cylinder(np.array([0.9, 0.0, 0.45]), np.array([0.0, 0.0, 1.0]), 0.005)
    cam = cyl.p + [-0.3, 0.0, 0.0]
    noise = NoiseModel(f_px=CAM.f)
    X = _cylinder_points(cyl, cam, n=300)
    I1 = information(cyl, X, cam, noise)
    I4 = 4 * I1                                                 # four times the points from the same pose
    P = pose_covariance(cyl, cam, noise)
    a1 = np.linalg.inv(view_information(I1, P) + 1e-6 * np.eye(5))[0, 0]
    a4 = np.linalg.inv(view_information(I4, P) + 1e-6 * np.eye(5))[0, 0]
    assert a4 > 0.5 * a1                                        # no 4x gain: the pose error remains
    assert np.sqrt(a4) >= noise.pose_m * 0.99


def test_predicted_view_matches_rendered_pixels_without_occluders():
    sc = make_ap_scene(1)
    p, d, r = sc.joint_axis()
    est = StemJointEstimator(p, noise=NoiseModel(f_px=CAM.f))
    est.cyl = Cylinder(p, d, r)
    T = side_view(sc, 80.0, 0.25)
    _, px = est.predicted_view(T, CAM.K, CAM.image_hw)
    fr = render(sc, T, CAM)
    tgt = fr["label"] == CLASSES.index("target")
    survive = 1.0 - DepthSensor().drop_probability(fr["radius"][tgt], fr["depth"][tgt], CAM.f)
    assert px == pytest.approx(survive.sum(), rel=0.25)         # expected pixels with a depth return
