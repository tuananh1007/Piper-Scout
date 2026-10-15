"""Image-space geometry of the stem servo (P0.4.12)."""

import numpy as np
import pytest

from stem_grasp.core import FullAdaptiveServoController, StemVelocityObserver
from stem_grasp.servo_geometry import axis_point_at_depth, desired_uv, project, stem_feature_uv

K = np.array([[380.0, 0.0, 320.0], [0.0, 380.0, 240.0], [0.0, 0.0, 1.0]])


def test_project_pinhole():
    assert project(K, np.array([0.0, 0.0, 1.0])) == pytest.approx([320.0, 240.0])
    assert project(K, np.array([0.1, -0.05, 0.5])) == pytest.approx([320 + 76.0, 240 - 38.0])
    assert project(K, np.array([0.0, 0.0, -0.1])) is None


def test_desired_uv_follows_the_gripper_axis():
    # TCP 5 cm below the optical axis (camera y down), gripper pointing along the optical axis
    tcp, axis = np.array([0.0, 0.05, 0.08]), np.array([0.0, 0.0, 1.0])
    for depth in (0.15, 0.3):
        uv = desired_uv(K, tcp, axis, depth)
        assert uv == pytest.approx([320.0, 240.0 + 380 * 0.05 / depth])   # not the image centre
    # a target on the axis projects exactly there
    target = axis_point_at_depth(tcp, axis + np.array([0.1, 0.0, 0.0]), 0.25)
    assert project(K, target) == pytest.approx(desired_uv(K, tcp, axis + np.array([0.1, 0.0, 0.0]), 0.25))
    assert desired_uv(K, tcp, np.array([1.0, 0.0, 0.0]), 0.25) is None       # axis across the view


def test_stem_feature_takes_the_stem_column_at_the_target_row():
    mask = np.zeros((480, 640), bool)
    rows = np.arange(100, 400)
    mask[rows, (300 + 0.2 * (rows - 100)).astype(int)] = True              # leaning stem
    uv = stem_feature_uv(mask, 300.0)
    assert uv == pytest.approx([340.0, 300.0], abs=1.0)
    assert stem_feature_uv(mask, 470.0) == pytest.approx([mask.nonzero()[1].mean(), mask.nonzero()[0].mean()])
    assert stem_feature_uv(np.zeros((480, 640), bool), 300.0) is None


def test_ibvs_velocity_is_a_camera_translation_toward_the_desired_pixel():
    servo = FullAdaptiveServoController(fx=380.0, fy=380.0, cx=320.0, cy=240.0)
    vel, _ = servo.step(raw_uv=[400.0, 240.0], desired_uv=[320.0, 240.0], depth_z=0.3, force_n=0.0)
    assert vel[0] > 0 and abs(vel[0]) > 5 * abs(vel[1])    # stem right of target: camera moves +x


def test_observer_velocity_uses_the_measurement_interval():
    """Masks arrive at 10-30 Hz; the pixel velocity must be per second, not per
    the observer's default 10 ms step."""
    obs = StemVelocityObserver()
    for k in range(40):
        _, vel = obs.update(np.array([100.0 + 10.0 * k, 200.0]), dt=0.1)   # 100 px/s
    assert vel == pytest.approx([100.0, 0.0], abs=2.0)


@pytest.mark.parametrize("measured, damped", [(True, False), (False, True)])
def test_sway_is_image_motion_the_camera_motion_does_not_explain(measured, damped):
    """A still stem seen from a camera moving at v drifts at J v in the image:
    no sway when the measured camera velocity is given; read as sway (gain
    damped) when the camera is claimed to be still."""
    fx, z, v = 380.0, 0.3, np.array([0.01, 0.0, 0.0])
    servo = FullAdaptiveServoController(fx=fx, fy=fx, cx=320.0, cy=240.0)
    u = 400.0
    for _ in range(30):
        u += -fx / z * v[0] * 0.1                               # image drift over 0.1 s
        _, diag = servo.step(raw_uv=[u, 240.0], desired_uv=[320.0, 240.0], depth_z=z,
                             force_n=0.0, dt=0.1, camera_vel=v if measured else np.zeros(3))
    undamped = servo._depth_compensated_gain(diag["error_norm"], z)
    assert (diag["lambda"] < 0.5 * undamped) == damped
