"""Phase 2B MPPI visual servo in a simulated eye-in-hand loop (no ROS).

The camera sits on link6 as in the URDF (5 cm out, optical axis along link6 z);
the target is a point in front of the gripper, measured by projecting it into
the *true* camera each step.

    cd Codes/src/scout_piper_whole_body_mpc && python -m pytest test/test_visual_servo.py -q
"""

import numpy as np
import pytest

from scout_piper_whole_body_mpc.visual_servo import (ImageJacobianEstimator, MppiVisualServo, ServoTarget,
                                                      VisualServoConfig, VisualServoCost, camera_geometry,
                                                      image_jacobian)

K = np.array([[380.0, 0.0, 320.0], [0.0, 380.0, 240.0], [0.0, 0.0, 1.0]])
HW = (480, 640)
Q0 = np.array([0.0, 1.2, -1.0, 0.0, 0.5, 0.0])


def T_flange_cam(dx=0.0):
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # optical z = link6 z
    T[:3, 3] = [0.05 + dx, 0.0, 0.05]
    return T


def _target_in_front(servo, q, ahead=0.12, side=(0.02, -0.015)):
    """A point `ahead` metres along the gripper axis from the TCP, shifted sideways."""
    F = servo.model.kin.link_frames(q)
    tcp, R = F[-1, :3, 3], F[-1, :3, :3]
    return tcp + ahead * R[:, 2] + side[0] * R[:, 0] + side[1] * R[:, 1]


def _measure(servo, q, p, T_fc):
    uv, z, uv_d, d = camera_geometry(servo.model, np.r_[0, 0, 0, q], T_fc, K, p)
    return uv, uv_d, d


def _run(servo, q, p, steps, T_true, T_assumed=None, depth_lost_after=None, force=0.0, desired_distance=None,
         own_error=None):
    """Closed loop; errors in the true image, or (``own_error`` list) also the
    controller's own error against its desired pixel."""
    T_assumed = T_true if T_assumed is None else T_assumed
    errs, dists = [], []
    uv_d_fixed = None
    for k in range(steps):
        uv, uv_d, d = _measure(servo, q, p, T_true)
        errs.append(float(np.linalg.norm(uv - uv_d)))
        dists.append(float(d))
        lost = depth_lost_after is not None and k >= depth_lost_after
        if lost and uv_d_fixed is None:
            uv_d_fixed = uv_d
        tg = ServoTarget(uv_meas=uv, K=K, image_hw=HW, T_flange_cam=T_assumed,
                         p_base=None if lost else p, uv_desired=uv_d_fixed if lost else None,
                         force_n=force, desired_distance_m=desired_distance)
        qd, diag = servo.step(q, tg)
        if own_error is not None:
            own_error.append(diag["error_px"])
        assert np.all(np.abs(qd) <= servo.model.arm.qd_max + 1e-9)
        q = q + servo.cfg.dt * qd
    return np.array(errs), np.array(dists), q


def _servo(**kw):
    return MppiVisualServo(VisualServoConfig(samples=256, seed=0, **kw))


def test_projective_model_jacobian_matches_finite_motion():
    s = _servo()
    p = _target_in_front(s, Q0)
    J = image_jacobian(s.model, Q0, T_flange_cam(), K, p)
    dq = 0.5 * np.array([0.01, -0.005, 0.004, 0.01, -0.01, 0.02])   # small: second-order terms ~dq²
    uv0 = camera_geometry(s.model, np.r_[0, 0, 0, Q0], T_flange_cam(), K, p)[0]
    uv1 = camera_geometry(s.model, np.r_[0, 0, 0, Q0 + dq], T_flange_cam(), K, p)[0]
    assert np.allclose(uv1 - uv0, J @ dq, atol=0.3)


def test_servo_centres_the_target_on_the_gripper_axis():
    s = _servo()
    p = _target_in_front(s, Q0)
    errs, dists, _ = _run(s, Q0.copy(), p, 50, T_flange_cam())
    assert errs[0] > 40 and errs[-1] < 3.0, errs[[0, 10, 20, -1]]
    assert np.all(np.diff(errs[10:]) < 2.0)               # settles without swinging back
    assert abs(dists[-1] - dists[0]) < 0.03              # no approach asked: distance roughly held


def test_calibration_error_is_absorbed_by_the_image_bias():
    """With an 8 mm hand-eye error the desired pixel itself is off (the IBVS has
    the same limit), but the image bias keeps the prediction consistent: the
    servo still converges on its own target, without oscillating."""
    s = _servo()
    p = _target_in_front(s, Q0)
    own = []
    _run(s, Q0.copy(), p, 40, T_true=T_flange_cam(), T_assumed=T_flange_cam(dx=0.008), own_error=own)
    assert own[0] > 30 and max(own[-10:]) < 3.0, np.round(own[::5], 1)


def test_depth_loss_switches_to_the_online_image_jacobian():
    s = _servo()
    p = _target_in_front(s, Q0)
    errs, _, _ = _run(s, Q0.copy(), p, 40, T_flange_cam(), depth_lost_after=5)
    assert errs[-1] < 0.25 * errs[5], errs[[0, 5, 20, -1]]


def test_final_approach_and_the_force_penalty():
    s = _servo()
    p = _target_in_front(s, Q0, side=(0.0, 0.0))
    _, d_free, _ = _run(s, Q0.copy(), p, 20, T_flange_cam(), desired_distance=0.06)
    s2 = _servo()
    _, d_force, _ = _run(s2, Q0.copy(), p, 20, T_flange_cam(), desired_distance=0.06, force=2.0)
    assert d_free[0] - d_free[-1] > 0.015                 # advanced toward 6 cm at ~2 cm/s (1 s)
    assert d_force[-1] - d_force[0] > 0.5 * (d_free[-1] - d_free[0])   # contact force holds it back


def test_broyden_estimator_learns_a_linear_map():
    rng = np.random.default_rng(0)
    J_true = rng.normal(size=(2, 6)) * 100
    est = ImageJacobianEstimator(alpha=1.0)
    est.reset(J_true + rng.normal(size=(2, 6)) * 30)
    q = np.zeros(6)
    for _ in range(60):
        q = q + rng.normal(0, 0.01, 6)
        est.update(q, J_true @ q)
    assert np.abs(est.J - J_true).max() < 0.15 * np.abs(J_true).max()


def test_cost_rejects_a_missing_model():
    s = _servo()
    tg = ServoTarget(uv_meas=np.array([300.0, 200.0]), K=K, image_hw=HW, T_flange_cam=T_flange_cam())
    with pytest.raises(ValueError):
        VisualServoCost(s.model, tg, np.r_[0, 0, 0, Q0])


def test_torch_cost_matches_numpy():
    torch = pytest.importorskip("torch")                  # noqa: F841
    from scout_piper_whole_body_mpc.torch_backend import TorchModel
    s = _servo()
    p = _target_in_front(s, Q0)
    uv, _, _ = _measure(s, Q0, p, T_flange_cam())
    tg = ServoTarget(uv_meas=uv + 3.0, K=K, image_hw=HW, T_flange_cam=T_flange_cam(), p_base=p,
                     force_n=1.0, desired_distance_m=0.08)
    x0 = np.r_[0, 0, 0, Q0]
    rng = np.random.default_rng(0)
    U = np.clip(rng.normal(0, 0.2, (8, 20, 8)), s.model.u_low, s.model.u_high)
    U[..., :2] = 0.0
    X = s.model.rollout(x0, U)
    tm = TorchModel(s.model, dtype=torch.float64)
    for mode in ("projective", "jacobian"):
        c = VisualServoCost(s.model, tg, x0, mode=mode)
        assert np.allclose(c(X, U), c.torch_cost(tm)(tm.tensor(X), tm.tensor(U)).numpy(), rtol=1e-6)


def test_torch_backend_closes_the_loop():
    pytest.importorskip("torch")
    s = MppiVisualServo(VisualServoConfig(samples=256, seed=0, backend="torch", device="cpu"))
    p = _target_in_front(s, Q0)
    errs, _, _ = _run(s, Q0.copy(), p, 40, T_flange_cam())
    assert errs[-1] < 0.2 * errs[0], errs[[0, 10, 20, -1]]
