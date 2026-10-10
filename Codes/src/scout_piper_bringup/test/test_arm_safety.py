"""Arm-safety checks for the bringup.

The upstream Piper driver executes whatever arrives on its command input as a
joint position command at full speed, and upstream wires that input to
/joint_states. These tests pin down that full_system.launch.py keeps the
driver's commands off every joint-state topic, never starts the joint sliders
together with the arm, and that the relay maps the driver's feedback onto the
unified URDF's joint names.

    cd Codes/src/scout_piper_bringup && python3 -m pytest test -q
(the launch tests need launch_ros and skip without it)
"""

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


relay = _load(os.path.join(PKG, "scripts", "piper_joint_state_relay.py"), "piper_joint_state_relay")

DRIVER_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


# ------------------------------------------------------------------ relay
def test_relay_renames_arm_joints_and_splits_the_gripper():
    pos = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.05]          # gripper opening 5 cm
    vel = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]                    # the driver sends 6 velocities
    names, p, v = relay.relay_joint_state(DRIVER_NAMES, pos, vel)
    assert names == [f"piper_joint{i}" for i in range(1, 9)]
    assert p[:6] == pos[:6]
    assert p[6:] == pytest.approx([0.025, -0.025])          # each finger moves half the opening
    assert v == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]
    assert len(names) == len(p) == len(v)


def test_relay_clamps_fingers_to_the_urdf_limits():
    _, p, _ = relay.relay_joint_state(DRIVER_NAMES, [0.0] * 6 + [0.09], [])
    assert p[6:] == pytest.approx([0.035, -0.035])
    _, p, _ = relay.relay_joint_state(DRIVER_NAMES, [0.0] * 6 + [-0.01], [])
    assert p[6:] == pytest.approx([0.0, 0.0])


def test_relay_handles_order_missing_velocity_and_incomplete_messages():
    names, p, v = relay.relay_joint_state(list(reversed(DRIVER_NAMES[:6])), [6, 5, 4, 3, 2, 1], [])
    assert names == [f"piper_joint{i}" for i in range(1, 7)]     # no gripper -> no fingers
    assert p == [1, 2, 3, 4, 5, 6] and v == [0.0] * 6
    assert relay.relay_joint_state(["joint1", "joint2"], [0.0, 0.0], []) is None


def test_relay_passes_arm_efforts_and_zeroes_the_fingers():
    eff = [0.1, -2.0, 1.5, 0.2, 0.3, 0.01, 4.0]             # 7 names, 7 efforts (gripper last)
    names, _, _ = relay.relay_joint_state(DRIVER_NAMES, [0.0] * 6 + [0.02], [])
    assert relay.relay_effort(DRIVER_NAMES, eff, len(names)) == eff[:6] + [0.0, 0.0]
    assert relay.relay_effort(DRIVER_NAMES, [], 8) == []              # none sent: none relayed
    assert relay.relay_effort(DRIVER_NAMES, eff[:6], 8) == []         # not one per name: ambiguous
    rev = list(reversed(DRIVER_NAMES[:6]))
    assert relay.relay_effort(rev, [6, 5, 4, 3, 2, 1], 6) == [1, 2, 3, 4, 5, 6]


def test_relay_refuses_to_publish_on_the_command_topic():
    assert relay._same_topic("/piper/joint_cmd", "piper/joint_cmd")
    assert not relay._same_topic("/joint_states", "/piper/joint_cmd")


# ----------------------------------------------------------------- launch
def _setup(**config):
    pytest.importorskip("launch_ros")
    from launch import LaunchContext  # noqa: PLC0415
    launch = _load(os.path.join(PKG, "launch", "full_system.launch.py"), "full_system_launch")
    ctx = LaunchContext()
    for arg in launch._declare_args():
        try:
            value = "".join(s.perform(ctx) for s in arg.default_value or [])
        except Exception:  # FindPackageShare defaults outside an installed workspace
            value = ""
        ctx.launch_configurations[arg.name] = value
    ctx.launch_configurations.update(config)
    return launch, ctx, launch._launch_setup(ctx)


def _nodes(actions, package):
    from launch_ros.actions import Node  # noqa: PLC0415
    return [a for a in actions if isinstance(a, Node) and a.node_package == package]


def _remaps(node, ctx):
    from launch.utilities import perform_substitutions  # noqa: PLC0415
    return {perform_substitutions(ctx, list(src)): perform_substitutions(ctx, list(dst))
            for src, dst in node._Node__remappings or []}


def test_driver_commands_are_not_on_joint_states():
    _, ctx, actions = _setup(bringup_arm="true")
    (driver,) = _nodes(actions, "piper")
    remaps = _remaps(driver, ctx)
    assert remaps == {"joint_ctrl_single": "/piper/joint_cmd"}
    assert "/joint_states" not in remaps.values()


def test_joint_sliders_never_run_with_the_arm():
    _, _, with_arm = _setup(bringup_arm="true", bringup_jsp_gui="true")
    assert _nodes(with_arm, "joint_state_publisher_gui") == []
    _, _, without_arm = _setup(bringup_arm="false", bringup_jsp_gui="true")
    assert len(_nodes(without_arm, "joint_state_publisher_gui")) == 1
    _, _, off = _setup(bringup_arm="false", bringup_jsp_gui="false")
    assert _nodes(off, "joint_state_publisher_gui") == []


def test_relay_runs_with_the_arm():
    _, ctx, actions = _setup(bringup_arm="true")
    (relay_node,) = [n for n in _nodes(actions, "scout_piper_bringup")
                     if n.node_executable == "piper_joint_state_relay.py"]
    assert relay_node.condition.evaluate(ctx)


@pytest.mark.parametrize("topic", ["/joint_states", "joint_states", " /joint_states_single",
                                   "/joint_states_feedback"])
def test_joint_state_topics_are_refused_as_command_topic(topic):
    launch, _, _ = _setup()
    with pytest.raises(RuntimeError):
        launch.check_arm_command_topic(topic)
    with pytest.raises(RuntimeError):
        _setup(bringup_arm="true", arm_command_topic=topic)


# ------------------------------------------------- servo bridge + fake driver
bridge = _load(os.path.join(PKG, "scripts", "piper_servo_bridge.py"), "piper_servo_bridge")
fake = _load(os.path.join(PKG, "scripts", "fake_piper_driver.py"), "fake_piper_driver")

SERVO_NAMES = [f"piper_joint{i}" for i in range(1, 7)]
MEASURED = {"joint1": 0.0, "joint2": 1.0, "joint3": -1.0, "joint4": 0.0, "joint5": 0.5, "joint6": 0.0}


def test_bridge_sends_all_joints_the_measured_gripper_and_a_speed():
    target = [0.01, 1.02, -1.01, 0.0, 0.49, -0.02]
    names, pos, vel = bridge.driver_command(SERVO_NAMES, target, MEASURED, 0.03, speed_percent=30)
    assert names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    assert pos[:6] == pytest.approx(target)
    assert pos[6] == 0.03                       # gripper held where it is
    assert vel == [0.0] * 6 + [30.0]            # driver reads velocity[6] as speed percent


def test_bridge_clamps_steps_and_joint_limits():
    far = [1.0, 1.0, -1.0, 0.0, 0.5, 0.0]       # joint1 jumps 1 rad
    _, pos, _ = bridge.driver_command(SERVO_NAMES, far, MEASURED, 0.0, max_step_rad=0.1)
    assert pos[0] == pytest.approx(0.1)
    low = dict(MEASURED, joint2=0.02)
    _, pos, _ = bridge.driver_command(SERVO_NAMES, [0.0, -0.5, -1.0, 0.0, 0.5, 0.0], low, 0.0)
    assert pos[1] == pytest.approx(0.0)         # URDF lower limit of joint2, not -0.08
    _, _, vel = bridge.driver_command(SERVO_NAMES, list(MEASURED.values()), MEASURED, 0.0,
                                      speed_percent=500)
    assert vel[6] == 100.0


def test_bridge_rejects_partial_or_unknown_targets():
    assert bridge.driver_command(SERVO_NAMES[:5], [0.0] * 5, MEASURED, 0.0) is None
    assert bridge.driver_command(SERVO_NAMES + ["piper_joint7"], [0.0] * 7, MEASURED, 0.0) is None
    partial = {k: v for k, v in MEASURED.items() if k != "joint6"}
    assert bridge.driver_command(SERVO_NAMES, [0.0] * 6, partial, 0.0) is None
    with pytest.raises(RuntimeError):
        bridge.check_command_topic("/joint_states")
    assert bridge.check_command_topic("piper/joint_cmd") == "/piper/joint_cmd"


def test_bridge_output_never_triggers_the_driver_hazards():
    """What the driver (as mimicked by the fake) does with the bridge's command."""
    target = [0.05, 1.05, -0.95, 0.05, 0.55, 0.05]
    cmd = bridge.driver_command(SERVO_NAMES, target, MEASURED, 0.025, speed_percent=30)
    joints, gripper, speed = fake.interpret_command(*cmd)
    assert joints == pytest.approx(target)      # no joint defaults to 0
    assert gripper == 0.025                     # gripper not closed
    assert speed == 30.0                        # not 100 %
    hold = bridge.hold_command(MEASURED, 0.025)
    assert fake.interpret_command(*hold)[0] == pytest.approx(list(MEASURED.values()))


def test_fake_driver_reproduces_the_real_driver_hazards():
    # what the sliders used to send: piper_joint* names, no velocities
    joints, gripper, speed = fake.interpret_command(SERVO_NAMES, [0.3] * 6, [])
    assert joints == [0.0] * 6 and gripper == 0.0 and speed == 100.0
    assert fake.step_toward([0.0, 1.0], [1.0, 0.0], 0.1) == pytest.approx([0.1, 0.9])


def test_bridge_gripper_commands_are_clamped_and_reach_the_driver():
    assert bridge.gripper_target(0.02, 0.07) == 0.02
    assert bridge.gripper_target(-0.01, 0.07) == 0.0
    assert bridge.gripper_target(0.5, 0.07) == 0.07
    assert bridge.gripper_target(float("nan"), 0.07) is None
    hold = bridge.hold_command(MEASURED, bridge.gripper_target(0.0, 0.07))
    joints, gripper, speed = fake.interpret_command(*hold)
    assert joints == pytest.approx(list(MEASURED.values())) and gripper == 0.0 and speed == 30.0


def test_fake_gripper_stops_at_an_object_between_the_fingers():
    w = 0.03
    for _ in range(100):
        w = fake.gripper_step(w, 0.0, 0.001, object_width=0.008)
    assert w == pytest.approx(0.008)
    assert fake.gripper_step(0.008, 0.03, 0.001, object_width=0.008) == pytest.approx(0.009)  # opens
    assert fake.gripper_step(0.005, 0.0, 0.001, object_width=0.008) == pytest.approx(0.004)  # no object inside
    assert fake.gripper_step(0.03, 0.0, 0.001) == pytest.approx(0.029)


def _effort_model():
    """scout_piper_whole_body_mpc from the install, else from the source tree."""
    import sys  # noqa: PLC0415
    src = os.path.join(PKG, "..", "scout_piper_whole_body_mpc")
    if src not in sys.path:
        sys.path.append(src)
    return pytest.importorskip("scout_piper_whole_body_mpc.dynamics.effort")


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_fake_efforts_calibrate_and_recover_an_injected_force(sign):
    """Fake driver efforts -> relay -> calibration fit -> estimator, as on the robot."""
    eff = _effort_model()
    from scout_piper_whole_body_mpc.dynamics.piper import PiperKinematics  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    kin = PiperKinematics()
    rng = np.random.default_rng(0)
    q0 = np.array([0.0, 0.8, -1.2, 0.0, 0.45, 0.0])
    Q, QD, TAU = [], [], []
    for t in np.arange(0.0, 120.0, 0.1):
        q, qd = eff.effort_excitation(q0, t, 120.0)
        tau = np.array(fake.simulated_effort(q, qd, sign=sign)) + rng.normal(0, 0.02, 6)
        names, pos, vel = relay.relay_joint_state(DRIVER_NAMES, list(q) + [0.02], list(qd))
        relayed = relay.relay_effort(DRIVER_NAMES, list(tau) + [0.0], len(names))
        Q.append(pos[:6]), QD.append(vel[:6]), TAU.append(relayed[:6])
    model = eff.fit_effort_model(kin, Q, QD, TAU)
    assert np.allclose(model.a, sign, atol=0.05)
    est = eff.ContactForceEstimator(kin, model, filter_s=0.0)
    q = np.array([0.0, 1.2, -1.0, 0.0, 0.6, 0.0])
    F = np.array([0.0, 0.0, -3.0])                           # 3 N pushing the gripper down
    est.update(q, np.zeros(6), fake.simulated_effort(q, np.zeros(6), sign=sign), 0.01)
    est.tare()
    got = est.update(q, np.zeros(6), fake.simulated_effort(q, np.zeros(6), F, sign=sign), 0.01).force
    assert np.allclose(got, F, atol=0.1), got


def test_force_estimate_starts_only_with_the_arm():
    _, ctx, actions = _setup(bringup_arm="true", fake_arm="true", bringup_force_estimate="true")
    (node,) = _nodes(actions, "scout_piper_whole_body_mpc")
    assert node.node_executable == "effort_force_node"
    _, _, off = _setup(bringup_force_estimate="true")         # no arm: no efforts to read
    assert _nodes(off, "scout_piper_whole_body_mpc") == []
    _, _, default = _setup(bringup_arm="true")
    assert _nodes(default, "scout_piper_whole_body_mpc") == []


def test_servo_and_bridge_start_together_on_the_command_topic():
    _, ctx, actions = _setup(bringup_servo="true", arm_command_topic="/piper/cmd_test")
    (servo,) = _nodes(actions, "moveit_servo")
    (bridge_node,) = [n for n in _nodes(actions, "scout_piper_bringup")
                      if n.node_executable == "piper_servo_bridge.py"]
    import yaml  # noqa: PLC0415
    from launch.utilities import perform_substitutions  # noqa: PLC0415
    params = {perform_substitutions(ctx, list(k)): v
              for k, v in bridge_node._Node__parameters[0].items()}
    assert yaml.safe_load(perform_substitutions(ctx, list(params["command_topic"]))) == "/piper/cmd_test"
    assert servo.condition.evaluate(ctx) and bridge_node.condition.evaluate(ctx)
    _, off_ctx, off = _setup()                  # default: servo off
    assert not _nodes(off, "moveit_servo")[0].condition.evaluate(off_ctx)


def test_fake_arm_replaces_the_real_driver():
    _, _, actions = _setup(bringup_arm="true", fake_arm="true")
    assert _nodes(actions, "piper") == []
    execs = [n.node_executable for n in _nodes(actions, "scout_piper_bringup")]
    assert "fake_piper_driver.py" in execs and "piper_joint_state_relay.py" in execs


# ------------------------------------------------------------------ fake base
base = _load(os.path.join(PKG, "scripts", "fake_scout_base.py"), "fake_scout_base")


def test_fake_base_integrates_a_unicycle():
    import math  # noqa: PLC0415
    x, y, yaw = base.unicycle_step(0.0, 0.0, 0.0, 1.0, 0.0, 2.0)
    assert (x, y, yaw) == pytest.approx((2.0, 0.0, 0.0))
    x, y, yaw = 0.0, 0.0, 0.0                    # quarter circle of radius 1
    for _ in range(100):
        x, y, yaw = base.unicycle_step(x, y, yaw, math.pi / 2, math.pi / 2, 0.01)
    assert (x, y, yaw) == pytest.approx((1.0, 1.0, math.pi / 2), abs=1e-6)


def test_fake_base_tracks_and_clamps_commands():
    assert base.track(0.0, 5.0, 0.02, 0.0, 1.5) == 1.5      # clamp, no lag
    v = 0.0
    for _ in range(50):                                     # 1 s at tau 0.15 s
        v = base.track(v, 1.0, 0.02, 0.15, 1.5)
    assert 0.99 < v <= 1.0


def test_fake_base_replaces_the_real_driver():
    _, ctx, actions = _setup(bringup_base="true", fake_base="true")
    execs = [n.node_executable for n in _nodes(actions, "scout_piper_bringup")]
    assert "fake_scout_base.py" in execs
    from launch.actions import IncludeLaunchDescription  # noqa: PLC0415
    _, _, real = _setup(bringup_base="true")
    assert "fake_scout_base.py" not in [n.node_executable for n in _nodes(real, "scout_piper_bringup")]
    assert any(isinstance(a, IncludeLaunchDescription) for a in real)


# ---------------------------------------------------------------- print_tcp
def test_print_tcp_puts_the_tcp_on_the_flange_z_axis():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    tcp = _load(os.path.join(PKG, "scripts", "print_tcp.py"), "print_tcp")
    # flange at (0.5, 0, 0.3), rotated 90 deg about y: its z axis points along +x
    s = np.sqrt(0.5)
    p, z = tcp.tcp_and_axis([0.5, 0.0, 0.3], [0.0, s, 0.0, s], 0.14)
    assert np.allclose(z, [1.0, 0.0, 0.0])
    assert np.allclose(p, [0.64, 0.0, 0.3])
    # base yawed 90 deg and pitched a little: the heading stays horizontal, along +y
    from scipy.spatial.transform import Rotation
    q = Rotation.from_euler("zy", [90, 10], degrees=True).as_quat()
    assert np.allclose(tcp.heading(q), [0.0, 1.0, 0.0], atol=1e-9)
