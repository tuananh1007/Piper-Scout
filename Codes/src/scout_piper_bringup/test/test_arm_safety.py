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
    _, _, actions = _setup(bringup_arm="true")
    assert len(_nodes(actions, "scout_piper_bringup")) == 1


@pytest.mark.parametrize("topic", ["/joint_states", "joint_states", " /joint_states_single",
                                   "/joint_states_feedback"])
def test_joint_state_topics_are_refused_as_command_topic(topic):
    launch, _, _ = _setup()
    with pytest.raises(RuntimeError):
        launch.check_arm_command_topic(topic)
    with pytest.raises(RuntimeError):
        _setup(bringup_arm="true", arm_command_topic=topic)
