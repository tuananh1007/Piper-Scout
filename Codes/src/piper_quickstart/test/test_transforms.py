import numpy as np
import pytest

from piper_quickstart.transforms import (
    CameraMount, invert, make_transform, matrix_to_quat, parse_moveit_calibration_export, quat_to_matrix, split,
)

# Shapes copied from moveit_calibration's ControlTabWidget::generateCalibration{Python,Xml,Yaml}.
PY_EXPORT = '''""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: link6 -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "link6",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "0.0123",
                "--y",
                "-0.0456",
                "--z",
                "0.0789",
                "--qx",
                "-0.5",
                "--qy",
                "0.5",
                "--qz",
                "-0.5",
                "--qw",
                "0.5",
                # "--roll",
                # "1.57",
            ],
        ),
    ]
    return LaunchDescription(nodes)
'''

XML_EXPORT = '''<!-- Static transform publisher acquired via MoveIt 2 hand-eye calibration -->
<!-- EYE-IN-HAND: link6 -> camera_link -->
<launch>
    <node pkg="tf2_ros" exec="static_transform_publisher" output="log" args="
            --frame-id link6
            --child-frame-id camera_link
            --x 1e-3
            --y -0.02
            --z 0.05
            --qx 0
            --qy -0.7071068
            --qz 0
            --qw 0.7071068
        "
    />
</launch>
'''


def test_parse_python_export():
    d = parse_moveit_calibration_export(PY_EXPORT)
    assert d["parent_frame"] == "link6"
    assert d["child_frame"] == "camera_color_optical_frame"
    assert d["translation"] == [0.0123, -0.0456, 0.0789]
    assert d["rotation_xyzw"] == [-0.5, 0.5, -0.5, 0.5]
    assert d["mount_type"].lower() == "eye-in-hand"


def test_parse_xml_export():
    d = parse_moveit_calibration_export(XML_EXPORT)
    assert (d["parent_frame"], d["child_frame"]) == ("link6", "camera_link")
    assert d["translation"] == [0.001, -0.02, 0.05]
    assert d["rotation_xyzw"][3] == pytest.approx(0.7071068)


def test_parse_rejects_other_files():
    with pytest.raises(ValueError, match="--frame-id"):
        parse_moveit_calibration_export("print('hello')")


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[3] >= 0 else -q
        assert np.allclose(matrix_to_quat(quat_to_matrix(*q)), q, atol=1e-9)


def test_compose_mount_through_driver_frame():
    # Calibration gives link6 -> optical; driver gives camera_link -> optical.
    t_link6_opt = make_transform([0.01, -0.04, 0.08], [-0.5, 0.5, -0.5, 0.5])
    t_link_opt = make_transform([0.0, 0.015, 0.0], [-0.5, 0.5, -0.5, 0.5])
    t_link6_link = t_link6_opt @ invert(t_link_opt)
    assert np.allclose(t_link6_link @ t_link_opt, t_link6_opt)
    trans, quat = split(t_link6_link)
    assert np.allclose(make_transform(trans, quat), t_link6_link)


def test_camera_mount_dict_keeps_extra_keys():
    m = CameraMount("link6", "camera_link", [0, 0, 0.05], [0, 0, 0, 1], True, "test", extra={"imported_at": "x"})
    assert m.to_dict()["imported_at"] == "x"
    assert np.allclose(m.matrix()[:3, 3], [0, 0, 0.05])


# ---- robot_model (D405 body placement) ------------------------------------
from piper_quickstart.robot_model import (  # noqa: E402
    D405_CAMERA_LINK_TO, add_disabled_collisions, camera_body_xacro_args, camera_link_in_parent, matrix_to_rpy,
    rpy_to_matrix,
)


def test_rpy_roundtrip_including_near_gimbal_lock():
    rng = np.random.default_rng(1)
    for rpy in [*rng.uniform(-3, 3, size=(50, 3)), (0.3, 1.5707963, -0.2), (1.64639727, -1.55416564, -1.61147859)]:
        m = rpy_to_matrix(*rpy)
        assert np.allclose(rpy_to_matrix(*matrix_to_rpy(m)), m, atol=1e-6)


def test_camera_link_pose_same_from_link_or_optical_calibration():
    t_link = make_transform([-0.0858, 0.00075, 0.048], [0.0182144, -0.70731623, 0.00647473, 0.7066329])
    t_opt = t_link @ D405_CAMERA_LINK_TO["camera_color_optical_frame"]
    q_opt = split(t_opt)[1]
    a = CameraMount("link6", "camera_link", list(t_link[:3, 3]), list(split(t_link)[1]))
    b = CameraMount("link6", "camera_color_optical_frame", list(t_opt[:3, 3]), list(q_opt))
    assert np.allclose(camera_link_in_parent(a), camera_link_in_parent(b), atol=1e-9)
    args = camera_body_xacro_args(b, margin_m=0.005)
    assert args["camera_parent"] == "link6" and args["camera_body_size"] == "0.033000 0.052000 0.052000"


def test_unknown_calibration_frame_is_rejected():
    with pytest.raises(ValueError, match="not a known D405 frame"):
        camera_link_in_parent(CameraMount("link6", "some_frame", [0, 0, 0], [0, 0, 0, 1]))


def test_add_disabled_collisions():
    out = add_disabled_collisions('<robot name="piper">\n</robot>\n', [("camera_body", "link6", "Adjacent")])
    assert '<disable_collisions link1="camera_body" link2="link6" reason="Adjacent"/>' in out
    assert out.rstrip().endswith("</robot>")
