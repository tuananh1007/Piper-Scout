"""Phase 1 plumbing smoke test — no nvblox, no hardware.

Brings up the test_mask_publisher (synthetic mask + depth) and the
class_demux_node, then logs the per-class output topic rates. Verifies
that the Phase 1 v0 wiring works end-to-end before we add nvblox.

Usage:
    ros2 launch scout_piper_scene_repr test_class_demux.launch.py

Then in another terminal:
    ros2 topic hz /scene_repr/mask/stem            # should see ~10 Hz
    ros2 topic hz /scene_repr/mask/leaf
    ros2 topic hz /scene_repr/depth/stem
    rqt --force-discover                            # use Image View on the masks
"""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


CLASS_NAMES = ["stem", "branch", "leaf", "target"]


def generate_launch_description():
    pkg_share = FindPackageShare("scout_piper_scene_repr")
    policy_yaml = PathJoinSubstitution(
        [pkg_share, "config", "semantic_classes.yaml"]
    )

    publisher = Node(
        package="scout_piper_scene_repr",
        executable="test_mask_publisher.py",
        name="test_mask_publisher",
        output="screen",
        parameters=[{
            "image_w": 640,
            "image_h": 480,
            "publish_hz": 10.0,
            "depth_mm": 500,
        }],
    )

    demux = Node(
        package="scout_piper_scene_repr",
        executable="class_demux_node.py",
        name="scene_repr_class_demux",
        output="screen",
        parameters=[{
            "input_mode": "merged",
            "label_topic": "/stem_grasp/semantic_label",
            "depth_topic": "/camera/depth/image_rect_raw",
            "class_ids": [1, 2, 3, 4],
            "class_names": CLASS_NAMES,
            "policy_yaml_path": policy_yaml,
        }],
    )

    return LaunchDescription([publisher, demux])
