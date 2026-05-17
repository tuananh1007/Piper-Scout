"""Phase 1 v0 launch — class_demux + N parallel nvblox instances.

Brings up the per-class semantic SDF stack. Assumes the upstream
isaac_ros_nvblox is installed and `realsense2_camera` is publishing
/camera/color/* and /camera/depth/*.

Usage:
    ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py
    ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=separate
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


CLASS_NAMES = ["stem", "branch", "leaf", "target"]


def _make_nvblox_node(class_name: str):
    """One nvblox node per class, namespaced under /scene_repr/<class>.

    The node receives a mask-gated depth stream from class_demux_node, so it
    only integrates depth pixels belonging to this class. This is the Phase 1
    v0 implementation; v1 collapses these into a single multi-class fork.
    """
    return Node(
        package="isaac_ros_nvblox",
        executable="nvblox_node",
        name=f"nvblox_{class_name}",
        namespace=f"scene_repr/{class_name}",
        output="screen",
        remappings=[
            # nvblox expects /depth and /color; we feed it our class-gated depth
            ("depth/image", f"/scene_repr/depth/{class_name}"),
            ("depth/camera_info", "/camera/depth/camera_info"),
            ("color/image", "/camera/color/image_raw"),
            ("color/camera_info", "/camera/color/camera_info"),
        ],
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("scout_piper_scene_repr"),
                 "config", "nvblox_per_class.yaml"]
            ),
            # Per-class scope: nvblox reads only the {class_name}: subtree.
            {"_class_scope": class_name},
        ],
    )


def _launch_setup(context, *args, **kwargs):
    pkg_share = FindPackageShare("scout_piper_scene_repr")

    policy_yaml = PathJoinSubstitution(
        [pkg_share, "config", "semantic_classes.yaml"]
    )

    demux = Node(
        package="scout_piper_scene_repr",
        executable="class_demux_node.py",
        name="scene_repr_class_demux",
        output="screen",
        parameters=[{
            "input_mode": LaunchConfiguration("input_mode"),
            "label_topic": LaunchConfiguration("label_topic"),
            "depth_topic": LaunchConfiguration("depth_topic"),
            "class_ids": [1, 2, 3, 4],
            "class_names": CLASS_NAMES,
            "policy_yaml_path": policy_yaml,
            # Honored only in separate mode:
            "legacy_masks": [
                "/stem_grasp/mask:stem",
                "/stem_grasp/target_mask:target",
            ],
        }],
    )

    nvblox_nodes = [_make_nvblox_node(c) for c in CLASS_NAMES]

    return [demux] + nvblox_nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "input_mode",
            default_value="merged",
            description="merged | separate (legacy 2-topic stem/target mode)",
        ),
        DeclareLaunchArgument(
            "label_topic",
            default_value="/stem_grasp/semantic_label",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/depth/image_rect_raw",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
