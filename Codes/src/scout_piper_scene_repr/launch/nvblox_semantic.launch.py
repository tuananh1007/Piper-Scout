"""Phase 1 v0 launch — class_demux + one nvblox instance per plant class (P1.2.1).

Each class gets its own ``nvblox_ros/nvblox_node`` (namespace
``/scene_repr/<class>``) fed with the mask-gated depth that ``class_demux_node``
publishes on ``/scene_repr/depth/<class>``. Node setup mirrors the validated
smoke test (``realsense_nvblox.launch.py``): ``nvblox_examples_bringup``'s
``nvblox_base.yaml`` + ``config/realsense_nvblox.yaml``, ``camera_0/*``
remappings, then the per-class overrides from ``config/nvblox_per_class.yaml``.
Unlike the smoke test the map lives in ``global_frame`` (default ``odom``), so
the camera may move with the arm; the TF tree must provide it.

Assumes the RealSense driver publishes /camera/color/* and aligned depth, and
the segmentation node publishes a label image (merged) or the legacy masks
(separate).

Usage:
    ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py
    ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=separate
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CLASS_NAMES = ["stem", "branch", "leaf", "target"]

# config/nvblox_per_class.yaml key -> nvblox_ros parameter
PARAM_MAP = {
    "voxel_size_m": "voxel_size",
    "max_integration_distance_m": "static_mapper.projective_integrator_max_integration_distance_m",
    "weighting": "static_mapper.projective_integrator_weighting_mode",
}


def per_class_parameters(table: dict, class_name: str, global_frame: str) -> dict:
    """nvblox_ros parameter overrides for one class (unknown keys are an error)."""
    entry = table.get(class_name, {}) or {}
    unknown = set(entry) - set(PARAM_MAP)
    if unknown:
        raise ValueError(f"nvblox_per_class.yaml[{class_name}]: unknown keys {sorted(unknown)}")
    params = {PARAM_MAP[k]: v for k, v in entry.items()}
    params.update({
        "use_color": False,                   # depth-only per class; colour is not integrated
        "global_frame": global_frame,
        "pose_frame": global_frame,
        "map_clearing_frame_id": global_frame,
        "esdf_slice_bounds_visualization_attachment_frame_id": global_frame,
        "workspace_height_bounds_visualization_attachment_frame_id": global_frame,
    })
    return params


def _make_nvblox_node(class_name: str, overrides: dict, base_params, camera_params):
    return Node(
        package="nvblox_ros",
        executable="nvblox_node",
        name=f"nvblox_{class_name}",
        namespace=f"scene_repr/{class_name}",
        output="screen",
        parameters=[base_params, camera_params, overrides],
        remappings=[
            ("camera_0/depth/image", f"/scene_repr/depth/{class_name}"),
            ("camera_0/depth/camera_info", "/camera/aligned_depth_to_color/camera_info"),
            ("camera_0/color/image", "/camera/color/image_raw"),
            ("camera_0/color/camera_info", "/camera/color/camera_info"),
        ],
    )


def _launch_setup(context, *args, **kwargs):
    share = get_package_share_directory("scout_piper_scene_repr")
    with open(os.path.join(share, "config", "nvblox_per_class.yaml"), encoding="utf-8") as f:
        table = yaml.safe_load(f) or {}
    global_frame = LaunchConfiguration("global_frame").perform(context)

    base_params = PathJoinSubstitution(
        [FindPackageShare("nvblox_examples_bringup"), "config", "nvblox", "nvblox_base.yaml"])
    camera_params = os.path.join(share, "config", "realsense_nvblox.yaml")

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
            "policy_yaml_path": os.path.join(share, "config", "semantic_classes.yaml"),
            # Honored only in separate mode:
            "legacy_masks": [
                "/stem_grasp/mask:stem",
                "/stem_grasp/target_mask:target",
            ],
        }],
    )
    nvblox_nodes = [
        _make_nvblox_node(c, per_class_parameters(table, c, global_frame), base_params, camera_params)
        for c in CLASS_NAMES
    ]
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
            default_value="/camera/aligned_depth_to_color/image_raw",
        ),
        DeclareLaunchArgument(
            "global_frame",
            default_value="odom",
            description="World-fixed frame the per-class maps live in (must exist in TF).",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
