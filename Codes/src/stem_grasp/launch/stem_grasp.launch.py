"""Standalone launch for the stem_grasp pipeline (no arm/base/camera bringup).

Use this when you already have arm + camera running and want to bring up
just the perception+planning nodes for debugging.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("stem_grasp")
    default_config = PathJoinSubstitution([pkg_share, "config", "pipeline.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description="Pipeline parameter file.",
            ),
            Node(
                package="stem_grasp",
                executable="segmentation_node",
                name="stem_grasp_segmentation",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
            Node(
                package="stem_grasp",
                executable="pointcloud_node",
                name="stem_grasp_pointcloud",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
            Node(
                package="stem_grasp",
                executable="pipeline_node",
                name="stem_grasp_pipeline",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
