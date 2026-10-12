"""Active perception node with config/active_perception.yaml.

    ros2 launch scout_piper_active_perception active_perception.launch.py [auto_publish:=true]
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("scout_piper_active_perception"), "config",
                       "active_perception.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("auto_publish", default_value="false",
                              description="publish a view whenever a required term is unmet"),
        DeclareLaunchArgument("policy", default_value="fused"),
        Node(package="scout_piper_active_perception", executable="active_perception_node",
             name="active_perception", output="screen",
             parameters=[cfg, {"auto_publish": ParameterValue(LaunchConfiguration("auto_publish"), value_type=bool),
                               "policy": LaunchConfiguration("policy")}]),
    ])
