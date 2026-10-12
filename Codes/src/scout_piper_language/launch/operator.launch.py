"""Instruction node + operator GUI with config/operator.yaml.

    ros2 launch scout_piper_language operator.launch.py [host:=0.0.0.0] [port:=8080] [gui:=true]

Run the pipeline with -p operator_target_topic:=/operator/target so it grasps
the stem the operator chose (stem_grasp README).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("scout_piper_language"), "config", "operator.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="8080"),
        DeclareLaunchArgument("gui", default_value="true"),
        Node(package="scout_piper_language", executable="instruction_node", name="operator_instruction",
             output="screen", parameters=[cfg]),
        Node(package="scout_piper_language", executable="operator_gui_node", name="operator_gui", output="screen",
             parameters=[cfg, {"host": LaunchConfiguration("host"),
                               "port": ParameterValue(LaunchConfiguration("port"), value_type=int)}],
             condition=IfCondition(LaunchConfiguration("gui"))),
    ])
