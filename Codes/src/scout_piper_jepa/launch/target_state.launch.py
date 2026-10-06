from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("scout_piper_jepa"), "config", "target_memory.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        Node(package="scout_piper_jepa", executable="target_state_node",
             name="piper_jepa_target_state", output="screen",
             parameters=[LaunchConfiguration("config")]),
    ])
