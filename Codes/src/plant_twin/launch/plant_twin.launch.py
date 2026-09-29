from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("plant_twin"), "config", "plant_twin.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        Node(
            package="plant_twin",
            executable="twin_node",
            name="plant_twin",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),
    ])
