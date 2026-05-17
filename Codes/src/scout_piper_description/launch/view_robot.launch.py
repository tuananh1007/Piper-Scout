"""View the unified Scout+Piper URDF in RViz.

Usage:
    ros2 launch scout_piper_description view_robot.launch.py
    ros2 launch scout_piper_description view_robot.launch.py use_gui:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("scout_piper_description")

    xacro_path = PathJoinSubstitution([pkg_share, "urdf", "scout_piper.urdf.xacro"])
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "view_robot.rviz"])

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_path,
        ]
    )

    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="true",
        description="Run joint_state_publisher_gui (sliders) instead of the headless publisher.",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description_content}],
    )

    jsp_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        condition=__import__("launch.conditions", fromlist=["IfCondition"]).IfCondition(
            LaunchConfiguration("use_gui")
        ),
    )

    jsp_headless = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        condition=__import__("launch.conditions", fromlist=["UnlessCondition"]).UnlessCondition(
            LaunchConfiguration("use_gui")
        ),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
    )

    return LaunchDescription(
        [
            use_gui_arg,
            robot_state_publisher,
            jsp_gui,
            jsp_headless,
            rviz,
        ]
    )
