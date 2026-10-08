from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("scout_piper_whole_body_mpc"), "config", "whole_body_mpc.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument(
            "execute", default_value="false",
            description="true: command /cmd_vel and /servo_node/delta_joint_cmds (see "
                        "INSTALL.md 10.13 before using it on the robot)"),
        # one BLAS thread: the batched solve is no faster with more (57.6 vs
        # 61.5 ms measured), and idle OpenBLAS threads spin on every core
        Node(package="scout_piper_whole_body_mpc", executable="whole_body_mpc_node",
             name="whole_body_mpc", output="screen",
             additional_env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
             parameters=[LaunchConfiguration("config"),
                         {"execute": ParameterValue(LaunchConfiguration("execute"), value_type=bool)}]),
    ])
