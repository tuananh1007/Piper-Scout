"""Full Scout+Piper system bringup (Phase 0).

Composes the upstream Piper driver + MoveIt 2 config, Scout base driver,
RealSense camera, and our stem_grasp pipeline node. RViz comes up by default.

This is the ROS 2 Humble replacement for the ROS 1 launch chain:
    stem_grasp_ros1.launch  ->  study_piper.launch  ->  demo.launch

Usage:
    ros2 launch scout_piper_bringup full_system.launch.py
    ros2 launch scout_piper_bringup full_system.launch.py use_sim:=true
    ros2 launch scout_piper_bringup full_system.launch.py bringup_camera:=false
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _declare_args():
    return [
        DeclareLaunchArgument(
            "use_sim",
            default_value="false",
            description="Use simulated time/hardware (no CAN, no camera).",
        ),
        DeclareLaunchArgument(
            "bringup_arm",
            default_value="true",
            description="Launch the Piper arm driver + MoveIt 2.",
        ),
        DeclareLaunchArgument(
            "bringup_base",
            default_value="true",
            description="Launch the Scout base driver.",
        ),
        DeclareLaunchArgument(
            "bringup_camera",
            default_value="true",
            description="Launch the RealSense camera driver.",
        ),
        DeclareLaunchArgument(
            "bringup_nav2",
            default_value="false",
            description="Launch Nav2 (Phase 0: off by default; Scout drives via teleop or goal_pose).",
        ),
        DeclareLaunchArgument(
            "bringup_pipeline",
            default_value="true",
            description="Launch the stem_grasp pipeline node.",
        ),
        DeclareLaunchArgument(
            "bringup_rviz",
            default_value="true",
            description="Open RViz with the integrated config.",
        ),
        DeclareLaunchArgument(
            "system_params",
            default_value=PathJoinSubstitution(
                [FindPackageShare("scout_piper_bringup"), "config", "system.yaml"]
            ),
            description="Top-level system parameter file.",
        ),
    ]


def _launch_setup(context, *args, **kwargs):
    bringup_share = FindPackageShare("scout_piper_bringup")
    desc_share = FindPackageShare("scout_piper_description")
    use_sim = LaunchConfiguration("use_sim")
    system_params = LaunchConfiguration("system_params")

    # ----------------------------------------------------------------------
    # 1. robot_state_publisher with the unified URDF
    # ----------------------------------------------------------------------
    xacro_path = PathJoinSubstitution([desc_share, "urdf", "scout_piper.urdf.xacro"])
    robot_description_content = Command(
        [FindExecutable(name="xacro"), " ", xacro_path]
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description_content},
            {"use_sim_time": use_sim},
        ],
    )

    # ----------------------------------------------------------------------
    # 2. Piper arm: driver + ros2_control + MoveIt 2
    #
    # These launch files come from agilexrobotics/piper_ros@humble. Path
    # references below match its current layout (May 2026); verify after
    # `vcs import` and update if upstream renames.
    # ----------------------------------------------------------------------
    arm_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("piper"), "launch", "start_single_piper.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("bringup_arm")),
        launch_arguments={
            "can_interface": "can0",
            "use_sim_time": use_sim,
        }.items(),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("piper_moveit"), "launch", "demo.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("bringup_arm")),
        launch_arguments={"use_sim_time": use_sim}.items(),
    )

    # ----------------------------------------------------------------------
    # 3. Scout base driver
    # ----------------------------------------------------------------------
    base_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("scout_base"), "launch", "scout_base.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("bringup_base")),
        launch_arguments={
            "port_name": "can1",
            "use_sim_time": use_sim,
        }.items(),
    )

    # ----------------------------------------------------------------------
    # 4. RealSense camera
    # ----------------------------------------------------------------------
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("bringup_camera")),
        launch_arguments={
            "align_depth.enable": "true",
            "pointcloud.enable": "true",
            "initial_reset": "true",
            "use_sim_time": use_sim,
        }.items(),
    )

    # ----------------------------------------------------------------------
    # 5. Nav2 (off by default in Phase 0)
    # ----------------------------------------------------------------------
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("scout_nav2"), "launch", "navigation.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("bringup_nav2")),
        launch_arguments={"use_sim_time": use_sim}.items(),
    )

    # ----------------------------------------------------------------------
    # 6. stem_grasp pipeline (our ROS 2 port)
    # ----------------------------------------------------------------------
    pipeline = Node(
        package="stem_grasp",
        executable="pipeline_node",
        name="stem_grasp_pipeline",
        output="screen",
        parameters=[system_params, {"use_sim_time": use_sim}],
        condition=IfCondition(LaunchConfiguration("bringup_pipeline")),
    )

    segmentation = Node(
        package="stem_grasp",
        executable="segmentation_node",
        name="stem_grasp_segmentation",
        output="screen",
        parameters=[system_params, {"use_sim_time": use_sim}],
        condition=IfCondition(LaunchConfiguration("bringup_pipeline")),
    )

    pointcloud = Node(
        package="stem_grasp",
        executable="pointcloud_node",
        name="stem_grasp_pointcloud",
        output="screen",
        parameters=[system_params, {"use_sim_time": use_sim}],
        condition=IfCondition(LaunchConfiguration("bringup_pipeline")),
    )

    # ----------------------------------------------------------------------
    # 7. RViz
    # ----------------------------------------------------------------------
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=[
            "-d",
            PathJoinSubstitution([bringup_share, "rviz", "full_system.rviz"]),
        ],
        condition=IfCondition(LaunchConfiguration("bringup_rviz")),
        output="screen",
    )

    return [
        robot_state_publisher,
        arm_driver,
        moveit,
        base_driver,
        camera,
        nav2,
        segmentation,
        pointcloud,
        pipeline,
        rviz,
    ]


def generate_launch_description():
    return LaunchDescription(_declare_args() + [OpaqueFunction(function=_launch_setup)])
