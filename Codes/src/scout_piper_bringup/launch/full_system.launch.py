"""Full Scout+Piper system bringup (Phase 0).

Composes the upstream Piper driver + MoveIt 2 config, Scout base driver,
RealSense camera, and our stem_grasp pipeline node. RViz comes up by default.

This is the ROS 2 Humble replacement for the ROS 1 launch chain:
    stem_grasp_ros1.launch  ->  study_piper.launch  ->  demo.launch

Usage:
    ros2 launch scout_piper_bringup full_system.launch.py
    ros2 launch scout_piper_bringup full_system.launch.py use_sim:=true
    ros2 launch scout_piper_bringup full_system.launch.py bringup_camera:=false

Arm safety: upstream start_single_piper.launch.py remaps the Piper driver's
command input (joint_ctrl_single) to /joint_states, so any /joint_states
publisher (joint sliders, MoveIt's joint_state_broadcaster, a test stub) moves
the real arm. This launch starts the driver node itself with that input on
``arm_command_topic`` (default /piper/joint_cmd), republishes the driver's
feedback on /joint_states as piper_joint1..8 (piper_joint_state_relay.py), and
never starts the joint sliders together with the arm.

moveit_servo (bringup_servo:=true) reaches the arm only through
piper_servo_bridge.py, which writes arm_command_topic and starts disabled:

    /servo_node/delta_{twist,joint}_cmds -> servo_node -> /piper/servo/joint_trajectory
        -> piper_servo_bridge (~/enable) -> /piper/joint_cmd -> Piper driver

fake_arm:=true replaces the driver with fake_piper_driver.py and fake_base:=true
the Scout driver with fake_scout_base.py (no hardware), e.g. to run the
whole-body MPC in execute mode end to end.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# The upstream driver executes every message on its command input as a joint
# position command at full speed; it must never be a joint-state topic.
FORBIDDEN_ARM_COMMAND_TOPICS = ("/joint_states", "/joint_states_single", "/joint_states_feedback")


def _is_true(context, name):
    # same truth values as IfCondition, which gates the arm nodes
    return LaunchConfiguration(name).perform(context).strip().lower() in ("true", "1")


def check_arm_command_topic(topic):
    """Return the command topic, or raise if it is a joint-state topic."""
    t = "/" + topic.strip().lstrip("/")
    if t in FORBIDDEN_ARM_COMMAND_TOPICS:
        raise RuntimeError(
            f"arm_command_topic:={topic} would make the Piper driver execute joint states as "
            "commands; use a dedicated topic such as /piper/joint_cmd")
    return t


def _declare_args():
    return [
        DeclareLaunchArgument(
            "use_sim",
            default_value="false",
            description="Use simulated time/hardware (no CAN, no camera).",
        ),
        DeclareLaunchArgument(
            "bringup_arm",
            default_value="false",
            description="Launch the Piper arm driver + MoveIt 2. Requires CAN0 hardware.",
        ),
        DeclareLaunchArgument(
            "bringup_base",
            default_value="false",
            description="Launch the Scout base driver. Requires CAN1 hardware.",
        ),
        DeclareLaunchArgument(
            "piper_can_port",
            default_value="can0",
            description="CAN interface of the Piper's USB-CAN adapter (INSTALL.md 9.1).",
        ),
        DeclareLaunchArgument(
            "scout_can_port",
            default_value="can1",
            description="CAN interface of the Scout's USB-CAN adapter. On a Jetson AGX Orin "
                        "the onboard CAN may already be can0/can1 (INSTALL.md Path A).",
        ),
        DeclareLaunchArgument(
            "bringup_camera",
            default_value="false",
            description="Launch the RealSense camera driver. Requires USB camera.",
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
            "bringup_scene_repr",
            default_value="false",
            description="Phase 1 semantic SDF stack (nvblox per-class). Off by "
                        "default until nvblox is installed.",
        ),
        DeclareLaunchArgument(
            "scene_classes",
            default_value="stem,branch,leaf,target,other",
            description="Classes the semantic stack maps, one nvblox process each "
                        "(e.g. stem,target,other on a 16 GB / 12 GB-GPU workstation); "
                        "'other' is the non-plant depth (pots, walls, supports).",
        ),
        DeclareLaunchArgument(
            "bringup_rviz",
            default_value="true",
            description="Open RViz with the integrated config.",
        ),
        DeclareLaunchArgument(
            "bringup_jsp_gui",
            default_value="true",
            description="Launch joint_state_publisher_gui sliders. Ignored when bringup_arm:=true "
                        "(the relay publishes the real joint states).",
        ),
        DeclareLaunchArgument(
            "fake_arm",
            default_value="false",
            description="With bringup_arm:=true, start fake_piper_driver.py instead of the "
                        "real driver (no CAN, no hardware).",
        ),
        DeclareLaunchArgument(
            "bringup_force_estimate",
            default_value="false",
            description="With bringup_arm:=true, start effort_force_node: the contact force "
                        "estimated from the Piper's joint efforts, on /ft_sensor/raw (there is "
                        "no wrist F/T sensor). Trusted only with effort_calibration.",
        ),
        DeclareLaunchArgument(
            "effort_calibration",
            default_value="",
            description="EffortModel JSON from calibrate_effort for bringup_force_estimate "
                        "(empty: uncalibrated defaults).",
        ),
        DeclareLaunchArgument(
            "fake_base",
            default_value="false",
            description="With bringup_base:=true, start fake_scout_base.py instead of the "
                        "real Scout driver (no CAN, no hardware).",
        ),
        DeclareLaunchArgument(
            "bringup_servo",
            default_value="false",
            description="Start moveit_servo (config/moveit/servo.yaml) and piper_servo_bridge "
                        "(starts disabled). Servo also needs /servo_node/start_servo.",
        ),
        DeclareLaunchArgument(
            "arm_command_topic",
            default_value="/piper/joint_cmd",
            description="Topic the Piper driver executes as joint position commands "
                        "(upstream uses /joint_states; joint-state topics are refused).",
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
    robot_description_content = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", xacro_path]),
        value_type=str,
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
    # The driver node is started here rather than through upstream
    # start_single_piper.launch.py, whose remapping makes the driver execute
    # /joint_states as commands (see the module docstring). Parameters match
    # that launch file's defaults.
    arm_on = _is_true(context, "bringup_arm")
    fake_arm = _is_true(context, "fake_arm")
    arm_command_topic = check_arm_command_topic(
        LaunchConfiguration("arm_command_topic").perform(context))
    if fake_arm:
        arm_driver = Node(
            package="scout_piper_bringup",
            executable="fake_piper_driver.py",
            name="piper_ctrl_single_node",
            output="screen",
            remappings=[("joint_ctrl_single", arm_command_topic)],
            condition=IfCondition(LaunchConfiguration("bringup_arm")),
        )
    else:
        arm_driver = Node(
            package="piper",
            executable="piper_single_ctrl",
            name="piper_ctrl_single_node",
            output="screen",
            parameters=[{
                "can_port": LaunchConfiguration("piper_can_port"),
                "auto_enable": True,
                "gripper_exist": True,
                "gripper_val_mutiple": 1,
            }],
            remappings=[("joint_ctrl_single", arm_command_topic)],
            condition=IfCondition(LaunchConfiguration("bringup_arm")),
        )

    # Driver feedback (joint1..6 + gripper on /joint_states_single) ->
    # /joint_states with the unified URDF's piper_joint1..8.
    arm_state_relay = Node(
        package="scout_piper_bringup",
        executable="piper_joint_state_relay.py",
        name="piper_joint_state_relay",
        output="screen",
        parameters=[{
            "arm_command_topic": arm_command_topic,
            "use_sim_time": use_sim,
        }],
        condition=IfCondition(LaunchConfiguration("bringup_arm")),
    )

    # Contact force from the joint efforts the relay passes on (no F/T sensor).
    force_estimate = []
    if _is_true(context, "bringup_force_estimate") and arm_on:
        force_estimate = [Node(
            package="scout_piper_whole_body_mpc",
            executable="effort_force_node",
            name="effort_force_estimator",
            output="screen",
            parameters=[system_params, {
                "calibration_file": LaunchConfiguration("effort_calibration"),
                "use_sim_time": use_sim,
            }],
            additional_env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )]
    elif _is_true(context, "bringup_force_estimate"):
        force_estimate = [LogInfo(msg="bringup_force_estimate ignored: it needs bringup_arm:=true")]

    # moveit_servo + the bridge that is its only way to the arm. Servo waits for
    # /servo_node/start_servo; the bridge waits for /piper_servo_bridge/enable.
    moveit_share = PathJoinSubstitution([bringup_share, "config", "moveit"])
    servo = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description_content,
                "robot_description_semantic": ParameterValue(
                    Command([FindExecutable(name="cat"), " ",
                             PathJoinSubstitution([moveit_share, "scout_piper.srdf"])]),
                    value_type=str,
                ),
                "use_sim_time": use_sim,
            },
            PathJoinSubstitution([moveit_share, "servo.yaml"]),
        ],
        condition=IfCondition(LaunchConfiguration("bringup_servo")),
    )
    servo_bridge = Node(
        package="scout_piper_bringup",
        executable="piper_servo_bridge.py",
        name="piper_servo_bridge",
        output="screen",
        parameters=[{
            "command_topic": arm_command_topic,
            "use_sim_time": use_sim,
        }],
        condition=IfCondition(LaunchConfiguration("bringup_servo")),
    )

    # NOTE: MoveIt 2 is intentionally NOT included here. piper_with_gripper_moveit's
    # demo.launch.py spawns its own robot_state_publisher with the standalone Piper
    # URDF, which conflicts with our unified scout_piper URDF on /robot_description.
    # Run it separately in another terminal:
    #     ros2 launch piper_with_gripper_moveit demo.launch.py
    # Phase 3 (whole-body MPC + cuMotion) replaces this with a unified planner that
    # uses our forked URDF natively.

    # ----------------------------------------------------------------------
    # 3. Scout base driver
    # ----------------------------------------------------------------------
    if _is_true(context, "fake_base"):
        base_driver = Node(
            package="scout_piper_bringup",
            executable="fake_scout_base.py",
            name="scout_base_node",
            output="screen",
            parameters=[{"use_sim_time": use_sim}],
            condition=IfCondition(LaunchConfiguration("bringup_base")),
        )
    else:
        base_driver = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("scout_base"), "launch", "scout_base.launch.py"]
                )
            ),
            condition=IfCondition(LaunchConfiguration("bringup_base")),
            launch_arguments={
                "port_name": LaunchConfiguration("scout_can_port"),
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
            # Keep topics at /camera/color/... and /camera/depth/...
            # rather than the driver default /camera/camera/... nesting.
            "camera_namespace": "/",
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
                [FindPackageShare("scout_nav2"), "launch", "nav2.launch.py"]
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
        additional_env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},   # small numpy ops
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
    # 6b. Phase 1 semantic scene representation (off by default)
    # ----------------------------------------------------------------------
    scene_repr = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("scout_piper_scene_repr"),
                 "launch", "nvblox_semantic.launch.py"]
            )
        ),
        launch_arguments={"classes": LaunchConfiguration("scene_classes")}.items(),
        condition=IfCondition(LaunchConfiguration("bringup_scene_repr")),
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

    # Joint sliders for testing the URDF without the real arm. Never together
    # with the arm: the relay owns /joint_states then.
    jsp_requested = _is_true(context, "bringup_jsp_gui")
    gui = []
    if jsp_requested and arm_on:
        gui = [LogInfo(msg="bringup_jsp_gui ignored because bringup_arm:=true "
                           "(/joint_states comes from piper_joint_state_relay)")]
    elif jsp_requested:
        gui = [Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="screen",
        )]

    return [
        robot_state_publisher,
        arm_driver,
        arm_state_relay,
        servo,
        servo_bridge,
        base_driver,
        camera,
        nav2,
        scene_repr,
        segmentation,
        pointcloud,
        pipeline,
        rviz,
    ] + force_estimate + gui


def generate_launch_description():
    return LaunchDescription(_declare_args() + [OpaqueFunction(function=_launch_setup)])
