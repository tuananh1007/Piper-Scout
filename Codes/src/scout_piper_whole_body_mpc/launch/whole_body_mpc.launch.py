from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

PROFILES = {"default": [], "orin": ["whole_body_mpc_orin.yaml"],
            "gpu": ["whole_body_mpc_gpu.yaml"],
            "orin_gpu": ["whole_body_mpc_gpu.yaml", "whole_body_mpc_orin_gpu.yaml"]}


def _node(context):
    profile = LaunchConfiguration("profile").perform(context)
    if profile not in PROFILES:
        raise ValueError(f"profile:={profile!r}: choose from {sorted(PROFILES)}")
    overrides = [PathJoinSubstitution([FindPackageShare("scout_piper_whole_body_mpc"), "config", f])
                 for f in PROFILES[profile]]
    # topic overrides given on the command line ("" keeps the config's value)
    topics = {k: LaunchConfiguration(k).perform(context) for k in ("field_topic", "goal_pose_topic")}
    topics = {k: v for k, v in topics.items() if v}
    # one BLAS thread: the batched solve is no faster with more (57.6 vs
    # 61.5 ms measured), and idle OpenBLAS threads spin on every core
    return [Node(package="scout_piper_whole_body_mpc", executable="whole_body_mpc_node",
                 name="whole_body_mpc", output="screen",
                 additional_env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                 parameters=[LaunchConfiguration("config"), *overrides,
                             {"execute": ParameterValue(LaunchConfiguration("execute"),
                                                        value_type=bool)}, topics])]


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("scout_piper_whole_body_mpc"), "config", "whole_body_mpc.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument(
            "execute", default_value="false",
            description="true: command /cmd_vel and /servo_node/delta_joint_cmds (see "
                        "INSTALL.md 10.13 before using it on the robot)"),
        DeclareLaunchArgument(
            "profile", default_value="default",
            description="default | orin | gpu | orin_gpu: overrides loaded after config. orin: "
                        "128 samples on the Jetson's CPU; gpu: torch backend on CUDA (workstation); "
                        "orin_gpu: torch backend on the Orin's GPU"),
        DeclareLaunchArgument(
            "field_topic", default_value="",
            description="SemanticDistanceField topic to plan around, e.g. /scene_repr/distance_field "
                        "(scene_query_node or nvblox_field_bridge.py); empty = config value"),
        DeclareLaunchArgument(
            "goal_pose_topic", default_value="",
            description="PoseStamped goal topic, e.g. /scene_repr/target_goal; empty = config value"),
        OpaqueFunction(function=_node),
    ])
