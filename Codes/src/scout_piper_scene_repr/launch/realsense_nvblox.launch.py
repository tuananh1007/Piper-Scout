"""P1.1.2 smoke test: RealSense color/depth -> nvblox TSDF/ESDF.

This launch intentionally keeps the map fixed in the RealSense root frame.
That makes the first nvblox validation independent of VSLAM, odometry, Nav2,
or the Scout base. Once this produces a mesh / ESDF pointcloud, the next step
is to move the global frame to the robot/world TF tree.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scene_share = FindPackageShare("scout_piper_scene_repr")
    nvblox_examples_share = FindPackageShare("nvblox_examples_bringup")

    nvblox_base_params = PathJoinSubstitution(
        [nvblox_examples_share, "config", "nvblox", "nvblox_base.yaml"]
    )
    nvblox_realsense_params = PathJoinSubstitution(
        [scene_share, "config", "realsense_nvblox.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bringup_camera",
                default_value="true",
                description="Launch the RealSense driver as part of this smoke test.",
            ),
            DeclareLaunchArgument("camera_name", default_value="camera"),
            DeclareLaunchArgument(
                "camera_namespace",
                default_value="/",
                description=(
                    "Use '/' so RealSense publishes /camera/color/... instead "
                    "of /camera/camera/color/..."
                ),
            ),
            DeclareLaunchArgument("initial_reset", default_value="false"),
            DeclareLaunchArgument("enable_sync", default_value="false"),
            DeclareLaunchArgument(
                "depth_image_topic",
                default_value="/camera/aligned_depth_to_color/image_raw",
            ),
            DeclareLaunchArgument(
                "depth_camera_info_topic",
                default_value="/camera/aligned_depth_to_color/camera_info",
            ),
            DeclareLaunchArgument(
                "color_image_topic",
                default_value="/camera/color/image_raw",
            ),
            DeclareLaunchArgument(
                "color_camera_info_topic",
                default_value="/camera/color/camera_info",
            ),
            DeclareLaunchArgument("global_frame", default_value="camera_link"),
            DeclareLaunchArgument("log_level", default_value="info"),
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace=LaunchConfiguration("camera_namespace"),
                name=LaunchConfiguration("camera_name"),
                output="screen",
                condition=IfCondition(LaunchConfiguration("bringup_camera")),
                parameters=[{
                    "camera_name": LaunchConfiguration("camera_name"),
                    "initial_reset": ParameterValue(
                        LaunchConfiguration("initial_reset"), value_type=bool
                    ),
                    "enable_color": True,
                    "enable_depth": True,
                    "align_depth.enable": True,
                    "pointcloud.enable": False,
                    "enable_sync": ParameterValue(
                        LaunchConfiguration("enable_sync"), value_type=bool
                    ),
                    "publish_tf": True,
                }],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    LaunchConfiguration("log_level"),
                ],
            ),
            Node(
                package="nvblox_ros",
                executable="nvblox_node",
                name="nvblox_node",
                output="screen",
                parameters=[
                    nvblox_base_params,
                    nvblox_realsense_params,
                    {
                        "global_frame": LaunchConfiguration("global_frame"),
                        "pose_frame": LaunchConfiguration("global_frame"),
                        "map_clearing_frame_id": LaunchConfiguration("global_frame"),
                        "esdf_slice_bounds_visualization_attachment_frame_id":
                            LaunchConfiguration("global_frame"),
                        "workspace_height_bounds_visualization_attachment_frame_id":
                            LaunchConfiguration("global_frame"),
                    },
                ],
                remappings=[
                    (
                        "camera_0/depth/image",
                        LaunchConfiguration("depth_image_topic"),
                    ),
                    (
                        "camera_0/depth/camera_info",
                        LaunchConfiguration("depth_camera_info_topic"),
                    ),
                    (
                        "camera_0/color/image",
                        LaunchConfiguration("color_image_topic"),
                    ),
                    (
                        "camera_0/color/camera_info",
                        LaunchConfiguration("color_camera_info_topic"),
                    ),
                ],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    LaunchConfiguration("log_level"),
                ],
            ),
        ]
    )
