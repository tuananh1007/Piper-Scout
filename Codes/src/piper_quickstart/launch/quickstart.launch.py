"""Piper + wrist RealSense quickstart: MoveIt control, camera, point cloud, recording-ready.

    ros2 launch piper_quickstart quickstart.launch.py

Starts, after a hardware preflight passes:
  - piper_ros driver (CAN)  + piper_moveit_bridge (MoveIt <-> real arm)
  - realsense2_camera (color, aligned depth, colored point cloud)
  - camera_mount_publisher (hand-eye calibration -> TF link6 -> camera_link)
  - robot_state_publisher, move_group, RViz (MotionPlanning + camera image + point cloud)
  - piper_health_monitor (shuts everything down if the arm or camera stops streaming)
  - scene_objects (static collision boxes from config/scene_objects.yaml into MoveIt's scene)
  - arm_safety (/piper/zero) + safety_panel window (FREEZE / UNFREEZE / ZERO); freeze itself
    is served by the bridge (/piper/freeze, /piper/unfreeze)

Any driver, the bridge or the monitor exiting shuts the whole launch down.

Arguments:
  can_port:=can0         CAN interface of the arm (bring it up on the host first)
  arm:=true              false = camera + RViz only (no arm driver, no MoveIt execution)
  camera:=true           false = arm only
  camera_serial:=''      pick a specific RealSense
  rviz:=true
  octomap:=true          MoveIt builds collision voxels from the point cloud (shown in RViz
                         under MotionPlanning > Scene Geometry); false = raw cloud only
  safety_panel:=true     the FREEZE / ZERO window (needs a display)
  camera_margin:=0.005   padding [m] added to each side of the D405 collision box
  calibrate:=false       true = hand-eye calibration session (MoveIt Calibration panel in RViz;
                         the camera mount TF is left to the calibration plugin)
  hand_eye_file:=''      calibration YAML; default <ws>/calibration/hand_eye.local.yaml if it
                         exists, else the nominal (uncalibrated) mount
  scene_objects:=true     false = don't add the static collision boxes
  scene_objects_file:='' default config/scene_objects.yaml
  skip_preflight:=false
  require_usb3:=false    true = fail (not warn) when the camera is on a USB 2 link
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from piper_quickstart.robot_model import RIGID_WITH_FLANGE, add_disabled_collisions, camera_body_xacro_args
from piper_quickstart.transforms import load_camera_mount

ARGS = {
    "can_port": "can0",
    "arm": "true",
    "camera": "true",
    "camera_serial": "",
    "rviz": "true",
    "octomap": "true",
    "safety_panel": "true",
    "camera_margin": "0.005",
    "calibrate": "false",
    "hand_eye_file": "",
    "scene_objects": "true",
    "scene_objects_file": "",
    "skip_preflight": "false",
    "require_usb3": "false",
}


def _resolve_hand_eye(explicit: str, share: Path) -> str:
    if explicit:
        return explicit
    # <ws>/install/piper_quickstart/share/piper_quickstart -> <ws>
    local = share.parents[3] / "calibration" / "hand_eye.local.yaml"
    return str(local if local.exists() else share / "config" / "hand_eye.nominal.yaml")


def _launch_setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    on = lambda k: a[k].lower() in ("1", "true", "yes")  # noqa: E731
    share = Path(get_package_share_directory("piper_quickstart"))
    # on_exit takes actions, not events: wrap the Shutdown event in EmitEvent.
    shutdown = [EmitEvent(event=Shutdown(reason="a required process exited"))]

    preflight_args = ["--can-port", a["can_port"]]
    if not on("arm"):
        preflight_args.append("--skip-arm")
    if not on("camera"):
        preflight_args.append("--skip-camera")
    if not on("rviz"):
        preflight_args.append("--skip-display")
    if a["camera_serial"]:
        preflight_args += ["--camera-serial", a["camera_serial"]]
    if on("require_usb3"):
        preflight_args.append("--require-usb3")

    # ---------------------------------------------------------------- MoveIt
    # Robot model = upstream Piper + a D405 collision box placed from the same
    # hand-eye file camera_mount_publisher uses.
    hand_eye_file = _resolve_hand_eye(a["hand_eye_file"], share)
    mount = load_camera_mount(Path(hand_eye_file))
    moveit_config = (
        MoveItConfigsBuilder("piper", package_name="piper_with_gripper_moveit")
        .robot_description(file_path=str(share / "urdf" / "piper_d405.urdf.xacro"),
                           mappings=camera_body_xacro_args(mount, float(a["camera_margin"])))
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )
    srdf = moveit_config.robot_description_semantic["robot_description_semantic"]
    moveit_config.robot_description_semantic["robot_description_semantic"] = add_disabled_collisions(
        srdf, [("camera_body", link, "Adjacent") for link in RIGID_WITH_FLANGE])
    if on("octomap"):
        sensors = yaml.safe_load((share / "config" / "sensors_3d.yaml").read_text())
        sensors.update({"octomap_frame": "world", "octomap_resolution": 0.01, "max_range": 1.5})
        moveit_config.sensors_3d = sensors
    else:
        # Upstream's sensors_3d.yaml points at a nonexistent Kinect topic; disable it.
        moveit_config.sensors_3d = {}

    rest = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[moveit_config.robot_description],
            output="log",
            on_exit=shutdown,
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "publish_robot_description_semantic": True,
                    # The real arm lags the plan slightly; let the bridge's settle check decide.
                    "trajectory_execution.allowed_execution_duration_scaling": 1.5,
                    "trajectory_execution.allowed_goal_duration_margin": 2.0,
                    "trajectory_execution.allowed_start_tolerance": 0.05,
                },
            ],
            on_exit=shutdown,
        ),
    ]

    if on("scene_objects"):
        rest.append(
            Node(
                package="piper_quickstart",
                executable="scene_objects",
                output="screen",
                parameters=[{"config_file": a["scene_objects_file"] or str(share / "config" / "scene_objects.yaml")}],
                # Stays up after applying; exits non-zero if the boxes couldn't be added.
                on_exit=shutdown,
            )
        )

    # ------------------------------------------------------------------- arm
    if on("arm"):
        rest += [
            Node(
                package="piper",
                executable="piper_single_ctrl",
                name="piper_ctrl_single_node",
                output="screen",
                # Its joint command callback logs every message at INFO.
                ros_arguments=["--log-level", "piper_ctrl_single_node:=warn"],
                parameters=[{
                    "can_port": a["can_port"],
                    "auto_enable": True,
                    "gripper_exist": True,
                    "gripper_val_mutiple": 2,  # matches the bridge's gripper_scale
                }],
                on_exit=shutdown,
            ),
            Node(
                package="piper_quickstart",
                executable="moveit_bridge",
                output="screen",
                parameters=[{"gripper_scale": 2.0}],
                on_exit=shutdown,
            ),
            Node(
                package="piper_quickstart",
                executable="arm_safety",
                output="screen",
                on_exit=shutdown,
            ),
        ]
        if on("safety_panel"):
            # Not required: closing the window leaves the services usable from a terminal.
            rest.append(Node(package="piper_quickstart", executable="safety_panel", output="screen"))

    # ---------------------------------------------------------------- camera
    if on("camera"):
        rest.append(
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                name="camera",
                namespace="",
                output="screen",
                parameters=[{
                    "camera_name": "camera",
                    "serial_no": a["camera_serial"],
                    "enable_color": True,
                    "enable_depth": True,
                    # D435/D455: color from the RGB sensor. D405: color comes from the
                    # depth module, which has its own profile parameter.
                    "rgb_camera.color_profile": "640,480,30",
                    "depth_module.color_profile": "640,480,30",
                    "depth_module.depth_profile": "640,480,30",
                    "align_depth.enable": True,
                    "pointcloud.enable": True,
                    "enable_sync": True,
                    "publish_tf": True,
                    "initial_reset": False,
                }],
                on_exit=shutdown,
            )
        )
        if not on("calibrate"):
            rest.append(
                Node(
                    package="piper_quickstart",
                    executable="camera_mount_publisher",
                    output="screen",
                    parameters=[{"calibration_file": hand_eye_file}],
                    on_exit=shutdown,
                )
            )

    rest.append(
        Node(
            package="piper_quickstart",
            executable="health_monitor",
            output="screen",
            parameters=[{"monitor_arm": on("arm"), "monitor_camera": on("camera")}],
            on_exit=shutdown,
        )
    )

    # ------------------------------------------------------------------ RViz
    if on("rviz"):
        rviz_cfg = share / "rviz" / ("calibration.rviz" if on("calibrate") else "quickstart.rviz")
        rest.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(rviz_cfg)],
                output="log",
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.planning_pipelines,
                    moveit_config.joint_limits,
                ],
            )
        )

    if on("skip_preflight"):
        return [LogInfo(msg="Preflight skipped (skip_preflight:=true)."), *rest]

    preflight = Node(package="piper_quickstart", executable="preflight", arguments=preflight_args,
                     output="screen")

    def after_preflight(event, _context):
        if event.returncode == 0:
            return rest
        return [LogInfo(msg="Preflight failed: nothing else was started."),
                EmitEvent(event=Shutdown(reason="preflight failed"))]

    return [preflight, RegisterEventHandler(OnProcessExit(target_action=preflight, on_exit=after_preflight))]


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
        + [OpaqueFunction(function=_launch_setup)]
    )
