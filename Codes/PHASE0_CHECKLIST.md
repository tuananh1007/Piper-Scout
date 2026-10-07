# Phase 0 Migration Checklist

Sequential steps to bring the new ROS 2 workspace up to functional parity
with the existing ROS 1 stack. **Mark items done in this file AND in
[`../PROGRESS.md`](../PROGRESS.md).**

Estimated total time: **6–8 weeks** of single-engineer effort.

---

## Prerequisites

- Ubuntu 22.04 with ROS 2 Humble installed (`/opt/ros/humble`), or, on the Ubuntu 20.04 workstation,
  the dev container in [`docker/README.md`](docker/README.md), which already provides the packages below and in Step 2.
- `python3-vcstool`, `python3-colcon-common-extensions`, `python3-rosdep`.
- Two CAN interfaces (`can0` for Piper at 1 Mbit/s, `can1` for Scout at 500 kbit/s),
  brought up by `piper_ros`'s `can_config.sh` with both USB ports listed
  (INSTALL.md 9.1; `src/piper_ros/` after Step 1's `vcs import`).
- Intel RealSense SDK: ROS Humble's `ros-humble-librealsense2` (2.58.x, installed
  by rosdep) plus Intel's udev rules (INSTALL.md 9.2). The camera is a D405.

## Step 1 — Pull upstream packages

```bash
cd Codes                              # from the repository root
vcs import src < repos.yaml
./scripts/patch_upstream.sh           # post-import fixes (ugv_sdk build_type, Isaac ROS VPI 4/NITROS patches, GXF LFS pull, nvblox submodule)
ls src   # Expect: piper_ros/ scout_ros2/ ugv_sdk/ scout_nav2/ realsense-ros/
         #         isaac_ros_common/ isaac_ros_nvblox/ isaac_ros_nitros/ isaac_ros_gxf/ negotiated/
         #         plant_twin/ scout_piper_bringup/ scout_piper_description/ scout_piper_jepa/
         #         scout_piper_scene_repr/ scout_piper_whole_body_mpc/ stem_grasp/
```

**Verify:**
- [x] `piper_ros@humble` has `src/piper/` (`launch/start_single_piper.launch.py`), `src/piper_humble/`, `src/piper_description/`, `src/piper_moveit/piper_with_gripper_moveit/`, `src/piper_msgs/`, `src/piper_sim/`.
- [x] `scout_ros2` has `scout_base/`, `scout_description/`, `scout_msgs/`.
- [x] `scout_nav2` has `scout_nav2/launch/nav2.launch.py`.
- [x] `realsense-ros` (pinned to `4.58.4` since 2026-10-07; was `ros2-development`) has `realsense2_camera/launch/rs_launch.py`.

If any of these paths drift, fix the references in:
- `src/scout_piper_bringup/launch/full_system.launch.py`
- `src/scout_piper_description/urdf/scout_piper.urdf.xacro`
- `scripts/fork_piper_arm.py`

## Step 2 — Install dependencies

```bash
source /opt/ros/humble/setup.bash
sudo rosdep init      # First time only; ignore if already done
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

Likely extra apt packages on a fresh machine:

```bash
sudo apt install -y \
  ros-humble-moveit ros-humble-moveit-servo \
  ros-humble-ros2-control ros-humble-ros2-controllers \
  ros-humble-controller-manager ros-humble-joint-state-publisher-gui \
  ros-humble-rviz2 ros-humble-xacro \
  ros-humble-nav2-bringup ros-humble-tf2-tools \
  ros-humble-realsense2-camera ros-humble-realsense2-description
```

## Step 3 — First build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Expected failures and fixes:
- Upstream `piper_ros` may need its own apt deps not in rosdep — read its
  README and install them.
- If `scout_description` ships only `.urdf` and not xacro, you may need
  to wrap it manually in `scout_piper.urdf.xacro` instead of `xacro:include`.

After a clean build:

```bash
source install/setup.bash
ros2 launch scout_piper_description view_robot.launch.py
```

✅ Pass: RViz opens; you see the Scout 2.0 chassis with the Piper arm on top
and a small camera at the EE. You can drag the joint sliders to bend the arm.

❌ Fail: Most likely TF or mesh path issues. Run `tf2_tools view_frames`
to inspect the tree.

## Step 4 — Per-subsystem hardware bring-up

Bring subsystems up **one at a time** — never debug the integrated stack
when an individual driver is broken.

### 4a — Piper arm only

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=true bringup_base:=false bringup_camera:=false \
  bringup_pipeline:=false bringup_jsp_gui:=false bringup_rviz:=true
```

⚠️ Upstream `start_single_piper.launch.py` makes the driver execute every
`/joint_states` message as a joint command at full speed. `full_system.launch.py`
therefore starts the driver itself with its command input on
`arm_command_topic` (default `/piper/joint_cmd`), relays the driver's feedback
to `/joint_states` as `piper_joint1..8` (`piper_joint_state_relay.py`) and never
starts the joint sliders with the arm. Do not run the upstream launch file
together with any `/joint_states` publisher.

✅ Pass: `ros2 topic echo --once /joint_states_single` shows live `joint1..6`
from the driver, `ros2 topic echo --once /joint_states` shows the same values as
`piper_joint1..8`, and the arm in RViz follows the real one. MoveIt is not part of
`full_system.launch.py` (its demo starts its own `robot_state_publisher`, which
conflicts with the unified URDF); with the bringup's arm driver stopped, run
`ros2 launch piper_with_gripper_moveit demo.launch.py` and check that its RViz
plugin plans a motion from home to a manual pose target (planning only: with the
bringup it no longer reaches the driver, and with upstream `start_single_piper.launch.py`
it would command the real arm, starting from all joints at 0).

### 4b — Scout base only

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=false bringup_base:=true bringup_camera:=false \
  bringup_pipeline:=false
```

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.1}}' --rate 5
```

✅ Pass: Scout creeps forward at 0.1 m/s.

### 4c — RealSense only

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=false bringup_base:=false bringup_camera:=true \
  bringup_pipeline:=false bringup_rviz:=true
```

✅ Pass: RViz shows live color image and point cloud.

### 4d — All three concurrently

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=true bringup_base:=true bringup_camera:=true \
  bringup_pipeline:=false bringup_jsp_gui:=false
```

✅ Pass: All three subsystems publish their canonical topics (`/joint_states`
from the relay, the camera topics, the base topics); TF tree connects
`odom → base_link → piper_mount_link → piper_base_link → ... → camera_link`.

❌ Fail: CAN bus contention — verify Piper is on `can0` and Scout on `can1`.

## Step 5 — Hand-eye recalibration on the integrated rig

The mount of Piper on Scout changes the camera-to-arm-base transform compared
to the previous (table-mounted) setup.

Port the existing calibration scripts (in the original `piper_ros` workspace on
the lab machine; not part of this repository):
- `calibration_transform.py`
- `calibration_samples.yaml`
- `calibration_cam_pose.launch`

Place the ported versions in a new `scout_piper_calibration` package (or
inside `scout_piper_bringup/scripts/`) and re-run on the integrated rig.

Save the result into `scout_piper_description/config/hand_eye.yaml` and
plumb it into the xacro `cam_xyz`/`cam_rpy` args.

## Step 6 — Port stem_grasp algorithm bodies

The port lives in [`src/stem_grasp/`](src/stem_grasp/); the remaining
`TODO(P0.4.x)` blocks are tracked in [`../PROGRESS.md`](../PROGRESS.md) P0.4.
Port order and status:

1. ☑ **core.py** — pure Python; no ROS deps; mechanical copy (P0.4.6).
2. ☑ **moveit_planner.py** — `moveit_py` wrapper (P0.4.7); needs a `moveit_py` runtime, which has no Humble binary package.
3. ☑ **segmentation_node.py** — YOLO seg path, then Grounded-SAM (P0.4.15, P0.4.16).
4. ☑ **pointcloud_node.py** — mask-gated filtering → `/stem_grasp/filtered_cloud` + `/stem_grasp/leaf_filtered_cloud` (P0.4.17).
5. ◐ **pipeline_node.py outer loop** — skeleton + candidate selection ☑ (P0.4.10); plan→execute pending `moveit_py` (P0.4.11).
6. ☐ **pipeline_node.py iterative approach** — multi-step approach state machine (P0.4.13).
7. ☑ **pipeline_node.py inner loop** — servo step (publishes to `moveit_servo`) (P0.4.12).

After each chunk, re-run the smoke test:
```bash
cd Codes && colcon test --packages-select stem_grasp   # /workspace inside the dev container
```

## Step 7 — moveit_servo wiring (the missing link from ROS 1)

The ROS 1 stack published `/servo_server/delta_twist_cmds` with no consumer.
In ROS 2, `full_system.launch.py bringup_servo:=true` starts `moveit_servo`
and `piper_servo_bridge`, its only path to the arm (details: INSTALL.md 10.4).

1. [x] Servo config for the Piper: `scout_piper_bringup/config/moveit/servo.yaml`
   (speed units, 50 Hz, 0.25 s timeout, singularity thresholds 45 / 100 from
   the Piper Jacobian) and the unified SRDF `config/moveit/scout_piper.srdf`.
2. [x] `servo_node` + `piper_servo_bridge` in `full_system.launch.py`; the bridge
   starts disabled and clamps steps, joint limits and speed.
3. [x] Hardware-free: `bringup_arm:=true fake_arm:=true bringup_servo:=true`, then
   `ros2 run scout_piper_bringup servo_chain_check.py` → `SERVO CHAIN OK`.
4. [ ] On the robot: start servo, enable the bridge, hand-publish a 2 cm/s
   TwistStamped → EE moves; Ctrl-C → it stops; `hotkey_stop_and_zero` `x` → holds.
5. [ ] Tune `speed_percent`, `max_step_rad`, the servo scales and singularity
   thresholds on the robot.

## Step 8 — Regression test against ROS 1 baseline

Record a "scan → candidate" episode in the ROS 1 stack first:

```bash
# In the old ROS 1 stack (realsense launched with align_depth:=true; the ROS 2 stack reads aligned depth):
rosbag record -O baseline.bag /joint_states /camera/color/image_raw /camera/color/camera_info \
  /camera/aligned_depth_to_color/image_raw /stem_grasp/target_pose /static_cloud_out
```

Convert to ROS 2 bag format with [`rosbags`](https://gitlab.com/ternaris/rosbags):

```bash
pip install rosbags
rosbags-convert --src baseline.bag --dst baseline_ros2/
```

Replay through the new stack and confirm the candidate pose matches within
tolerance (2 cm / 5°).

## Step 9 — Nav2 verification

```bash
ros2 launch scout_piper_bringup full_system.launch.py bringup_base:=true bringup_nav2:=true
```

In RViz, click "2D Goal Pose" 2 m in front of the Scout. The base should
drive to it while the arm stays folded.

## Step 10 — Phase 0 sign-off

Update [`../PROGRESS.md`](../PROGRESS.md) section P0 — flip the exit
criteria to ☑ and timestamp them. Commit and tag `phase0-complete`.
Then start Phase 1 (semantic SDF).
