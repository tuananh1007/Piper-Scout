# Phase 0 Migration Checklist

Sequential steps to bring the new ROS 2 workspace up to functional parity
with the existing ROS 1 stack. **Mark items done in this file AND in
[`../PROGRESS.md`](../PROGRESS.md).**

Estimated total time: **6–8 weeks** of single-engineer effort.

---

## Prerequisites

- Ubuntu 22.04 with ROS 2 Humble installed (`/opt/ros/humble`).
- `python3-vcstool`, `python3-colcon-common-extensions`, `python3-rosdep`.
- Two CAN interfaces (`can0` for Piper, `can1` for Scout). Use the existing
  [`../../can_activate.sh`](../../can_activate.sh) as the template; create
  a Scout-specific version that brings up the second bus.
- Intel RealSense SDK 2.55+ (already present at [`../../lib/`](../../lib/)).

## Step 1 — Pull upstream packages

```bash
cd Piper_Scout_ws/Codes
vcs import src < repos.yaml
./scripts/patch_upstream.sh           # apply local-only fixes (ugv_sdk build_type)
ls src   # Expect: piper_ros/ scout_ros2/ scout_nav2/ realsense-ros/ ugv_sdk/
         #         scout_piper_bringup/ scout_piper_description/
         #         scout_piper_scene_repr/ stem_grasp/
```

**Verify:**
- [ ] `piper_ros@humble` has `piper/`, `piper_control/`, `piper_description/`, `piper_moveit/`, `piper_msgs/`, `piper_sim/`.
- [ ] `scout_ros2` has `scout_base/`, `scout_description/`, `scout_msgs/`.
- [ ] `scout_nav2` has a `launch/navigation.launch.py`.
- [ ] `realsense-ros@ros2-development` has `realsense2_camera/launch/rs_launch.py`.

If any of these paths drift, fix the references in:
- `src/scout_piper_bringup/launch/full_system.launch.py`
- `src/scout_piper_description/urdf/scout_piper.urdf.xacro`

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
  bringup_pipeline:=false bringup_rviz:=true
```

✅ Pass: `ros2 topic list` shows `/joint_states`; MoveIt's RViz plugin
plans a motion from home to a manual pose target.

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
ros2 launch scout_piper_bringup full_system.launch.py bringup_pipeline:=false
```

✅ Pass: All three subsystems publish their canonical topics; TF tree
connects `odom → base_link → piper_mount_link → piper_base_link → ... → camera_link`.

❌ Fail: CAN bus contention — verify Piper is on `can0` and Scout on `can1`.

## Step 5 — Hand-eye recalibration on the integrated rig

The mount of Piper on Scout changes the camera-to-arm-base transform compared
to the previous (table-mounted) setup.

Port the existing calibration scripts:
- [`../../calibration_transform.py`](../../calibration_transform.py)
- [`../../calibration_samples.yaml`](../../calibration_samples.yaml)
- [`../../calibration_cam_pose.launch`](../../calibration_cam_pose.launch)

Place the ported versions in a new `scout_piper_calibration` package (or
inside `scout_piper_bringup/scripts/`) and re-run on the integrated rig.

Save the result into `scout_piper_description/config/hand_eye.yaml` and
plumb it into the xacro `cam_xyz`/`cam_rpy` args.

## Step 6 — Port stem_grasp algorithm bodies

The skeleton is at [`src/stem_grasp/`](src/stem_grasp/). Each TODO block
references a line range in the ROS 1 source. Recommended order (each
~1–3 days):

1. **core.py** — pure Python; no ROS deps; mechanical copy.
2. **moveit_planner.py** — port to `moveit_py` (the new MoveIt 2 Python API).
3. **segmentation_node.py** — YOLO seg path first, then Grounded-SAM.
4. **pointcloud_node.py** — mask-gated filtering + `/static_cloud_out`.
5. **pipeline_node.py outer loop** — skeleton + candidate selection.
6. **pipeline_node.py iterative approach** — multi-step approach state machine.
7. **pipeline_node.py inner loop** — servo step (publishes to `moveit_servo`).

After each chunk, re-run the smoke test:
```bash
cd Piper_Scout_ws/Codes && colcon test --packages-select stem_grasp
```

## Step 7 — moveit_servo wiring (the missing link from ROS 1)

The ROS 1 stack published `/servo_server/delta_twist_cmds` with no consumer.
In ROS 2, we add `moveit_servo` to the bringup so the topic actually drives
the arm.

1. Copy the AgileX `piper_moveit` MoveIt 2 config's example servo YAML.
2. Tune `scale.linear/angular`, `joint_topic`, `command_in_type`,
   `singularity_threshold`, `incoming_command_timeout` for the Piper.
3. Add the `moveit_servo` Node to `full_system.launch.py`.
4. Verify: hand-publish a small TwistStamped → EE moves.

## Step 8 — Regression test against ROS 1 baseline

Record a "scan → candidate" episode in the ROS 1 stack first:

```bash
# In the old ROS 1 stack:
rosbag record -O baseline.bag /joint_states /camera/color/image_raw \
  /camera/depth/image_rect_raw /stem_grasp/target_pose /static_cloud_out
```

Convert to ROS 2 bag format with [`rosbags`](https://gitlab.com/ternaris/rosbags):

```bash
pip install rosbags
rosbags-convert baseline.bag --dst baseline_ros2/
```

Replay through the new stack and confirm the candidate pose matches within
tolerance (2 cm / 5°).

## Step 9 — Nav2 verification

```bash
ros2 launch scout_piper_bringup full_system.launch.py bringup_nav2:=true
```

In RViz, click "2D Goal Pose" 2 m in front of the Scout. The base should
drive to it while the arm stays folded.

## Step 10 — Phase 0 sign-off

Update [`../PROGRESS.md`](../PROGRESS.md) section P0 — flip the exit
criteria to ☑ and timestamp them. Commit and tag `phase0-complete`.
Then start Phase 1 (semantic SDF).
