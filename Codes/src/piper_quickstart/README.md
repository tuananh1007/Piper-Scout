# piper_quickstart

A minimal bringup for the Piper arm with its wrist RealSense, as a starting point for data collection:

- One launch: MoveIt + RViz driving the **real** arm, the live camera image, and the raw point cloud
  shown in the MoveIt scene.
- A **preflight** check that stops the launch, with a fix-it message, if the arm's CAN bus or the camera
  isn't connected properly. A **health monitor** shuts everything down if either one stops streaming
  later.
- **Hand-eye calibration** with MoveIt Calibration (AX=XB). The result is loaded automatically on the
  next launch.
- **Episode recording** (raw MCAP bag) and **export** to a frame-aligned dataset: every color frame is
  paired with aligned depth, the arm's joint positions at that instant, the camera pose, and the camera
  intrinsics/extrinsics.

Only the Piper is needed. The Scout base, nvblox and the rest of the workspace aren't involved.

## 1. Setup (once)

On the host, from `Codes/`:

```bash
vcs import src < repos.yaml          # pulls piper_ros (+ moveit_calibration)
./scripts/patch_upstream.sh
./docker/build_dev.sh                # image includes piper_sdk, python-can, iproute2, MCAP storage, MoveIt octomap plugins
```

In the container (`docker compose -f docker/compose.dev.yml run --rm dev`):

```bash
colcon build --symlink-install --packages-up-to piper_quickstart
# For hand-eye calibration, also build the RViz plugin:
colcon build --symlink-install --packages-up-to moveit_calibration_gui
source install/setup.bash
```

## 2. Every session

**On the host**, bring up the arm's CAN bus. It doesn't survive a reboot or replugging the adapter:

```bash
bash Codes/src/piper_ros/can_activate.sh can0 1000000
```

**In the container:**

```bash
ros2 launch piper_quickstart quickstart.launch.py
```

The preflight runs first:

```
=== Piper quickstart preflight ===
Piper arm on can0:
  [ OK ] 'can0' is UP
  [ OK ] bitrate 1000000
  [ OK ] Piper is talking: 2210 frames/s including joint feedback
RealSense camera:
  [ OK ] Intel(R) RealSense(TM) Depth Camera 405 (serial 219323071475) on USB 4-2.3
  [ OK ] USB 3 link (5000 Mb/s)
Display (RViz):
  [ OK ] DISPLAY=:1
```

If any check fails, it prints `[FAIL]` with the fix, and nothing else starts. It checks for these
failures:

| Problem | Detected as |
|---|---|
| CAN adapter unplugged / `can_activate.sh` not run | `can0` missing or DOWN |
| Wrong bitrate | bitrate ≠ 1000000 |
| Arm powered off, CAN cable loose | `can0` up but silent |
| Plugged into the Scout's bus instead | traffic but no Piper joint feedback (0x2A5–0x2A7) |
| Camera unplugged / plugged in after the container started | no RealSense on USB |
| Camera on a USB 2 link | warning (failure with `require_usb3:=true`) |

After startup, the health monitor ends the launch with a `FATAL` message in these cases:

- no joint states within 12 s (the arm failed to enable)
- no camera frames within 25 s
- either stream stops (cable pulled, e-stop, USB reset)
- the arm stream keeps arriving but its data is frozen

Liveness is judged by `header.stamp` advancing, not by messages arriving. When CAN feedback stops, the
Piper driver keeps republishing its last cached values at 200 Hz with the old stamp, so a plain
heartbeat check would see a healthy stream of stale data.

The Piper driver, camera driver, bridge and `move_group` also take the whole launch down if any of them
dies.

**Moving the arm:** in RViz's MotionPlanning panel, drag the marker, then click *Plan* and *Execute*.
Use the `gripper` planning group with the `open`/`close` states to move the gripper. The robot model
always shows the measured arm pose, and the arm only moves while a trajectory is executing.

**Freeze and zero:** a small always-on-top *Piper safety* window opens next to RViz:

| Action | Window | Terminal |
|---|---|---|
| Freeze: stop whatever is executing, hold the current measured position, refuse new goals | **FREEZE** button, or Space / Esc while the window has focus | `ros2 service call /piper/freeze std_srvs/srv/Trigger` |
| Unfreeze: accept goals again | **Unfreeze**, or U | `ros2 service call /piper/unfreeze std_srvs/srv/Trigger` |
| Zero: move all six arm joints to 0, via a collision-checked MoveIt plan at 20% speed (gripper unchanged) | **Zero arm**, or Z (asks to confirm) | `ros2 service call /piper/zero std_srvs/srv/Trigger` |

Zero is refused while frozen. The current state is on `/piper/frozen` (latched `std_msgs/Bool`).
Freeze is a software hold through the driver: the motors stay powered and holding position. If the
driver, CAN link or bridge is what's misbehaving, a software freeze can't reach the arm, so use the
arm's power switch.

**Camera collision body:** the robot model ([`urdf/piper_d405.urdf.xacro`](urdf/piper_d405.urdf.xacro))
adds a `camera_body` link: a box the size of the D405 (42 × 42 × 23 mm, from `realsense2_description`)
padded by `camera_margin:=0.005` m on each side. It's fixed to `link6` at the pose given by the same
hand-eye file the camera TF uses, so the box always sits where the camera TF says the camera is.
MoveIt checks it against the scene and the arm, except `link6` and `gripper_base`, which it's bolted
to.

**Collision scene:** MoveIt plans around
- the boxes in [`config/scene_objects.yaml`](config/scene_objects.yaml) (base slab, PC monitor, no-go
  region below the floor), and
- an octomap built from the RealSense point cloud.

Both show in RViz under MotionPlanning > Scene Geometry. MoveIt filters the arm itself out of the
cloud before it reaches the octomap. Anything the camera has seen counts as an obstacle, including an
object you mean to grasp; `octomap:=false` turns it off, or clear it with
`ros2 service call /clear_octomap std_srvs/srv/Empty`.

Useful arguments: `camera:=false` (arm only), `arm:=false` (camera only), `octomap:=false`,
`scene_objects:=false`, `safety_panel:=false`, `camera_serial:=...`, `can_port:=...`, `rviz:=false`.

## 3. Hand-eye calibration (MoveIt Calibration)

1. Print an ArUco or ChArUco board (the panel's *Target* tab can generate one) and fix it to the table.
2. `ros2 launch piper_quickstart quickstart.launch.py calibrate:=true`
   This loads the HandEyeCalibration display. The camera-mount TF is left to the plugin.
3. Fill in the panel:
   - *Target* tab: board parameters, image topic `/camera/color/image_raw`, camera info
     `/camera/color/camera_info`.
   - *Context* tab: sensor configuration **Eye-in-hand**, sensor frame **camera_link**, object frame
     `handeye_target`, end-effector frame **link6**, robot base frame **base_link**.
4. *Calibrate* tab: take 12 or more samples from varied wrist orientations (move the arm with the
   MotionPlanning panel between samples), then solve. Check the residuals.
5. Click *Save camera pose* and save as `.launch.py`. Then import it:

   ```bash
   ros2 run piper_quickstart import_handeye ~/handeye.launch.py --notes "charuco 5x7, 15 samples"
   ```

   This writes `<ws>/calibration/hand_eye.local.yaml`, which is gitignored because it's specific to this
   rig. Later launches use it automatically.

Using `camera_link` as the sensor frame keeps the RealSense TF tree with a single parent. If you
calibrated against `camera_color_optical_frame` instead, that works too: the mount publisher converts it
through the driver's own camera_link→optical transform.

Until a calibration exists, the nominal mount in `config/hand_eye.nominal.yaml` is used. The launch,
the recorder and the exporter all warn when that's the case, and episodes record `calibrated: false`.

## 4. Recording episodes

With the quickstart running, open a second shell in the same container. On the host:
`docker exec -it $(docker ps -qf ancestor=piper-scout-dev:humble | head -1) bash`. Then:

```bash
ros2 run piper_quickstart record_episode --name pick_tomato --task "pick the red tomato"
# move the arm ... Ctrl-C to stop
ros2 run piper_quickstart export_episode data/episodes/<YYYYmmdd_HHMMSS>_pick_tomato
```

`record_episode` refuses to start unless joint states, color, aligned depth, both camera_infos, the
camera mount and the URDF are all live. It writes:

```
data/episodes/<stamp>_<name>/
  meta.json     intrinsics, camera mount (+ whether it's calibrated), URDF, joint names, task
  bag/          MCAP: /joint_states (measured, 200 Hz), setpoints, executed MoveIt trajectories,
                /tf, /tf_static, color, aligned depth, raw depth, camera_infos, depth->color extrinsics
```

`export_episode` writes `<episode>/export/`:

| File | Content |
|---|---|
| `color/NNNNNN.png`, `depth/NNNNNN.png` | RGB frame and aligned depth (uint16, mm) |
| `frames.csv` | per color frame: stamp, files, `joint1..7` and gripper opening interpolated to the frame stamp, camera pose `T_base_camera` and flange pose `T_base_flange` at that stamp, sync quality (`joint_gap_ms`, `depth_dt_ms`) |
| `joints.csv` | the full-rate measured trajectory (positions, velocities, efforts) |
| `commands.csv` | setpoints sent to the arm |
| `trajectories.json` | every MoveIt trajectory executed, with absolute start time |
| `camera.json` | color and depth intrinsics (K, D, P), depth units, `T_flange_camera`, depth→color extrinsics, calibration provenance |
| `episode.npz` | per-frame arrays: stamps, joints, 4×4 `T_base_camera`, `T_base_flange`, K |

**Timing:** all stamps are host wall-clock. RealSense frames use the driver's global time (the
hardware clock mapped to the host), and arm samples use the time piper_sdk received the CAN frame.
Joints are linearly interpolated between 200 Hz samples (5 ms apart). `joint_gap_ms` flags frames where
the arm stream had a gap.

## How the arm is driven

Upstream piper_ros runs MoveIt against *mock* hardware and forwards the mock's `/joint_states` to the
arm. That has two problems: RViz shows the commanded state, not the real one, and the arm is driven to
the mock's all-zero start pose as soon as both are running. Here, `moveit_bridge` sits in between:

```
piper driver --/joint_states_single (measured)--> bridge --/joint_states--> robot_state_publisher, MoveIt
MoveIt --FollowJointTrajectory (arm_controller, gripper_controller)--> bridge
bridge --/joint_ctrl_single (100 Hz setpoints while executing)--> piper driver --CAN--> arm
```

The bridge rejects a goal in these cases:

- no new arm feedback (advancing stamp) in the last 0.5 s
- another goal is already running on that controller
- the trajectory starts more than 0.15 rad from the measured pose

During execution, it aborts if new arm feedback stops arriving. It also forwards only samples with a
new stamp to `/joint_states`, so recordings contain no repeated cached values. When the trajectory ends, it reports success
only once the arm has settled within 0.03 rad of the goal. For the gripper, it reports success when the
trajectory ends, because a gripper closing on an object never reaches its target.

**Angle units:** the firmware works in millidegrees. Upstream's driver converts with 0.017444 and
1000·180/3.14 instead of π/180 and 1000·180/π, which skews every reported and commanded angle by
0.05%. Section 7 of `scripts/patch_upstream.sh` replaces both with exact π, so all joint topics are
true radians.

The gripper mapping: the driver reports the total finger opening in metres. URDF `joint7` is one finger,
so `joint7 = opening / 2`, matching `gripper_val_mutiple:=2` on the command side.

## Not covered yet

- The camera's collision box covers only the D405 body itself (plus a 5 mm margin). The mount
  bracket and the USB cable aren't modelled.
- No teleoperation or teach mode. Trajectories come from MoveIt planning only.
- No hardware-in-the-loop tests. The unit tests (`python3 -m pytest src/piper_quickstart/test`) cover
  trajectory sampling, the calibration import, transform composition and export interpolation.
