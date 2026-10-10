# Test procedure

Step-by-step tests for the workstation, the Jetson AGX Orin 64 GB and the
robot. Each test gives its goal, the steps, the pass criteria, what to record
and the first things to check if it fails. Background and alternatives are
in [`INSTALL.md`](INSTALL.md); this file only says what to run, in which
order, and what to write down.

The semantic scene, Piper-JEPA and whole-body MPC tasks that build on these
tests are in [`MODULE_TASKS.md`](MODULE_TASKS.md).

**Order.** Part W (workstation) → Part J (Jetson) → Part R (robot). Within a
part, run the tests in order and stop at the first failure: later tests
assume the earlier ones passed. Part R runs only after Part J has passed
completely.

**Contents:** [Recording results](#recording-results) ·
[Part W — Workstation](#part-w--workstation) ·
[Part J — Jetson AGX Orin](#part-j--jetson-agx-orin-64-gb) ·
[Part R — On the robot](#part-r--on-the-robot) ·
[Results log](#results-log) · [What to send back](#what-to-send-back)

## Recording results

1. Create one log folder per test day inside the workspace (it is ignored by
   git and, in the dev container, survives the container):

   ```bash
   cd Codes                                     # /workspace in the dev container
   export LOG=$PWD/test_logs/$(date +%Y-%m-%d)
   mkdir -p $LOG
   ```

   Run `export LOG=...` again in every new terminal.
2. Save the output of each command with `| tee $LOG/<test-id>_<name>.txt`, as
   shown in the steps.
3. Copy the [results log](#results-log) table into `$LOG/RESULTS.md` and fill
   in one row per test: date, machine, PASS / FAIL / SKIPPED, and the numbers
   each test asks for.
4. On a failure, also save the last 100 lines of the relevant terminal and a
   photo or short video if the robot did something unexpected.

---

## Part W — Workstation

Reference machine: Ryzen 7 3700X (16 threads), RTX 3060 12 GB, 16 GB RAM.
These tests use the dev container (INSTALL.md path C); on a native Ubuntu
22.04 install (path B) run the same commands on the host.

### W1 — Memory precautions

**Goal:** a memory overload ends in a killed process, not a frozen machine.

**Steps (host):**
1. `free -h | tee $LOG/W1_memory.txt` and `swapon --show | tee -a $LOG/W1_memory.txt`.
2. `sudo apt-get install -y earlyoom`
3. `systemctl is-active earlyoom | tee -a $LOG/W1_memory.txt`
4. If the machine froze before, look for the kernel's memory messages from the
   previous boot:
   `journalctl -k -b -1 | grep -iE 'out of memory|oom-kill|killed process' | tee $LOG/W1_previous_oom.txt`

**Pass:** `earlyoom` reports `active`.
**Record:** total RAM, swap size, whether the previous boot shows OOM messages.

### W2 — Build the workspace with the memory limits

**Goal:** a full build that never exhausts memory.

**Steps:**
1. Import and patch as in INSTALL.md Steps 4–5 (host, in `Codes/`):

   ```bash
   PATH=$HOME/.local/bin:$PATH vcs import src < repos.yaml
   git lfs version || ./scripts/install_git_lfs_and_pull.sh   # once: GXF prebuilts (Git LFS)
   ./scripts/patch_upstream.sh | tee $LOG/W2_patch.txt
   ```

   The first line reads `Patching for x86_64, VPI <n>` and the last
   `Patches applied. Next: ...`. Record the VPI version.
2. In a second host terminal, log the available memory every 10 s:

   ```bash
   while sleep 10; do echo "$(date +%T) $(free -m | awk '/^Mem:/ {print $7}') MB available"; done | tee $LOG/W2_memory.txt
   ```

3. Enter the container and build (INSTALL.md 6C):

   ```bash
   docker compose -f docker/compose.dev.yml run --rm dev
   # in the container:
   export LOG=/workspace/test_logs/$(date +%Y-%m-%d)
   time ./scripts/colcon_build_safe.sh --symlink-install 2>&1 | tee $LOG/W2_build.txt
   ```

4. Stop the memory log (Ctrl-C) when the build ends.

**Pass:** the build ends with `Summary: <N> packages finished` and no
`packages failed` line; the desktop stays responsive throughout.
**Record:** the `colcon_build_safe:` line (free memory, compilers, CUDA
architectures: expect `86`), the build time (`real`), the lowest value in
`W2_memory.txt`.
**If it fails:** a `Killed` / `signal 9` message means memory ran out despite
the limits: rerun with `BUILD_JOBS=2 ./scripts/colcon_build_safe.sh --symlink-install`
and record that. Any other error: save the first error block of `W2_build.txt`
and check INSTALL.md [Troubleshooting](INSTALL.md#troubleshooting).

### W3 — Unit tests

**Goal:** every package's own tests pass on this machine.

**Steps (container, in `/workspace`):**

```bash
source install/setup.bash
cd src
( cd stem_grasp                 && python3 -m pytest test -q ) 2>&1 | tee $LOG/W3_stem_grasp.txt
( cd scout_piper_bringup        && python3 -m pytest test -q ) 2>&1 | tee $LOG/W3_bringup.txt
( cd scout_piper_scene_repr     && python3 -m pytest test -q ) 2>&1 | tee $LOG/W3_scene_repr.txt
( cd plant_twin                 && python3 -m pytest test -q ) 2>&1 | tee $LOG/W3_plant_twin.txt
( cd scout_piper_jepa           && python3 -m pytest test -q ) 2>&1 | tee $LOG/W3_jepa.txt
( cd scout_piper_whole_body_mpc && PYTHONPATH=../scout_piper_scene_repr/python python3 -m pytest test -q ) 2>&1 | tee $LOG/W3_mpc.txt
cd ..
colcon test --packages-select scout_piper_scene_repr --ctest-args -R test_semantic_collision_plugin
colcon test-result --verbose 2>&1 | tee $LOG/W3_colcon_test.txt
```

If `python3 -m pytest` is missing in the container, install it first with
`sudo apt-get install -y python3-pytest` (lost when the container exits).

**Pass** (counts on 2026-10-09; more tests may have been added since):
`stem_grasp` 41, `scout_piper_bringup` 24, `scout_piper_scene_repr` 31,
`plant_twin` 22, `scout_piper_jepa` 20 (+4 skipped without torch, 24 with
torch), `scout_piper_whole_body_mpc` 28 (+7 skipped without torch, 35 with
torch) passed; no failures in `colcon test-result`.
**Record:** the last line of each file.
**If it fails:** `plant_twin` `test_jacobian.py` asserts a wall-clock
speed-up and can fail on a busy machine; rerun it once on an idle machine.
Anything else: save the failure.

### W4 — Hardware-free chain checks

**Goal:** the whole control chain works with the fake arm and base: servo,
whole-body MPC, stem grasp (reach, servo, approach, grasp, release).

**Steps (container, in `/workspace`, with `install/setup.bash` sourced):**

```bash
./scripts/hardware_free_checks.sh 2>&1 | tee $LOG/W4_hardware_free.txt
```

It runs seven checks one after another (about 10 minutes), each with its own
fake-driver bringup, on localhost only, and writes its logs to
`test_logs/hwfree_<date>_<time>/`.

**Pass:** the summary lists seven `PASS` lines and the script exits with 0.
**Record:** the summary, and the `info: ... MPPI solve median ... ms, max ... ms`
lines of the two MPC checks. Note any `MPPI solves overrun` warning.
**If it fails:** open `check.out` and `pipeline.log` / `mpc.log` of the
failing check in the log folder; rerun that check once to see whether it
repeats.

### W5 — Memory of the GPU components (needs only the RealSense)

**Goal:** measure what the GPU parts cost on 16 GB (not measurable without a
GPU, so still open).

**Steps:** plug the D405 into the workstation.
1. Baseline: `free -m | tee $LOG/W5_baseline.txt; nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a $LOG/W5_baseline.txt`
2. Start the camera with nvblox (container): `ros2 launch scout_piper_scene_repr realsense_nvblox.launch.py`
3. After 30 s, in another terminal:

   ```bash
   free -m | tee $LOG/W5_nvblox.txt
   nvidia-smi --query-gpu=memory.used --format=csv | tee -a $LOG/W5_nvblox.txt
   ps -eo rss,comm --sort=-rss | head -8 | tee -a $LOG/W5_nvblox.txt
   ```

4. Add the segmentation node with `mask_mode:=hsv_green` (no model needed),
   or YOLO if the weights are available, and repeat step 3 into `W5_segmentation.txt`:
   `ros2 run stem_grasp segmentation_node --ros-args -p mask_mode:=hsv_green`
5. Stop everything with Ctrl-C.

**Pass:** nothing freezes; the numbers are recorded.
**Record:** RAM available and GPU memory used for baseline, nvblox, nvblox +
segmentation; the five largest processes.

---

## Part J — Jetson AGX Orin 64 GB

Background: INSTALL.md [Path A](INSTALL.md#path-a-jetson-agx-orin-64-gb-on-the-robot).
Nothing in this part moves the robot; leave the robot's USB-CAN adapters
unplugged until Part R.

### J1 — JetPack and system check

**Goal:** the Orin runs the software versions the workspace is built for.

**Steps:**
1. Install JetPack **6.1 or 6.2** with NVIDIA SDK Manager, on the NVMe SSD if
   one is fitted. Record which one.
2. On the Orin:

   ```bash
   sudo apt-get update && sudo apt-get install -y nvidia-jetpack earlyoom
   sudo nvpmodel -m 0 && sudo jetson_clocks
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
   git clone https://github.com/tuananh1007/Piper-Scout.git && cd Piper-Scout/Codes
   export LOG=$PWD/test_logs/$(date +%Y-%m-%d) && mkdir -p $LOG
   ./scripts/check_jetson.sh | tee $LOG/J1_check_jetson.txt
   ```

**Pass:** OK lines for `L4T: # R36 (release), REVISION: 4.x`, `CUDA 12.6`,
`VPI: /opt/nvidia/vpi3` and a power mode containing `MAXN`. FAIL lines are
expected only for `/opt/ros/humble missing` (installed in J2); a FAIL for
`gs_usb` does not stop Part J but must be solved before R1 (INSTALL.md A.5).
**Record:** the whole `J1_check_jetson.txt` (it answers INSTALL.md's open
JetPack / L4T question), the storage device used, free disk space.
**If it fails:** an L4T revision below 4 means JetPack older than 6.1:
reflash. CUDA not found: reopen the terminal after the `PATH` line.

### J2 — ROS 2 Humble and dependencies

**Steps:**
1. Install ROS 2 Humble and the apt packages exactly as INSTALL.md 3B.1–3B.2
   (the commands are the same on arm64).
2. Install the build libraries and `magic_enum` (INSTALL.md 3B.4, without its
   CUDA and VPI lines):

   ```bash
   sudo apt-get install -y --no-install-recommends \
     libgflags-dev libgoogle-glog-dev libsqlite3-dev libbenchmark-dev libgtest-dev libgmock-dev
   ```

   then the `magic_enum` block of 3B.4.
3. Install the Python packages of 3B.3 **without** `ultralytics` (it would
   pull a torch without GPU support; it comes in J3):

   ```bash
   python3 -m pip install --user 'numpy<2' scipy scikit-image scikit-learn networkx open3d \
     'transforms3d>=0.4' piper_sdk python-can 2>&1 | tee $LOG/J2_pip.txt
   ```

4. `./scripts/check_jetson.sh | tee $LOG/J2_check_jetson.txt`

**Pass:** check_jetson.sh shows `ROS 2 Humble installed` and `numpy 1.x`.
**Record:** any package apt or pip could not install.

### J3 — PyTorch with the Jetson GPU, then YOLO

**Steps:**
1. Install NVIDIA's PyTorch wheel for JetPack 6.1 / Python 3.10 (NVIDIA's
   *Installing PyTorch for Jetson Platform* guide) and a matching
   `torchvision` (the Ultralytics *NVIDIA Jetson* guide links matching wheels).
2. Check before going on:

   ```bash
   python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" | tee $LOG/J3_torch.txt
   ```

3. Only then: `python3 -m pip install --user ultralytics`, and repeat step 2
   (`tee -a`) to confirm pip kept the Jetson torch.

**Pass:** both checks print `True`.
**Record:** torch and torchvision versions, the wheel source used.
**If it fails:** `False` after step 3 means pip replaced torch: uninstall
`torch torchvision` and repeat steps 1–3.

### J4 — Import, patch and build the workspace

**Steps (on the Orin, in `Codes/`):**

```bash
source /opt/ros/humble/setup.bash
vcs import src < repos.yaml
git lfs version || ./scripts/install_git_lfs_and_pull.sh   # once: GXF prebuilts (Git LFS)
./scripts/patch_upstream.sh | tee $LOG/J4_patch.txt
sudo rosdep init; rosdep update                        # "already initialized" is fine
rosdep install --from-paths src --ignore-src -r -y 2>&1 | tee $LOG/J4_rosdep.txt
time ./scripts/colcon_build_safe.sh --symlink-install 2>&1 | tee $LOG/J4_build.txt
```

**Pass:** `J4_patch.txt` starts with `Patching for aarch64, VPI 3` and has no
`VPI_BACKEND_NVENC` or `pBase` lines (those are the VPI 4 patches); the `colcon_build_safe:` line shows `CUDA
architectures: 87`; the build ends with `Summary: <N> packages finished` and
no `packages failed` line.
**Record:** the `colcon_build_safe:` line, the build time, the rosdep keys it
could not resolve (only `nvblox_examples_bringup` dependencies are expected).
**If it fails:** save the first error block. Errors in `isaac_ros_nitros`
mentioning `pBase`, `offsetBytes` or `NVENC` mean the VPI patch selection is
wrong: rerun `VPI_MAJOR=3 ./scripts/patch_upstream.sh` and build again.

### J5 — Unit tests on the Orin

**Steps:** as W3, with `source install/setup.bash` in `Codes/`, saving to
`$LOG/J5_<package>.txt`.
**Pass:** the same counts as W3; with torch, `scout_piper_jepa` 24 and
`scout_piper_whole_body_mpc` 35 passed.
**Record:** the last line of each file and the MPC suite's run time (it is
CPU-bound and shows how much slower the Orin is).

### J6 — Hardware-free chain checks and MPC timing

**Goal:** the control chain works on the Orin's CPU, and the MPC keeps its
100 ms period.

**Steps:**

```bash
source install/setup.bash
./scripts/hardware_free_checks.sh --profile orin 2>&1 | tee $LOG/J6_orin_profile.txt
./scripts/hardware_free_checks.sh --quick 2>&1 | tee $LOG/J6_default_profile.txt
```

Watch `tegrastats` in another terminal during the first run and save a few
lines: `tegrastats --interval 2000 | tee $LOG/J6_tegrastats.txt` (Ctrl-C after
the run).

**Pass:** seven `PASS` lines with `--profile orin`; no `MPPI solves overrun`
warning with `--profile orin`.
**Record:** the `MPPI solve median / max` lines of both runs, any overrun
warnings, the highest RAM use in `tegrastats`.
**If it fails:** overrun warnings or a median above 70 ms with
`--profile orin`: lower `samples` in
`src/scout_piper_whole_body_mpc/config/whole_body_mpc_orin.yaml` to 96, rebuild
the package (`./scripts/colcon_build_safe.sh --symlink-install --packages-select scout_piper_whole_body_mpc`)
and repeat; record the value that passes. A failing grasp check: see W4.

### J7 — Workstation as the operator station

**Goal:** RViz on the workstation shows what runs on the Orin.

**Steps:**
1. Connect both machines to the same wired network. On both (the dev container
   already uses host networking):

   ```bash
   sudo apt-get install -y chrony            # host of each machine
   export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
   ```

   Use the same `RMW_IMPLEMENTATION` on both, or leave it unset on both.
2. On the Orin:

   ```bash
   ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true \
     bringup_servo:=true bringup_rviz:=false bringup_pipeline:=false
   ```

3. On the workstation (container):

   ```bash
   ros2 topic hz /joint_states | tee $LOG/J7_hz.txt                 # Ctrl-C after 10 s
   rviz2 -d $(ros2 pkg prefix scout_piper_bringup)/share/scout_piper_bringup/rviz/full_system.rviz
   ```

4. On both hosts: `chronyc tracking | tee $LOG/J7_chrony_<machine>.txt`.

**Pass:** `/joint_states` arrives on the workstation at the same rate as on
the Orin (`ros2 topic hz /joint_states` there); RViz shows the robot model;
`System time` in `chronyc tracking` is within 5 ms on both.
**If it fails:** no topics: check `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY=0`, the
same RMW, and that no firewall blocks UDP between the machines.

### J8 — RealSense on the Orin

**Steps:** plug the D405 into the Orin.

```bash
ros2 launch scout_piper_bringup full_system.launch.py bringup_camera:=true \
  bringup_rviz:=false bringup_pipeline:=false bringup_jsp_gui:=false
ros2 topic hz /camera/color/image_raw | tee $LOG/J8_color_hz.txt                       # 10 s
ros2 topic hz /camera/aligned_depth_to_color/image_raw | tee $LOG/J8_depth_hz.txt      # 10 s
```

**Pass:** both at about 30 Hz.
**If it fails:** camera not found: see INSTALL.md A.5 (RSUSB backend).

### J9 — GPU stack on the Orin

**Goal:** confirm that the perception stack fits and record its cost. The
per-class semantic nvblox (step 4) has never run anywhere, so this is also its
first run.

**Steps:** stop J8's launch first (the next launch starts its own camera).
Record the idle state: `tegrastats --interval 2000 | head -3 | tee $LOG/J9_idle.txt`.
Then start each command in its own terminal, in this order:
1. Camera and the single nvblox map (validated on the workstation):
   `ros2 launch scout_piper_scene_repr realsense_nvblox.launch.py`
2. Segmentation, with a green object or plant in view:
   `ros2 run stem_grasp segmentation_node --ros-args -p mask_mode:=hsv_green`
   (or YOLO with its weights: `-p mask_mode:=yolo_seg -p yolo_model_path:=<file>`).
3. A fixed `odom` frame at the camera (the semantic maps live in `odom`; on
   the robot the base provides it):
   `ros2 run tf2_ros static_transform_publisher --frame-id odom --child-frame-id camera_link`
4. Per-class semantic nvblox:
   `ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=separate 2>&1 | tee $LOG/J9_semantic.txt`
5. After 60 s:

   ```bash
   tegrastats --interval 2000 | head -5 | tee $LOG/J9_tegrastats.txt
   ps -eo rss,comm --sort=-rss | head -10 | tee $LOG/J9_processes.txt
   ros2 topic list | grep scene_repr | tee $LOG/J9_topics.txt
   ros2 topic hz /scene_repr/mask/stem | tee $LOG/J9_mask_hz.txt          # 10 s
   ```

**Pass:** all nodes keep running; RAM use (the `RAM` field of `tegrastats`)
stays below 48 GB; `/scene_repr/mask/stem` publishes.
**Record:** RAM use and GPU load (`GR3D_FREQ`) idle and after step 5, the
largest processes, whether the `nvblox_stem` node publishes a mesh
(`ros2 topic list | grep scene_repr/stem`), and any errors in
`J9_semantic.txt`.

---

## Part R — On the robot

The computer is the Orin from now on (Parts W and J passed). These tests
move the robot.

### R0 — Safety rules (read before every session)

- Keep a second person at the power cut-off of the Piper and the Scout for
  every test that can move the robot.
- Clear 2 m around the robot. For base tests on the ground, nobody stands in
  front of or behind it.
- Know the software stops (INSTALL.md Step 9): disable the servo bridge
  (`ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: false}"`),
  stop the base (`ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"`).
  None replaces cutting power.
- Start every motion test at low speed and small distances. Stop at the first
  unexpected motion, cut power if needed, and record what happened before
  trying again.
- Use a stand-in stem for the first grasps: a straight wooden dowel, 6–10 mm
  diameter, painted green, upright in a pot, in front of a plain non-green
  background.

### R1 — CAN interfaces

**Steps:**
1. `modinfo gs_usb | head -3 | tee $LOG/R1_gs_usb.txt` (must exist; see INSTALL.md A.5 if not).
2. `ip -br link show type can | tee $LOG/R1_can_before.txt` — note whether the
   Orin's onboard CAN already uses `can0` / `can1`.
3. Plug in both USB-CAN adapters and bring them up as INSTALL.md 9.1
   (`find_all_can_port.sh`, then `can_activate.sh` with each bus-info).
   If `can0` / `can1` are taken, use other names (e.g. `can_piper`,
   `can_scout`) and pass them to every bringup below as
   `piper_can_port:=<name> scout_can_port:=<name>`.
4. `ip -details link show | grep -A2 can | tee $LOG/R1_can_after.txt`

**Pass:** two interfaces `UP`, the Piper's at 1000000 bit/s, the Scout's at 500000.
**Record:** each adapter's USB bus-info and interface name (INSTALL.md 9.1 asks for them).

### R2 — Piper driver alone

**Steps:**
1. Launch the arm alone (CAN names from R1 if they differ):

   ```bash
   ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true \
     bringup_pipeline:=false bringup_jsp_gui:=false bringup_rviz:=false 2>&1 | tee $LOG/R2_launch.txt
   ```

2. In a second terminal, run the four checks of INSTALL.md 10.3:

   ```bash
   ros2 topic hz /joint_states_single | tee $LOG/R2_feedback_hz.txt     # 10 s
   ros2 topic echo --once /joint_states | tee $LOG/R2_joint_states.txt
   ros2 topic echo --once /arm_status | tee $LOG/R2_arm_status.txt
   ros2 topic info /piper/joint_cmd | tee $LOG/R2_cmd_info.txt
   ```

3. Open RViz on the workstation (J7) and compare the model with the arm.

**Pass:** feedback at a steady rate; `/joint_states` carries `piper_joint1`
to `piper_joint8`; `Publisher count: 0` on `/piper/joint_cmd`; the launch log
shows the arm enabled without an `Automatic enable timeout`; the arm holds
its pose; RViz shows the same pose as the real arm.
**Record:** the feedback rate and any driver warnings.

### R3 — Piper disable behaviour

**Goal:** know whether the arm holds or drops when its motors are disabled.

**Steps:** with R2's launch running, the arm in a low pose, a hand under the
forearm, and the second person at the power cut-off:

```bash
ros2 service call /enable_srv piper_msgs/srv/Enable "enable_request: false"
```

Then stop the launch with Ctrl-C while still supporting the arm; the next
launch enables the arm again.

**Record:** holds / sags / drops (INSTALL.md Step 9 asks for this), and the
physical stop procedure used (how the power of the Piper and the Scout is
cut).

### R4 — Servo on the arm

**Steps:**
1. Launch the arm with servo (CAN names from R1 if they differ):

   ```bash
   ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true bringup_servo:=true \
     bringup_pipeline:=false bringup_jsp_gui:=false bringup_rviz:=false 2>&1 | tee $LOG/R4_launch.txt
   ```

2. In a second terminal, record the flange height, start servo and enable the
   bridge:

   ```bash
   ros2 run scout_piper_bringup print_tcp.py --frame piper_base_link | tee $LOG/R4_before.txt
   ros2 topic echo --field data /servo_node/status std_msgs/msg/Int8 > $LOG/R4_servo_status.txt &
   ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
   ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: true}"
   ```

3. Move up at 2 cm/s for about 3 s (the command stops by itself):

   ```bash
   timeout 3 ros2 topic pub -r 50 /servo_node/delta_twist_cmds geometry_msgs/msg/TwistStamped \
     "{header: {stamp: now, frame_id: piper_base_link}, twist: {linear: {z: 0.02}}}"
   ```

4. Disable the bridge and record the new height:

   ```bash
   ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: false}"
   ros2 run scout_piper_bringup print_tcp.py --frame piper_base_link | tee $LOG/R4_after.txt
   kill %1
   ```

**Pass:** the arm rises smoothly by a few centimetres (about 2 cm/s while
commands arrive), with no sideways motion, and stops when the command
stops; `R4_servo_status.txt` contains only 0.
**Record:** the rise (z in `R4_after.txt` minus `R4_before.txt`), any status
other than 0 (1 / 2: slowing / stopped near a singularity, 3 / 4: slowing /
stopped for a collision, 5: joint limit).
Keep this launch running for R5.

### R5 — Gripper commands through the bridge

**Steps:** with R4's launch running and servo started (the bridge may be
disabled):

```bash
ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: true}"
ros2 topic pub --once /piper_servo_bridge/gripper_cmd std_msgs/msg/Float64 "{data: 0.04}"
ros2 topic echo --once /joint_states_single | tee $LOG/R5_gripper_40mm.txt
ros2 topic pub --once /piper_servo_bridge/gripper_cmd std_msgs/msg/Float64 "{data: 0.06}"
ros2 topic echo --once /joint_states_single | tee $LOG/R5_gripper_60mm.txt
ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: false}"
ros2 topic pub --once /piper_servo_bridge/gripper_cmd std_msgs/msg/Float64 "{data: 0.02}"   # must be ignored
```

Wait about 2 s after each command before reading `/joint_states_single`. In
each saved message, the `position` value at the place of `gripper` in the
`name` list is the opening in metres.

**Pass:** the gripper opens to about 40 mm, then 60 mm, the arm does not
move, and the last command (bridge disabled) does nothing; the launch log
shows `gripper command ignored`.
**Record:** the measured openings. Then stop the launch.

### R6 — Scout base and its command timeout

**Steps:**
1. Lift the Scout so that all wheels are off the ground. Launch the base
   alone (CAN names from R1 if they differ):

   ```bash
   ros2 launch scout_piper_bringup full_system.launch.py bringup_base:=true \
     bringup_pipeline:=false bringup_jsp_gui:=false bringup_rviz:=false 2>&1 | tee $LOG/R6_launch.txt
   ```

2. In a second terminal, command 0.1 m/s, then press Ctrl-C **without**
   sending a zero twist, and watch the wheels for 5 s:

   ```bash
   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.1}}' --rate 5
   ```

3. Send the zero twist and check that the wheels stop:
   `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"`
4. With the base on the ground and 2 m clear in front of it, repeat step 2
   for 2 s and stop it with step 3 (immediately, without waiting).

**Pass:** the wheels turn slowly forward while commands arrive and stop on
the zero twist; on the ground the base creeps forward and stops.
**Record:** whether the wheels kept turning after Ctrl-C in step 2 (and for
how long): this answers the open question of INSTALL.md 10.5 and decides
whether the base needs a software watchdog. Then stop the launch.

### R7 — All three together

**Steps:**
1. Launch arm, base, camera and servo (CAN names from R1 if they differ):

   ```bash
   ros2 launch scout_piper_bringup full_system.launch.py \
     bringup_arm:=true bringup_base:=true bringup_camera:=true bringup_servo:=true \
     bringup_pipeline:=false bringup_jsp_gui:=false bringup_rviz:=false 2>&1 | tee $LOG/R7_launch.txt
   ```

   Leave this launch running for R8–R12 (it is "the R7 bringup" below).
2. In a second terminal:

   ```bash
   ros2 topic hz /joint_states | tee $LOG/R7_joint_states_hz.txt                        # 10 s
   ros2 topic hz /odom | tee $LOG/R7_odom_hz.txt                                        # 10 s
   ros2 topic hz /camera/aligned_depth_to_color/image_raw | tee $LOG/R7_depth_hz.txt    # 10 s
   cd $LOG && ros2 run tf2_tools view_frames && cd -
   ```

**Pass:** all three topics publish at a steady rate; the TF tree in
`$LOG/frames_*.pdf` reaches `camera_link` from `odom`
(`odom → base_link → piper_mount_link → piper_base_link → … → camera_link`).
**If it fails:** a driver that does not start next to the other: check the
CAN names (R1) and that the two robots are on separate adapters.

### R8 — Whole-body MPC

**Goal:** the MPC moves the arm to a point goal, then base and arm together.

**Steps:**
1. Make a slower MPC config for the first runs (base 0.1 m/s, 0.3 rad/s;
   joints 0.3 rad/s):

   ```bash
   sed -e 's/^    v_max: .*/    v_max: 0.1/' -e 's/^    omega_max: .*/    omega_max: 0.3/' \
       -e 's/^    qd_max: .*/    qd_max: 0.3/' \
       $(ros2 pkg prefix scout_piper_whole_body_mpc)/share/scout_piper_whole_body_mpc/config/whole_body_mpc.yaml \
       > ~/mpc_slow.yaml
   grep -E 'v_max|omega_max|qd_max' ~/mpc_slow.yaml        # 0.1, 0.3, 0.3
   ```

2. **Dry run** (moves nothing). With the R7 bringup running:

   ```bash
   ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py \
     config:=$HOME/mpc_slow.yaml profile:=orin 2>&1 | tee $LOG/R8_dry_run.txt
   ```

   In a second terminal, print a goal 5 cm ahead of the gripper tip and send it:

   ```bash
   ros2 run scout_piper_bringup print_tcp.py --forward 0.05 | tee $LOG/R8_goal_arm.txt
   # run the "ros2 topic pub --once /whole_body_mpc/goal ..." line it prints
   ros2 topic echo --once /whole_body_mpc/preview/joint_jog
   ros2 topic echo --once --field data /whole_body_mpc/status std_msgs/msg/String
   ```

   Expect small joint velocities in `joint_jog`, `"execute": false` in the
   status, and no motion. Stop the MPC launch with Ctrl-C.
3. **Arm-only goal, executed.** Start servo and enable the bridge:

   ```bash
   ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
   ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: true}"
   ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py \
     config:=$HOME/mpc_slow.yaml profile:=orin execute:=true 2>&1 | tee $LOG/R8_execute.txt
   ```

   In a second terminal, record the status, then send a fresh 5 cm goal:

   ```bash
   ros2 topic echo --field data /whole_body_mpc/status std_msgs/msg/String > $LOG/R8_status_arm.txt &
   ros2 run scout_piper_bringup print_tcp.py --forward 0.05
   # run the printed "ros2 topic pub --once /whole_body_mpc/goal ..." line
   ```

   To cancel at any moment: `ros2 topic pub --once /whole_body_mpc/cancel std_msgs/msg/Empty "{}"`
   (the MPC stops and goes idle), or disable the bridge and stop the base
   (R0).
4. When the status shows `"mode": "reached"`, stop the recording (`kill %1`)
   and check the tip: `ros2 run scout_piper_bringup print_tcp.py`.
5. **Base and arm goal, executed.** Area clear in front of the robot, base on
   the ground. Record into `R8_status_base.txt` as in step 3, then send a goal
   50 cm ahead, which the arm cannot reach alone:
   `ros2 run scout_piper_bringup print_tcp.py --forward 0.5` and run the
   printed line.
6. Afterwards: disable the bridge, stop the MPC launch, and send the zero
   twist (R0).

**Pass:** the dry run moves nothing; both executed goals end in
`"mode": "reached"` with `tcp_error_m` at or below 0.01, the robot moving
smoothly; the base moves only for the 50 cm goal and stops (last status lines
and `ros2 topic echo --once /odom` show zero twist). Hardware-free reference
with the same config: 5 cm goal reached in 3.9 s, 50 cm goal in 6.6 s with
34 cm of base travel.
**Record:** for each run: time from the goal to `reached`, final
`tcp_error_m`, median `solve_ms` (from the status file), base travel (TCP
printouts before and after), any `MPPI solves overrun` warning in
`R8_execute.txt`, any servo status other than 0
(`ros2 topic echo /servo_node/status`).

### R9 — stem_grasp: scanning the stand-in stem

**Steps:**
1. Make a test config that segments green and does not move the robot:

   ```bash
   sed -e 's/^    mask_mode: .*/    mask_mode: "hsv_green"/' \
       -e 's/^    reach_executor: .*/    reach_executor: "none"/' \
       $(ros2 pkg prefix stem_grasp)/share/stem_grasp/config/pipeline.yaml > ~/stem_grasp_test.yaml
   grep -E '^ +(mask_mode|reach_executor|approach_enabled|grasp_close_gripper):' ~/stem_grasp_test.yaml
   ```

   Expect `approach_enabled: false`, `grasp_close_gripper: false`,
   `reach_executor: "none"`, `mask_mode: "hsv_green"`.
2. Place the dowel 0.6–0.9 m in front of the camera, upright, fully in view.
3. With the R7 bringup running:

   ```bash
   ros2 launch stem_grasp stem_grasp.launch.py config:=$HOME/stem_grasp_test.yaml 2>&1 | tee $LOG/R9_stem_grasp.txt
   ```

4. In a second terminal:

   ```bash
   ros2 topic hz /stem_grasp/mask | tee $LOG/R9_mask_hz.txt                           # 10 s
   ros2 topic echo --once /stem_grasp/target_pose | tee $LOG/R9_target_pose.txt
   ros2 topic echo --once /stem_grasp/pipeline_state
   ```

5. In RViz on the workstation (J7), add an Image display for
   `/stem_grasp/debug_image` and a MarkerArray display for
   `/stem_grasp/skeleton_markers`; take a screenshot.

**Pass:** the mask covers the dowel and nothing else; the skeleton markers
run along the dowel; `/stem_grasp/target_pose` publishes the pre-grasp pose
(in `piper_base_link`), about 12 cm in front of the dowel; the state stays
`SCANNING`.
**Record:** the mask rate, the target pose, the screenshot.
**If it fails:** the mask also covers other things: change the background, or
narrow the colour range by adding, under `stem_grasp_segmentation:` →
`ros__parameters:` in `~/stem_grasp_test.yaml`, the lines
`hsv_lower: [30, 30, 30]` and `hsv_upper: [95, 255, 255]` (OpenCV HSV: hue
0–179) with tighter values, and restart the launch.

### R10 — Reach and image-based servo

**Steps:**
1. Let the pipeline drive the MPC (the final approach stays off):

   ```bash
   sed -i 's/^    reach_executor: .*/    reach_executor: "whole_body_mpc"/' ~/stem_grasp_test.yaml
   ```

2. Stop the R9 launch. With the R7 bringup running, start in this order: the
   MPC with `execute:=true` (R8 step 3, including start_servo and enabling
   the bridge), then the recordings:

   ```bash
   ros2 topic echo --field data /stem_grasp/pipeline_state std_msgs/msg/String > $LOG/R10_states.txt &
   ros2 topic echo --field data /stem_grasp/servo_status std_msgs/msg/String > $LOG/R10_servo_status.txt &
   ```

   and last the pipeline, which starts moving the robot as soon as it sees
   the dowel:

   ```bash
   ros2 launch stem_grasp stem_grasp.launch.py config:=$HOME/stem_grasp_test.yaml 2>&1 | tee $LOG/R10_stem_grasp.txt
   ```

3. After about 20 s in `SERVOING`, disable the bridge, stop the stem_grasp
   launch, and stop the recordings (`kill %1 %2`).

**Pass:** the states go `SCANNING → REACHING → SERVOING`; the log line
`reaching pre-grasp ...; stem diameter X mm` gives about the dowel's diameter
(±2 mm); `error_px` in `R10_servo_status.txt` falls below 8 and stays there;
the gripper points at the dowel, about 12 cm in front of it.
**Record:** the time from `REACHING` to `SERVOING`, the logged diameter and
the dowel's measured diameter, the first and last `error_px`, any servo
status other than 0.

### R11 — Final approach, grasp and release

**Steps:**
1. Turn on the final approach and the grasp:

   ```bash
   sed -i -e 's/^    approach_enabled: .*/    approach_enabled: true/' \
          -e 's/^    grasp_close_gripper: .*/    grasp_close_gripper: true/' ~/stem_grasp_test.yaml
   grep -E '^ +(approach_enabled|grasp_close_gripper|approach_speed_mps):' ~/stem_grasp_test.yaml   # true, 0.02, true
   ```

2. Repeat R10 step 2 (recordings into `R11_run<N>_*.txt`). The pipeline opens
   the gripper, reaches, servos, approaches the last 12 cm at 2 cm/s in steps
   of up to 5 cm, and closes the gripper.
3. When the state is `GRASPED`, check the grip by eye, then release:

   ```bash
   ros2 service call /stem_grasp_pipeline/release std_srvs/srv/Trigger "{}"
   ```

4. When the state is `IDLE`, disable the bridge. Move the dowel 5–10 cm,
   enable the bridge again, and start the next run with
   `ros2 service call /stem_grasp_pipeline/scan std_srvs/srv/Trigger "{}"`.
   Do three runs.

**Pass:** each run goes `APPROACHING → AT_GRASP → GRASPING → GRASPED` and,
after the release, `RELEASING → RETREATING → IDLE`; the fingers close around
the dowel without pushing it over; the log line `grasped: gripper settled at
X mm` is within 2 mm of the dowel's diameter; `retreated X cm` is about 10.
**Record:** per run: the final state, the `approach done after N steps` line,
the settled opening, whether the dowel moved, the retreat distance. On an
`ABORTED` state: the log lines before it.

### R12 — Real plant

Only after three clean R11 runs, and with the stem-segmentation weights on
the Orin:
1. Switch the test config to the model:

   ```bash
   sed -i -e 's/^    mask_mode: .*/    mask_mode: "yolo_seg"/' \
          -e 's|^    yolo_model_path: .*|    yolo_model_path: "/path/to/stem_weights.pt"|' ~/stem_grasp_test.yaml
   ```

   (replace the path with the real one), and turn the approach off again for
   the first look: `sed -i -e 's/^    reach_executor: .*/    reach_executor: "none"/' ~/stem_grasp_test.yaml`.
2. Check the mask on the plant as in R9 (the mask must cover the stem, not
   the leaves).
3. Set `reach_executor` back to `"whole_body_mpc"` and run R10, then R11, on
   a plant stem.

**Record:** the same values as R9–R11, the weights file used, and photos of
the grasp.

---

## Results log

Copy into `$LOG/RESULTS.md` and fill in.

| Test | Date | Machine | Result | Numbers / notes |
|---|---|---|---|---|
| W1 Memory precautions | | workstation | | RAM, swap, previous OOM? |
| W2 Build | | workstation | | limits line, time, lowest available MB |
| W3 Unit tests | | workstation | | counts per package |
| W4 Hardware-free chains | | workstation | | summary, MPPI median/max |
| W5 GPU memory | | workstation | | RAM / GPU MB per stage |
| J1 JetPack check | | Orin | | L4T, JetPack, storage |
| J2 ROS and dependencies | | Orin | | missing packages |
| J3 PyTorch | | Orin | | versions, wheel source |
| J4 Build | | Orin | | limits line, time |
| J5 Unit tests | | Orin | | counts, MPC suite time |
| J6 Hardware-free chains | | Orin | | MPPI median/max (orin, default), overruns |
| J7 Operator station | | both | | hz, clock offsets |
| J8 RealSense | | Orin | | colour / depth Hz |
| J9 GPU stack | | Orin | | RAM, GPU load |
| R1 CAN | | robot | | bus-info and names |
| R2 Piper driver | | robot | | feedback Hz, warnings |
| R3 Disable behaviour | | robot | | holds / sags / drops |
| R4 Servo | | robot | | rise, status |
| R5 Gripper | | robot | | openings |
| R6 Scout timeout | | robot | | keeps turning? |
| R7 All three | | robot | | topic rates, TF tree |
| R8 Whole-body MPC | | robot | | time, error, solve_ms, base travel |
| R9 Scanning | | robot | | mask Hz, target pose |
| R10 Reach + servo | | robot | | reach time, diameter, error_px |
| R11 Approach, grasp, release | | robot | | per run |
| R12 Real plant | | robot | | weights file, per run |

## What to send back

- `RESULTS.md` and the whole log folder (`Codes/test_logs/<date>/` and the
  `hwfree_*` folders), zipped: `cd Codes && zip -r test_logs_$(date +%Y-%m-%d).zip test_logs`.
- For any FAIL: the test ID, the saved output, and what was tried.
- The facts INSTALL.md still lists as open, which these tests produce: the
  JetPack / L4T version (J1), the CAN bus-info values (R1), the Piper disable
  behaviour and the physical stop procedure (R3), the Scout command-timeout
  result (R6), the build time (W2, J4), and where the YOLO stem weights are
  kept (R12).
