# Installation and setup

This guide takes you from a clean machine to a built workspace, then runs
the system one subsystem at a time ([README](README.md) has the short version). Each step says which path it applies to,
where to run it, whether it needs `sudo`, and how to check that it worked.
Lines marked **TODO(maintainer)** are details the repository does not record
yet; do not guess them on the robot.

**Contents:** [what works today](#before-you-start-what-works-today) ·
[choose your path](#choose-your-path) · [1 prerequisites](#step-1--prerequisites) ·
[2 clone](#step-2--clone-the-repository) · [3 system dependencies](#step-3--install-system-dependencies) ·
[4 upstream packages](#step-4--pull-the-upstream-packages) · [5 patches](#step-5--patch-the-upstream-packages) ·
[6 build](#step-6--install-ros-dependencies-and-build) · [7 environment](#step-7--source-the-environment) ·
[8 tests](#step-8--verify-the-build) · [9 hardware](#step-9--set-up-the-robot-hardware) ·
[10 run](#step-10--run-the-system-one-piece-at-a-time) ·
[research without ROS](#research-quick-start-without-ros-path-d) · [troubleshooting](#troubleshooting)

## Before you start: what works today

The system is mid-migration and is not a finished product. Read this before
you power anything on.

| Area | State |
|---|---|
| Full workspace `colcon build` (all upstream + our packages) | **Not recorded as passing** (PROGRESS P0.1.7). Recorded: the hardware-free bringup packages and `--packages-up-to isaac_ros_nvblox` |
| Unified URDF, hardware-free bringup (`robot_state_publisher`, `stem_grasp` nodes, RViz, joint sliders) | Runs on the workstation (P0.3.10, P0.3.11) |
| Piper driver, Scout driver, both together, RealSense inside the bringup | **Not validated on hardware** (P0.3.5–P0.3.8) |
| Piper driver ↔ unified URDF | The driver publishes `joint1`…`joint6` + `gripper` on `/joint_states_single`; the bringup's `piper_joint_state_relay.py` republishes them as `piper_joint1`…`piper_joint8` on `/joint_states` (tested offline, not on hardware) |
| MoveIt 2 demo for the Piper | Upstream demo, separate from the bringup; not validated (P0.3.9) |
| `moveit_servo` | Wired (`bringup_servo:=true`, P0.5.1): reaches the arm only through `piper_servo_bridge`, which starts disabled. Verified end to end on the fake arm (`servo_chain_check.py`); **not yet run on hardware**, speeds not tuned on the robot (P0.5.2, P0.5.3) |
| `stem_grasp` plan → execute, iterative approach | **Not done** (P0.4.11, P0.4.13): the pipeline only publishes `/stem_grasp/target_pose` |
| Software stop (`hotkey_stop_and_zero`) | `x` sends zero twists and disables `piper_servo_bridge`, so servo-driven arm motion stops and the arm holds its measured pose. It does not stop the base and is no substitute for the physical stops |
| Hand-eye calibration | Tools are on the lab machine, not in this repository; camera and mount offsets in the URDF are placeholders |
| Nav2 | **Not usable on this robot as configured** (P0.6.2): `scout_nav2` expects an Ouster 3D lidar and a site map, and the bringup starts its simulation configuration (10.8) |
| RealSense → nvblox reconstruction | Validated in the dev container with a D405 (P1.1.2) |
| Per-class semantic nvblox (`nvblox_semantic.launch.py`) | Never run (P1.2.1) |
| CPU semantic map, distance field, MoveIt semantic collision plugin | Unit-tested; not run in `move_group` on the robot |
| `plant_twin`, `scout_piper_jepa`, `scout_piper_whole_body_mpc` | Tested offline on synthetic data only; the MPC is dry-run by default; V-JEPA inference has never run |
| Jetson Orin AGX deployment | Not documented (path A) |

> **Safety: how the Piper driver takes commands.** The driver executes every
> message on its command input as a joint position command at 100 % speed,
> and any joint not named `joint1`…`joint6` is commanded to 0. Upstream
> `start_single_piper.launch.py` wires that input to **`/joint_states`**, so
> with the upstream launch file the joint sliders, `view_robot.launch.py`, the
> MoveIt demo or any test publisher move the real arm, typically straight to
> the all-zero pose. `full_system.launch.py` (Step 10) therefore:
>
> - starts the driver with its command input on `arm_command_topic` (default `/piper/joint_cmd`) and refuses joint-state topics there;
> - republishes the driver's feedback on `/joint_states` as `piper_joint1`…`piper_joint8` (`piper_joint_state_relay.py`);
> - never starts the joint sliders while `bringup_arm:=true`.
>
> Anything published on `/piper/joint_cmd` moves the arm at full speed, and the
> bringup enables the arm automatically (`auto_enable: true`). Do not start
> the upstream `start_single_piper.launch.py` directly unless no node
> publishes `/joint_states`. (Upstream code: `src/piper_ros/src/piper/launch/start_single_piper.launch.py`
> and `piper_ctrl_single_node.py`, `joint_callback`, after Step 4.)

## Choose your path

| Path | Machine | Use it for | Documented here |
|---|---|---|---|
| **A** | Jetson Orin AGX on the robot | Final deployment | **No** — see [Path A](#path-a-jetson-orin-agx-not-documented-yet) |
| **B** | x86_64 Ubuntu 22.04, native ROS 2 Humble | Development; hardware bring-up from a 22.04 machine | Yes; the nvblox build dependencies exist only as Dockerfile steps (3B.4, untested natively) |
| **C** | x86_64 Ubuntu 20.04 workstation + Docker dev container | The lab workstation; the path on which nvblox was built and validated | Yes ([`Codes/README.md`](Codes/README.md), [`Codes/docker/README.md`](Codes/docker/README.md)) |
| **D** | Any machine with Python 3.10–3.12, no ROS, no robot | Algorithm work: pure-Python tests and synthetic benchmarks | Yes ([below](#research-quick-start-without-ros-path-d)) |

| Step | B | C | D |
|---|---|---|---|
| 1 Prerequisites | yes | yes | yes |
| 2 Clone | yes | yes | yes |
| 3 System dependencies | 3B | 3C | skip |
| 4 Pull upstream packages | yes | yes, on the host | skip |
| 5 Patch upstream packages | yes | yes, on the host | skip |
| 6 rosdep + build | 6B | 6C, in the container | skip |
| 7 Source the environment | yes | yes (container) | skip |
| 8 Verify the build | yes | yes (container) | [Research quick start](#research-quick-start-without-ros-path-d) |
| 9 Hardware setup | robot only | robot only | skip |
| 10 Run the system | yes | yes (container) | skip |

### Path A: Jetson Orin AGX (not documented yet)

The Phase 0 decision ([`Codes/README.md`](Codes/README.md)) is to bring up
on the workstation first and move to the Jetson Orin AGX later, "via Docker
or native rebuild". No Jetson procedure exists yet, and the dev image cannot
be reused as is: it is x86_64 only (CUDA apt repo `ubuntu2204/x86_64`, VPI
repo `jetson/x86_64/jammy`, GXF prebuilts `gxf_x86_64_cuda_12_6`). The dev
container uses host networking, so it can already talk DDS to a Jetson on the
same LAN if both use the same `ROS_DOMAIN_ID` (the container defaults to `42`).

Constraint from the pinned upstreams: Isaac ROS release-3.2 builds its
arm64 images on CUDA 12.6 / Ubuntu 22.04 (`isaac_ros_common` `docker/Dockerfile.x86_64`,
`base-arm64` stage), and ROS 2 Humble needs Ubuntu 22.04, so the Jetson needs a
JetPack 6 release with Ubuntu 22.04 and CUDA 12.6.

- **TODO(maintainer):** record the Orin's JetPack / L4T version, then document native vs. aarch64 container, the aarch64 VPI and GXF sources, the nvblox build and the torch wheel for V-JEPA.

## Step 1 — Prerequisites

**Hardware (robot):** AgileX Piper arm, AgileX Scout 2.0 base, one USB-CAN
adapter per robot (the Piper on `can0`, the Scout on `can1`), and an Intel
RealSense **D405** mounted eye-in-hand (confirmed 2026-10-06; the URDF uses
the `sensor_d405` macro). In Phase 0 the adapters and the camera plug into the
x86_64 machine running path B or C (the dev container passes USB through).

**Workstation (paths B and C):** x86_64 with an NVIDIA GPU and its driver
already installed. The dev container requests the `nvidia` runtime and will
not start without it; the nvblox build needs CUDA. Path D needs no GPU.

- **Driver:** R560 or newer. The GXF prebuilts that `patch_upstream.sh` pulls are built for CUDA 12.6 (`gxf_x86_64_cuda_12_6`) and the Isaac ROS 3.2 images are CUDA 12.6, whose release driver branch is R560. Older CUDA 12-capable drivers rely on CUDA minor-version compatibility and are untested here. Check with `nvidia-smi` (driver version and "CUDA Version: 12.6" or higher).
- **TODO(maintainer):** record the GPU of the lab workstation where nvblox was validated (P1.1.2) as the known-good model.

**Operating system:** path B Ubuntu 22.04 (jammy); path C Ubuntu 20.04
(focal), which `install_docker_nvidia.sh` is written for; path D any OS with
Python 3.10–3.12 (3.13+ works but has to compile NumPy 1.x from source, see
path D).

**Base tools** (Ubuntu, `sudo`, once): `git` for Step 2, `python3-pip` for
path C's host tools, `python3-venv` for path D.

```bash
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv
```

**Disk and time.** The repository records only per-component figures:

| Item | Size / time |
|---|---|
| Dev image: `osrf/ros:humble-desktop-full` base | ~3 GB download, plus ROS/Python, CUDA and VPI layers |
| CUDA 12.4 toolkit layer | ~1.5 GB; rebuilding the image takes ~10 min |
| GXF prebuilt libraries (Git LFS, `isaac_ros_nitros`) | ~50–100 MB |
| `nvblox_core` submodule | ~50 MB |
| nvblox CUDA build | 15–30 min on a workstation |
| Optional: torch (path D, Linux, default CUDA build) | ~4–5 GB; the CPU build is much smaller |
| Optional: Grounded-SAM | torch plus a ~2 GB checkpoint set |

- **TODO(maintainer):** state the total disk space and full-workspace build time.

**Network.** Internet access to GitHub (this repository, the upstream
repositories in `Codes/repos.yaml`, `magic_enum`, Git LFS objects), Docker
Hub, PyPI, and the Docker / NVIDIA / ROS apt repositories.

## Step 2 — Clone the repository

**Paths:** all. **Where:** host. **sudo:** no.

```bash
git clone https://github.com/tuananh1007/Piper-Scout.git
cd Piper-Scout
```

**Check:** `ls` shows `Codes  PROGRESS.md  README.md  ROADMAP.md  research`.

This repository has no Git LFS files; Git LFS is needed only in Step 5, for
the upstream `isaac_ros_nitros` binaries. All later commands start from the
repository root unless they say otherwise. Inside the dev container, `Codes/`
is mounted at `/workspace`.

## Step 3 — Install system dependencies

### 3B — Native Ubuntu 22.04 (path B)

**3B.1 ROS 2 Humble.** **Where:** host. **sudo:** yes. Follow the official
[ROS 2 Humble installation guide](https://docs.ros.org/en/humble/Installation.html)
and install `ros-humble-desktop-full` (the dev image's base is
`osrf/ros:humble-desktop-full`).

```bash
source /opt/ros/humble/setup.bash
printenv ROS_DISTRO        # expect: humble
```

**3B.2 apt packages.** **Where:** host. **sudo:** yes. This list mirrors
`Codes/docker/Dockerfile.dev`, plus `can-utils` and `ethtool` for the Piper's
CAN script (Step 9).

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake git git-lfs curl gnupg lsb-release \
  python3-pip python3-pytest python3-colcon-common-extensions python3-vcstool \
  python3-rosdep python3-argcomplete \
  ros-humble-moveit ros-humble-moveit-servo \
  ros-humble-ros2-control ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui \
  ros-humble-rviz2 ros-humble-xacro \
  ros-humble-nav2-bringup ros-humble-tf2-tools ros-humble-tf-transformations \
  ros-humble-realsense2-camera ros-humble-realsense2-description \
  ros-humble-rqt-gui ros-humble-rqt-graph ros-humble-rqt-image-view \
  libopencv-dev python3-opencv libasio-dev ros-humble-rmw-cyclonedds-cpp \
  can-utils ethtool
```

**Check:** `vcs --version`, `colcon --help` and `git lfs version` run without errors.

**3B.3 Python packages.** **Where:** host. **sudo:** no (`--user`).

```bash
python3 -m pip install --user \
  'numpy<2' scipy scikit-image scikit-learn networkx open3d ultralytics 'transforms3d>=0.4' \
  piper_sdk python-can
```

- Keep NumPy below 2: the ROS Humble apt packages (`cv_bridge`, `sensor_msgs_py`) are built against NumPy 1.x and fail with `_ARRAY_API not found` under NumPy 2.
- `piper_sdk` and `python-can` are runtime dependencies of the upstream Piper driver (its README, §1.1) that rosdep does not install; without them `bringup_arm:=true` fails with `ModuleNotFoundError: No module named 'piper_sdk'`.
- `ultralytics` pulls in `torch` and `torchvision`, a large download.

**Check:** `python3 -c "import numpy, piper_sdk; print(numpy.__version__)"` prints a 1.x version.

**3B.4 CUDA 12.4, build libraries, VPI 4, `magic_enum`** (only for the
nvblox / Isaac ROS packages). **Where:** host. **sudo:** yes. A plain
`colcon build` in Step 6 builds the Isaac ROS packages too, and they need all
of this. These are the `Codes/docker/Dockerfile.dev` steps run with `sudo`;
they have **not been tested on a native machine**.

```bash
# CUDA 12.4 toolkit (nvcc + headers) and the nvblox build libraries
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
  -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  cuda-toolkit-12-4 \
  libgflags-dev libgoogle-glog-dev libsqlite3-dev libbenchmark-dev \
  libgtest-dev libgmock-dev
export CUDA_HOME=/usr/local/cuda-12.4                       # add these three lines to ~/.bashrc
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# VPI 4 (libnvvpi4, vpi4-dev); the script checks for /opt/ros/humble and jammy
(cd Codes && ./scripts/install_vpi.sh)

# magic_enum v0.9.7 plus the include symlink that GXF release-3.2 expects
git clone --depth 1 -b v0.9.7 https://github.com/Neargye/magic_enum.git /tmp/magic_enum
cmake -S /tmp/magic_enum -B /tmp/magic_enum/build \
  -DMAGIC_ENUM_OPT_INSTALL=ON -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF \
  -DMAGIC_ENUM_OPT_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
sudo cmake --install /tmp/magic_enum/build
sudo ln -sfn /usr/local/include/magic_enum/magic_enum.hpp /usr/local/include/magic_enum.hpp
```

**Check:** `nvcc --version` reports release 12.4; `install_vpi.sh` prints
where `vpiConfig.cmake` is; `ls -l /usr/local/include/magic_enum.hpp` shows
the symlink.

- **TODO(maintainer):** validate or replace this native CUDA / VPI / `magic_enum` setup.

### 3C — Ubuntu 20.04 workstation with the dev container (path C)

Ubuntu 20.04 has no ROS 2 Humble packages, so ROS runs in a Docker container
that provides ROS 2 Humble, MoveIt 2, ros2_control, Nav2, RealSense, the
Python packages of 3B.3 (except `piper_sdk` / `python-can`, see 3C.3) and the
CUDA 12.4 / VPI 4 / `magic_enum` build dependencies. The host needs Docker,
the NVIDIA Container Toolkit and the tools for Steps 4 and 5.

**3C.1 Docker and the NVIDIA Container Toolkit.** **Where:** host. **sudo:**
used internally; run it as your normal user (it exits if run as root).

```bash
cd Codes
./scripts/install_docker_nvidia.sh
```

It installs Docker CE with the compose and buildx plugins, adds you to the
`docker` group, installs `nvidia-container-toolkit`, registers the NVIDIA
runtime and restarts Docker. Log out and back in (or `newgrp docker`) so the
group change takes effect.

**Check** (prints the `nvidia-smi` GPU table):

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**3C.2 Host tools for Steps 4, 5 and 9.** **Where:** host. **sudo:** for apt only.

```bash
python3 -m pip install --user vcstool       # needs python3-pip from Step 1
PATH=$HOME/.local/bin:$PATH vcs --version   # expect: vcs 0.3.0 or newer
sudo apt-get install -y can-utils ethtool   # robot only: the Piper CAN script needs them (Step 9)
```

**3C.3 Build the dev image.** **Where:** host, in `Codes/`. **sudo:** no
(needs the `docker` group from 3C.1).

```bash
./docker/build_dev.sh
```

This builds `piper-scout-dev:humble` with your host UID/GID (expect a ~3 GB
base download plus the CUDA and VPI layers).

**Check:** the script prints `Done. Drop into the dev container with:`
followed by the `docker compose` command, and `docker image ls piper-scout-dev`
lists the `humble` tag.

- The image installs `piper_sdk` and `python-can` (the arm driver's runtime dependencies) since 2026-10-07; an image built before that needs `./docker/build_dev.sh` again.
- `ultralytics` pulls in an unpinned `torch`. V-JEPA additionally needs `timm` and `einops` (vjepa2 `requirements.txt`: `torch>=2`); they are not in the image, so install them when switching the JEPA node to `encoder: vjepa` (10.12).

## Step 4 — Pull the upstream packages

**Paths:** B, C. **Where:** host, in `Codes/` (also on path C). **sudo:** no.

Upstream packages are not committed: `vcs import` clones them into
`Codes/src/` from `Codes/repos.yaml`, and `.gitignore` excludes them.

```bash
# in Codes/
vcs import src < repos.yaml                              # path B
PATH=$HOME/.local/bin:$PATH vcs import src < repos.yaml  # path C (vcstool from pip --user)
```

| Directory | Source @ version | Role |
|---|---|---|
| `piper_ros/` | `agilexrobotics/piper_ros@humble` | Piper driver, ros2_control, MoveIt 2 config, CAN scripts |
| `scout_ros2/` | `agilexrobotics/scout_ros2@humble` | Scout base CAN driver |
| `ugv_sdk/` | `westonrobot/ugv_sdk@main` | CAN SDK used by `scout_base` |
| `scout_nav2/` | `AIRLab-POLIMI/scout_nav2@main` | Nav2 stack tuned for the Scout |
| `realsense-ros/` | `IntelRealSense/realsense-ros@4.58.4` | Camera driver (matches ROS Humble's librealsense2 2.58, 9.2) |
| `isaac_ros_common/`, `isaac_ros_nvblox/`, `isaac_ros_nitros/`, `isaac_ros_gxf/` | `NVIDIA-ISAAC-ROS/*@release-3.2` (`isaac_ros_gxf` is the `gxf` repo) | nvblox from source; release-3.2 is the last Humble tag and there are no Isaac ROS apt packages for Humble any more |
| `negotiated/` | `osrf/negotiated@master` | Needed by `isaac_ros_nitros` |

**Check:**

```bash
ls src
# expect, besides our packages: piper_ros scout_ros2 ugv_sdk scout_nav2 realsense-ros
#   isaac_ros_common isaac_ros_nvblox isaac_ros_nitros isaac_ros_gxf negotiated
ls src/piper_ros/src/piper/launch/start_single_piper.launch.py \
   src/scout_nav2/scout_nav2/launch/nav2.launch.py \
   src/realsense-ros/realsense2_camera/launch/rs_launch.py
```

If an upstream path has moved, fix the references in
`src/scout_piper_bringup/launch/full_system.launch.py`,
`src/scout_piper_description/urdf/scout_piper.urdf.xacro` and
`scripts/fork_piper_arm.py`.

## Step 5 — Patch the upstream packages

**Paths:** B, C. **Where:** host, in `Codes/`. **sudo:** only to install
`git-lfs` the first time. Run it after **every** `vcs import`; it is idempotent.

```bash
git lfs version || ./scripts/install_git_lfs_and_pull.sh   # one-time: apt-installs git-lfs (sudo) and pulls the GXF LFS files
./scripts/patch_upstream.sh
```

`patch_upstream.sh` declares `build_type` `cmake` in `ugv_sdk/package.xml`;
removes the `NVENC` backend entry that VPI 4 dropped; adds the missing
`magic_enum` / `negotiated` CMake dependencies to `isaac_ros_nitros` and
ports `nitros_image.cpp` to the VPI 4 field names; pulls the GXF prebuilt
`.so` files through Git LFS (without them NITROS fails to link); and
initialises the `nvblox_core` submodule. If `git-lfs` is missing it only
warns and carries on.

**Check:**

```bash
# the script ends with: "Patches applied. You can now run colcon build."
grep build_type src/ugv_sdk/package.xml       # expect: <build_type>cmake</build_type>
file src/isaac_ros_nitros/isaac_ros_gxf/gxf/core/lib/gxf_x86_64_cuda_12_6/core/libgxf_core.so
                                              # expect: ELF 64-bit ... shared object (not ASCII text)
ls src/isaac_ros_nvblox/nvblox_ros/nvblox_core/CMakeLists.txt
```

`Codes/build_workspace.sh` runs `vcs import` by itself when `src/piper_ros`
is missing but never runs `patch_upstream.sh`, so always import and patch
first as shown here.

## Step 6 — Install ROS dependencies and build

The first full build includes the Isaac ROS CUDA packages (15–30 min or
more). `colcon build` writes `build/`, `install/` and `log/` into `Codes/`;
all three are gitignored. A full-workspace build is the goal but has not
been recorded as passing yet (P0.1.7): if it fails, check
[Troubleshooting](#troubleshooting) before assuming you made a mistake.

### 6B — Native Ubuntu 22.04 (path B)

**Where:** host, in `Codes/`. **sudo:** for `rosdep init` and `rosdep install`.

```bash
source /opt/ros/humble/setup.bash
sudo rosdep init          # first time on this machine only; "already initialized" is fine
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

**Check:**

- Because of `-r`, rosdep prints `#All required rosdeps installed successfully` even when some keys could not be resolved, so read the output above that line. An `ERROR: the following packages/stacks could not have their rosdep keys resolved` entry is expected for `nvblox_examples_bringup` (Isaac ROS example dependencies that are not built here); any other package in that list is a real missing dependency.
- `colcon build` ends with `Summary: <N> packages finished` and no `packages failed` line.

`./build_workspace.sh` (with extra colcon arguments, e.g.
`./build_workspace.sh --packages-select stem_grasp`) is a shortcut for later
rebuilds only: it runs `rosdep install ... || true`, which hides failures,
and skips `rosdep init` / `rosdep update`.

Without CUDA, build only the packages that do not depend on Isaac ROS:

```bash
colcon build --symlink-install --packages-up-to scout_piper_bringup plant_twin scout_piper_jepa
```

`colcon list --packages-up-to` on the imported workspace resolves this to 19
packages (our bringup, description, `stem_grasp`, `plant_twin`,
`scout_piper_jepa`, the Piper, Scout and RealSense packages, `ugv_sdk`,
`scout_nav2`) and none from Isaac ROS. `scout_piper_scene_repr` and
`scout_piper_whole_body_mpc` each pull in 26 Isaac ROS packages, so they need
the CUDA setup; their pure-Python parts still run without ROS (path D).
- If the upstream `piper_ros` build needs apt packages that rosdep does not cover, see its README (`src/piper_ros/README(EN).MD`).

### 6C — Inside the dev container (path C)

**Where:** enter from the host in `Codes/`, build in the container. **sudo:** no.

RViz in the container needs X11 access. The compose file bind-mounts
`~/.Xauthority`; if that file does not exist, Docker creates a root-owned
**directory** in its place. Check it **before** the first start:

```bash
# Host:
ls -l ~/.Xauthority || xauth extract ~/.Xauthority "$DISPLAY"   # must be a regular file
# (if it is a directory: sudo rmdir ~/.Xauthority, then run the line again)
docker compose -f docker/compose.dev.yml run --rm dev
```

You are now in `/workspace` (the bind-mounted `Codes/`) as user `dev`
(passwordless `sudo`), with ROS sourced.

```bash
# Container, in /workspace:
printenv ROS_DISTRO                       # expect: humble
nvcc --version                            # expect: release 12.4
colcon build --symlink-install --packages-up-to isaac_ros_nvblox   # optional first pass: 15-30 min of CUDA
colcon build --symlink-install
```

**Check:** `colcon build` ends with `Summary: <N> packages finished` and no
`packages failed` line; on the host, `Codes/install/` now exists.

The image pre-installs the dependencies, and the Codes README does not run
rosdep in the container. If colcon reports a missing package, run
`rosdep update && rosdep install --from-paths src --ignore-src -r -y` in the
container, then add the package to `docker/Dockerfile.dev` and rebuild the
image, because the container is discarded on exit.

Notes on the container:

- Every `docker compose ... run --rm dev` creates a new container and removes it on exit; only `/workspace` (`Codes/`) persists.
- It runs privileged with host networking, host IPC, the NVIDIA runtime, X11, USB (`/dev/bus/usb`) and `/sys/class/net`, so it sees the host's CAN interfaces and the camera.
- `ROS_DOMAIN_ID` defaults to `42`; machines outside the container must use the same value to see its topics.
- A second shell in the **same** container: `docker exec -it $(docker ps -q --filter ancestor=piper-scout-dev:humble | head -n 1) bash` (standard Docker; `docker compose run` containers have generated names, so filter by image). Each new shell sources ROS from `~/.bashrc`.
- X11: the container runs as your UID with host networking and mounts `~/.Xauthority` and `/tmp/.X11-unix`, so RViz authenticates with your own X cookie and `xhost` is not needed as long as `~/.Xauthority` holds the cookie for `$DISPLAY` (the check above). If RViz still reports `Authorization required`, allow your local user only: `xhost +SI:localuser:$(id -un)` (never `xhost +`).

## Step 7 — Source the environment

**Paths:** B, C. **Where:** every new terminal (host for B, container for C). **sudo:** no.

```bash
cd Codes                           # from the repository root; /workspace in the container
source /opt/ros/humble/setup.bash  # path B; the container's ~/.bashrc already does this
source install/setup.bash
```

`source setup_env.sh` (from `Codes/`, sourced, not executed) does the same
and exports `PIPER_SCOUT_WS`. In the container, `~/.bashrc` sources
`/workspace/install/setup.bash` in shells started after the first build; in
the shell where you just built, source it by hand.

**Check:**

```bash
ros2 pkg prefix scout_piper_bringup     # expect: .../Codes/install/scout_piper_bringup
ros2 pkg list | grep -E 'scout_piper|stem_grasp|plant_twin'
```

- RMW: nothing in the repository sets `RMW_IMPLEMENTATION`, so everything so far, including the nvblox validation (P1.1.2), ran on Humble's default, Fast DDS (`rmw_fastrtps_cpp`). Keep the default. `rmw_cyclonedds_cpp` is installed as an alternative; if you switch, export the same `RMW_IMPLEMENTATION` on every machine and container, since mixed RMWs do not reliably talk to each other.

## Step 8 — Verify the build

**Paths:** B, C. **Where:** `Codes/` on the host (B) or `/workspace` in the container (C). **sudo:** no.

ROS package tests:

```bash
colcon test --packages-select stem_grasp scout_piper_bringup && colcon test-result --verbose
colcon test --packages-select scout_piper_scene_repr \
  --ctest-args -R 'scene_repr_python|test_semantic_collision_plugin'
colcon test-result --verbose
```

`scout_piper_bringup` checks the arm-command isolation of `full_system.launch.py`
and the joint-state relay. The `scout_piper_scene_repr` filter runs the functional tests: the pytest
suite and a gtest that loads the semantic collision plugin into a real
MoveIt robot model. Without the filter, the `ament_lint_auto` tests
(copyright, cpplint, uncrustify, flake8, pep257, …) also run; they currently
fail because the sources have no license headers or lint formatting, which
does not mean the install is broken.

**Check:** `colcon test-result --verbose` reports no errors or failures for these tests.

Pure-Python test suites (no ROS needed; the same commands work on path D):

```bash
cd src                  # from Codes/ (or /workspace in the container)
( cd scout_piper_scene_repr     && python3 -m pytest test -q )
( cd plant_twin                 && python3 -m pytest test -q )
( cd scout_piper_jepa           && python3 -m pytest test -q )
( cd scout_piper_whole_body_mpc && PYTHONPATH=../scout_piper_scene_repr/python python3 -m pytest test -q )
cd ..
```

Expected results are listed under
[Research quick start](#research-quick-start-without-ros-path-d). If
`python3 -m pytest --version` fails in the container, install it with
`sudo apt-get install -y python3-pytest` (lost when the container exits).

## Step 9 — Set up the robot hardware

**Paths:** B, C. **Where:** the x86_64 machine running path B or C, with the
Piper and Scout USB-CAN adapters and the RealSense plugged into it (CAN
interfaces are brought up on the host; the dev container shares the host
network namespace). **sudo:** yes.

> **Safety.** Nothing in this step moves the robot, but Step 10 does. Before
> Step 10.3, make sure you know how to cut power to the Piper and the Scout.
> Read the [`/joint_states` warning](#before-you-start-what-works-today) above.
> Software stops, none of them a substitute for cutting power:
>
> - servo-driven arm motion: `ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: false}"` or `x` in `hotkey_stop_and_zero`; the arm holds its measured pose (10.4);
> - Scout: `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"`; nothing stops the base when commands stop arriving (10.5);
> - Piper motors: `ros2 service call /enable_srv piper_msgs/srv/Enable "enable_request: false"` (driver `DisableArm`). Upstream does not say whether the arm holds or drops when disabled, so support the arm the first time you try it.
>
> - **TODO(maintainer):** record the physical stop procedure of this robot (how the Scout and the Piper are stopped or their power cut) and the result of the first disable test.

**9.1 CAN interfaces.** The bringup hardcodes the interface names in
`full_system.launch.py`: `can_port: can0` for the Piper, `port_name: can1`
for the Scout (the `piper_driver` / `scout_driver` entries in `system.yaml`
are not passed to the drivers). Keep the two robots on separate buses. The
CAN scripts come with `piper_ros` (Step 4) and need `can-utils` and `ethtool`
(3B.2 / 3C.2). With two adapters plugged in, `can_activate.sh` needs each
adapter's USB bus-info, which `find_all_can_port.sh` prints:

```bash
cd src/piper_ros                                     # from Codes/
bash find_all_can_port.sh                            # note each adapter's USB bus-info, e.g. 3-1.4:1.0
bash can_activate.sh can0 1000000 <piper-bus-info>   # Piper: 1 Mbit/s, fixed (piper_ros README §2)
bash can_activate.sh can1 500000  <scout-bus-info>   # Scout: 500 kbit/s (scout_ros2 README)
```

**Check:**

```bash
ip -details link show can0     # expect: state UP, bitrate 1000000
ip -details link show can1     # expect: state UP, bitrate 500000
```

**Fixed names across reboots.** The bus-info is the physical USB port, so as
long as each adapter stays in its port the mapping is stable. Upstream
`can_config.sh` brings up both adapters with per-port names and bitrates (no
separate Scout script needed). Copy it outside the workspace so `vcs import`
does not overwrite it, and fill in the two ports:

```bash
cp src/piper_ros/can_config.sh ~/can_config.sh      # from Codes/
# edit ~/can_config.sh: keep EXPECTED_CAN_COUNT=2 and replace the USB_PORTS lines with
#   USB_PORTS["<piper-bus-info>"]="can0:1000000"
#   USB_PORTS["<scout-bus-info>"]="can1:500000"
bash ~/can_config.sh                                 # run after every boot (or from a systemd unit)
```

- These commands follow the upstream scripts and READMEs (`src/piper_ros/README(EN).MD` §2, `src/scout_ros2/README.md`) and have not been run on this robot yet. **TODO(maintainer):** record the two bus-info values once confirmed.

**9.2 RealSense (D405).** The SDK is ROS Humble's `ros-humble-librealsense2`
(2.58.4 as of October 2026), which rosdep and the `ros-humble-realsense2-camera`
apt package install; no separate Intel SDK is needed. `repos.yaml` pins the
source `realsense-ros` to `4.58.4`, which needs librealsense2 ≥ 2.58.0 (the
`ros2-development` branch needs 2.59 and would not configure). After
`source install/setup.bash` the workspace build of `realsense2_camera`
overrides the apt one; both are 4.58.x.

The ROS SDK package uses the libusb backend and installs no udev rules. On a
fresh host (the one the camera plugs into), install Intel's rules for that
release, then replug the camera:

```bash
curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.58.4/config/99-realsense-libusb.rules \
  | sudo tee /etc/udev/rules.d/99-realsense-libusb.rules > /dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**Check:** `dpkg -l | grep librealsense2` shows 2.58.x; `lsusb | grep -i intel`
lists the camera; the real check is Step 10.6.

**9.3 Calibration placeholders.** The Piper mount offset (`base_link` →
`piper_mount_link`, xyz `0.15 0.0 0.18`) and the camera offset (`cam_xyz`
`0.05 0.0 0.05`) in `scout_piper.urdf.xacro` are placeholders; no launch file
passes `cam_xyz` / `cam_rpy`, so a calibrated pose currently means editing
the xacro defaults. The hand-eye calibration scripts are still on the lab
machine ([checklist Step 5](Codes/PHASE0_CHECKLIST.md)).

## Step 10 — Run the system, one piece at a time

**Paths:** B, C. **Where:** host (B) or container (C), with Step 7 sourced in
every terminal. **sudo:** no.

Run **one** `full_system.launch.py` at a time: stop it with Ctrl-C before the
next numbered bringup, and check with `ros2 node list` that its nodes are
gone. Use extra terminals only for the check commands, or for items that say
they need a running bringup. Do not debug the integrated stack while an
individual driver is broken.

### 10.1 URDF check (no hardware)

```bash
ros2 launch scout_piper_description view_robot.launch.py                  # with sliders
ros2 launch scout_piper_description view_robot.launch.py use_gui:=false   # headless joint_state_publisher
```

**Pass:** RViz shows the Scout 2.0 with the Piper on top and a camera at the
end effector, and the sliders move the arm. If not, inspect the TF tree with
`ros2 run tf2_tools view_frames`. (This publishes `/joint_states`; do not run
it next to a bringup with `bringup_arm:=true`, whose relay owns that topic.)

### 10.2 Hardware-free bringup

```bash
ros2 launch scout_piper_bringup full_system.launch.py
```

With the defaults no CAN or USB device is touched: arm, base and camera are
off, and the launch starts `robot_state_publisher`, the three `stem_grasp`
nodes, RViz and the joint sliders.

**Check:** `ros2 node list` includes `/stem_grasp_pipeline`,
`/stem_grasp_segmentation`, `/stem_grasp_pointcloud` and
`/robot_state_publisher`. Segmentation runs with its code defaults here, so
it publishes empty masks (see 10.9). `use_sim:=true` only sets
`use_sim_time`; it starts no simulator.

> **Safety (10.3 onward).** These steps command real hardware, and none of
> them is validated on hardware yet. `bringup_arm:=true` enables the Piper
> automatically, and the arm then executes whatever arrives on
> `/piper/joint_cmd` (see the [warning](#before-you-start-what-works-today)).
> `hotkey_stop_and_zero` stops only motion that goes through moveit_servo. Keep people clear of the arm
> and the base and keep the power cut-off within reach.

### 10.3 Piper arm only

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=true bringup_base:=false bringup_camera:=false \
  bringup_pipeline:=false bringup_jsp_gui:=false bringup_rviz:=true
```

With the arm on, the bringup skips the joint sliders (it logs that
`bringup_jsp_gui` is ignored), starts the driver with its commands on
`/piper/joint_cmd`, and starts `piper_joint_state_relay`. Nothing in the
bringup publishes on `/piper/joint_cmd`, so the enabled arm holds its pose.

**Pass:**

```bash
ros2 topic hz /joint_states_single              # driver feedback (joint1..joint6, gripper)
ros2 topic echo --once /joint_states            # the same values as piper_joint1..piper_joint8, from the relay
ros2 topic echo --once /arm_status              # driver status
ros2 topic info /piper/joint_cmd                # Publisher count: 0
```

The driver log shows the arm enabled without an "Automatic enable timeout",
and in RViz the arm model follows the real arm. (`ros2 topic list` showing
`/joint_states` alone proves nothing: a topic is listed as soon as anything
subscribes to it.)

### 10.4 moveit_servo

`bringup_servo:=true` starts `servo_node` (moveit_servo, `config/moveit/servo.yaml`,
unified SRDF `config/moveit/scout_piper.srdf`) and `piper_servo_bridge`:

```text
/servo_node/delta_twist_cmds (TwistStamped, m/s, rad/s)  ─┐
/servo_node/delta_joint_cmds (JointJog, rad/s)           ─┴─▶ servo_node ─▶ /piper/servo/joint_trajectory
    ─▶ piper_servo_bridge (~/enable) ─▶ /piper/joint_cmd ─▶ Piper driver ─▶ /joint_states_single ─▶ relay ─▶ /joint_states
```

Two latches must be opened before anything moves: servo itself waits for
`/servo_node/start_servo`, and the bridge starts **disabled**. While enabled,
the bridge keeps every target inside the joint limits and within 0.1 rad of
the measured position, sends the measured gripper opening (so the gripper
holds) and caps the driver at 30 % speed. Servo checks self-collision,
including the Scout chassis, but knows nothing about plants or obstacles
unless they are added to its planning scene. Disabling the bridge sends one
"hold the measured pose" command. Its parameters are `speed_percent`,
`max_step_rad` and `max_feedback_age_s` (`scripts/piper_servo_bridge.py`).

**1. Hardware-free check (do this first).** `fake_arm:=true` replaces the
driver with `fake_piper_driver.py`, which mimics the real driver's command
handling without touching CAN:

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=true fake_arm:=true bringup_servo:=true bringup_pipeline:=false
ros2 run scout_piper_bringup servo_chain_check.py      # second terminal
```

**Pass:** eight `PASS` lines and `SERVO CHAIN OK` (relay names, bridge latch,
joint jog, command contents, gripper hold, Cartesian twist, hold after
disable, no singularity / collision halt). The check refuses to run unless
the fake driver is the one answering.

**2. On the robot** (after 10.3 passes; arm clear, power cut-off within
reach, start with small speeds):

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=true bringup_servo:=true bringup_pipeline:=false bringup_jsp_gui:=false
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"
ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: true}"
ros2 topic pub -r 50 /servo_node/delta_twist_cmds geometry_msgs/msg/TwistStamped \
  "{header: {stamp: now, frame_id: piper_base_link}, twist: {linear: {z: 0.02}}}"   # 2 cm/s up; Ctrl-C stops
ros2 service call /piper_servo_bridge/enable std_srvs/srv/SetBool "{data: false}"  # hold
```

Commands need a current stamp (`stamp: now`): servo treats zero stamps as
stale. It halts by itself 0.25 s after the last command
(`incoming_command_timeout`); `ros2 topic echo /servo_node/status` shows 0 while
it is free to move (2 = singularity stop, 4 = collision stop). The keyboard
stop `ros2 run stem_grasp hotkey_stop_and_zero` (`x`) disables the bridge.
The singularity thresholds (45 / 100) come from the Piper's Jacobian over the
URDF, not from tests on the robot; tune them and the speeds there (P0.5.2).
When a `move_group` runs as well (10.10.6), set
`is_primary_planning_scene_monitor: false` in `servo.yaml`.

### 10.4b MoveIt 2 upstream demo (planning only)

MoveIt is not part of `full_system.launch.py`; the upstream demo starts its
own `robot_state_publisher` with the standalone Piper URDF, and its mock
`joint_state_broadcaster` publishes `/joint_states` (all joints at 0 at
start). Run it with the 10.3 launch stopped: next to the bringup it would
fight the relay over `/joint_states` (it no longer reaches the driver, whose
commands are on `/piper/joint_cmd`).

> **Safety.** Upstream documents this demo as the way to drive the real arm
> together with upstream `start_single_piper.launch.py` (its
> `src/piper_ros/src/piper_moveit/README(EN).md` §3). In that combination the
> arm is commanded to the zero pose at full speed as soon as the demo starts,
> and "Plan & Execute" moves it. Do not use that combination on this robot.

```bash
ros2 launch piper_with_gripper_moveit demo.launch.py
```

**Pass** (planning only, no driver running): the RViz MotionPlanning plugin
plans from home to a manual pose target (not yet validated: P0.3.9).

`moveit_py` has no Humble binary, so `stem_grasp` cannot plan and execute;
the documented options are building moveit2 from source in the container
(~30 min) or `pymoveit2`. Until then the `stem_grasp` pipeline never reaches
its servoing state, so it publishes no twists.

### 10.5 Scout base only

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=false bringup_base:=true bringup_camera:=false \
  bringup_pipeline:=false
```

In a second terminal, with the base on the ground and the area clear:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.1}}' --rate 5
```

Then stop it explicitly: Ctrl-C only stops publishing, it does not command a stop.

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

**Pass:** the Scout creeps forward at 0.1 m/s and halts on the zero twist.

Neither `scout_ros2` (each `/cmd_vel` goes straight to `SetMotionCommand`) nor
`ugv_sdk` has a command timeout, so whether the base stops when commands stop
arriving depends on the Scout firmware, which these sources do not document.
Find out with the wheels off the ground first: publish as above, press Ctrl-C
**without** sending the zero twist, and watch the wheels.

- **TODO(maintainer):** record the result of that test.

### 10.6 RealSense only

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=false bringup_base:=false bringup_camera:=true \
  bringup_pipeline:=false bringup_rviz:=true
```

**Pass:** RViz shows the colour image (`/camera/color/image_raw`) and the
point cloud (`/camera/depth/color/points`). The bringup sets
`camera_namespace:=/`, so topics are `/camera/...`, not `/camera/camera/...`.

### 10.7 All three together

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_arm:=true bringup_base:=true bringup_camera:=true \
  bringup_pipeline:=false bringup_jsp_gui:=false
```

**Pass:** `/joint_states` (from the relay), the camera topics and the base
topics all publish, and `ros2 run tf2_tools view_frames` shows
`odom → base_link → piper_mount_link → piper_base_link → … → camera_link`.
**Fail:** CAN contention — check that the Piper is on `can0` and the Scout on
`can1`.

### 10.8 Nav2 (optional, not validated)

```bash
ros2 launch scout_piper_bringup full_system.launch.py \
  bringup_base:=true bringup_nav2:=true bringup_pipeline:=false
```

**Not usable on this robot as configured.** `scout_nav2` (AIRLab-POLIMI) is
set up for a Scout with an **Ouster 3D lidar**: its AMCL and SLAM-toolbox
parameters read `/ouster/points` and `/ouster/scan` and odometry on
`/odometry` (this robot has no lidar, and `scout_base` publishes `odom`). Its
`nav2.launch.py` loads the simulated warehouse map and parameters unless
`simulation:=false`, and `full_system.launch.py` does not pass that argument;
the real-robot branch expects `maps/airlab/map_lidar3d_v3.yaml`, which the
repository does not contain. Running Nav2 here needs a laser-scan source on
the base (the eye-in-hand camera is unsuitable), a map of your site, and
parameters for both.

To try the plumbing anyway, add the "2D Goal Pose" tool in RViz
(`full_system.rviz` does not include it).

### 10.9 stem_grasp pipeline

`stem_grasp` runs inside the bringup (`bringup_pipeline:=true`, the default)
or on its own, with the camera (and arm) already running:

```bash
ros2 launch stem_grasp stem_grasp.launch.py                          # share/stem_grasp/config/pipeline.yaml
ros2 launch stem_grasp stem_grasp.launch.py config:=/path/to/my_pipeline.yaml
```

Segmentation needs a model: `pipeline.yaml` and `system.yaml` set
`yolo_model_path: /home/sciarm/models/yolo26n-seg.pt`, a lab-machine path;
if the file is missing the node logs a warning and publishes empty masks.
Copy `src/stem_grasp/config/pipeline.yaml`, set `yolo_model_path` (or
`mask_mode: hsv_green`, which needs no model) under `stem_grasp_segmentation`,
and pass the copy with `config:=`. Under `full_system.launch.py` these
segmentation settings are not applied, because `system.yaml` keys them under
`stem_grasp_pipeline`. `mask_mode: grounded_sam` also needs
`groundingdino-py`, `segment-anything` and checkpoints, none of which are
installed.

- **YOLO:** the node loads `yolo_model_path` only if the file exists (no automatic download) and keeps only class `yolo_stem_class_id`. Stock Ultralytics weights have no stem class, so this is the lab's custom-trained stem model. **TODO(maintainer):** record where the trained weights are kept and the class id they use.
- **Grounded-SAM** (public weights): GroundingDINO config `groundingdino/config/GroundingDINO_SwinT_OGC.py` (inside the `groundingdino` package) and checkpoint `https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`; SAM for the default `sam_model_type: vit_b`: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` (from the two projects' READMEs). Set `gdino_config`, `gdino_checkpoint` and `sam_checkpoint` to the downloaded paths.

**Check:**

```bash
ros2 topic echo /stem_grasp/pipeline_state
ros2 topic hz /stem_grasp/mask
ros2 topic echo /stem_grasp/target_pose
```

The pipeline publishes the target pose and markers but does not plan,
execute or servo (P0.4.11). The hot-key helper `ros2 run stem_grasp hotkey_stop_and_zero`
(`x` sends zero twists and disables the servo bridge, `q` quits) stops
servo-driven arm motion only; it is not an emergency stop.

### 10.10 Perception: nvblox, semantic scene, MoveIt semantic collision

Details: [`PHASE1_RUNTIME.md`](Codes/src/scout_piper_scene_repr/docs/PHASE1_RUNTIME.md). In order:

1. **Plumbing test (no nvblox, no hardware).**

   ```bash
   ros2 launch scout_piper_scene_repr test_class_demux.launch.py
   ```

   **Check** (second terminal): `ros2 topic list | grep scene_repr` lists
   `/scene_repr/mask/<class>` and `/scene_repr/depth/<class>`, and
   `ros2 topic hz /scene_repr/mask/stem` shows ~10 Hz.

2. **nvblox built?** `ros2 pkg executables nvblox_ros` lists `nvblox_node`; if
   not, `colcon build --symlink-install --packages-up-to isaac_ros_nvblox`.

3. **RealSense → nvblox (validated with a D405).** Do not run it while the
   bringup camera is on (pass `bringup_camera:=false` to reuse an
   already-running camera).

   ```bash
   ros2 launch scout_piper_scene_repr realsense_nvblox.launch.py
   ```

   **Check:** `ros2 topic hz /camera/aligned_depth_to_color/image_raw` shows
   ~29–30 Hz; `ros2 topic list | grep nvblox_node` shows
   `/nvblox_node/tsdf_layer`, `/nvblox_node/mesh` and
   `/nvblox_node/static_esdf_pointcloud`. In RViz set the fixed frame to
   `camera_link`. Use `initial_reset:=true` only when the driver is stuck.

4. **Per-class semantic nvblox (never run).** Needs the camera, the
   `stem_grasp` segmentation node and an `odom` frame in TF. Use
   `input_mode:=separate`: the segmentation node publishes `/stem_grasp/mask`
   (and `/stem_grasp/target_mask` only in `grounded_sam` mode), not a merged
   label image.

   ```bash
   ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=separate
   ```

   `full_system.launch.py bringup_scene_repr:=true` includes this launch file
   with its default, `input_mode:=merged`.

5. **CPU semantic map and distance field.** No launch file starts this node.
   It needs `class_demux_node` running (item 1 for a synthetic test, item 4
   for live data), aligned depth from the camera, and TF from the camera to
   `odom`; started alone it maps nothing.

   ```bash
   ros2 run scout_piper_scene_repr scene_query_node.py --ros-args \
     -p world_frame:=odom -p grid_center:="[0.6, 0.0, 0.6]" -p grid_half_extent_m:=0.4
   ros2 topic hz /scene_repr/distance_field          # about field_rate_hz (1 Hz)
   ```

6. **MoveIt semantic collision plugin (not run on the robot).** Give the
   `move_group` node `collision_detector: "Semantic"` (in its launch file's
   parameter dict, or in a params file as below) and the plugin parameters
   from `config/semantic_collision.yaml`:

   ```yaml
   move_group:
     ros__parameters:
       collision_detector: "Semantic"
   ```

   ```bash
   --ros-args --params-file $(ros2 pkg prefix scout_piper_scene_repr)/share/scout_piper_scene_repr/config/semantic_collision.yaml
   ```

   **Check:** the `move_group` log shows `Listening for the semantic distance
   field`. With `require_field: true` (the default) every plan is refused
   until `/scene_repr/distance_field` arrives and TF connects the robot model
   frame to `odom`.

   - The unified robot now has an SRDF (`scout_piper_bringup/config/moveit/scout_piper.srdf`, used by servo), but the repository has no `move_group` launch for it yet (planning pipeline and controller configuration missing). **TODO(maintainer):** add one when planning is needed; set `is_primary_planning_scene_monitor: false` in `servo.yaml` when both run.

### 10.11 plant_twin

```bash
ros2 launch plant_twin plant_twin.launch.py      # config:= defaults to share/plant_twin/config/plant_twin.yaml
```

Inputs: `/stem_grasp/leaf_filtered_cloud`, `/stem_grasp/filtered_cloud` and
`/stem_grasp/target_mask`; aligned depth and colour camera info; `/joint_states`
and TF `piper_base_link` → `piper_link7`; `/ft_sensor/raw` (WrenchStamped).
It initialises on the first frame with clouds, mask, depth and intrinsics.
`/stem_grasp/target_mask` and `/stem_grasp/leaf_filtered_cloud` exist only
when segmentation runs `mask_mode: grounded_sam` with `target_caption` set,
so with YOLO or `hsv_green` the node never initialises.

**Check:** `ros2 topic echo /plant_twin/leaf_tip` and the `/plant_twin/markers`
MarkerArray in RViz. Untested on hardware.

- Nothing publishes `/ft_sensor/raw`: the platform has no wrist force/torque sensor ([research platform facts](research/README.md)). Without it `plant_twin` never sees a pull force, and force-based gates stay inactive until a sensor or a joint-effort estimate is added.

### 10.12 Piper-JEPA target state node

```bash
ros2 launch scout_piper_jepa target_state.launch.py   # config:= defaults to share/scout_piper_jepa/config/target_memory.yaml
```

The node tracks nothing until a mono8 mask arrives on `/piper_jepa/init_mask`;
each new mask re-grounds the target, and once the state is `lost` it needs a
new one. `/piper_jepa/target_point` also needs aligned depth and TF from
`odom` to the camera optical frame.

**Check:** `ros2 topic echo /piper_jepa/target_state` (JSON) and
`ros2 topic echo /piper_jepa/target_visible`.

The default encoder, `color_patch`, is a numpy reference, not V-JEPA. To use
V-JEPA, set `encoder: vjepa` in a copy of the config and retune
`visible_similarity`. The config already names the hub entry
`vjepa2_1_vit_base_384` (V-JEPA 2.1 ViT-B/16 at 384 px, from vjepa2's
`hubconf.py`); larger ones are `vjepa2_1_vit_large_384` and
`vjepa2_1_vit_giant_384`. It needs `torch>=2`, `timm` and `einops`
(`python3 -m pip install --user timm einops`), downloads the hub repository
and checkpoint on first use, and has never been run on this robot.

- **TODO(maintainer):** there is no grounding tool yet (the operator GUI is Phase 5); any mono8 mask the size of the colour image works. Decide what publishes it for experiments.

### 10.13 Whole-body MPC (dry run)

```bash
ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py   # execute: false by default
```

The node stays silent until it has `/odom`, `/joint_states` containing every
name in `joint_names` (default `piper_joint1` … `piper_joint6`), and a goal on
`/whole_body_mpc/goal` in `odom`. In dry run it publishes only
`/whole_body_mpc/preview/cmd_vel`, `/whole_body_mpc/preview/joint_jog`,
`/whole_body_mpc/status` and `/whole_body_mpc/plan`. On the robot the arm
state comes from the bringup's relay (10.3).

Plumbing check with stub inputs, without the arm bringup (terminal 1 publishes
`/joint_states`, which the relay owns when the arm is on):

```bash
ros2 launch scout_piper_description view_robot.launch.py use_gui:=false                 # terminal 1: /joint_states
ros2 topic pub -r 20 /odom nav_msgs/msg/Odometry "{header: {frame_id: odom}}"           # terminal 2: base at the origin
ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py                         # terminal 3
ros2 topic pub --once /whole_body_mpc/goal geometry_msgs/msg/PointStamped \
  "{header: {frame_id: odom}, point: {x: 0.7, y: 0.0, z: 0.45}}"                        # terminal 4: example goal
ros2 topic echo /whole_body_mpc/status
```

**Check:** the status JSON shows `"execute": false` and a `mode` field; a
`safety` value of `watchdog` means an input is older than `max_state_age_s`
(0.2 s).

> **Safety.** Set `execute: true` only after the Phase 0 stop validation and
> after servo has been checked on the robot (10.4). With `execute: true` the
> base commands go straight to `/cmd_vel`, and the arm's `JointJog` goes to
> `/servo_node/delta_joint_cmds`, which moves the arm only while servo is
> started and the bridge enabled.

## Research quick start without ROS (path D)

The numpy cores of `scout_piper_scene_repr`, `plant_twin`, `scout_piper_jepa`
and `scout_piper_whole_body_mpc` run without ROS; the ROS nodes, `stem_grasp`
and the rosbag export do not.

**Where:** any machine, from the repository root. **sudo:** only for
`python3-venv` (Step 1) and, optionally, a C++ compiler.

Use Python 3.10–3.12: `numpy<2` has prebuilt wheels only up to Python 3.12.
On 3.13 or newer, pip compiles NumPy 1.26 from source (a few minutes), which
needs a compiler and the Python headers first (e.g.
`sudo apt-get install -y build-essential python3.13-dev`).

```bash
python3 -m venv .venv                     # .venv/ is gitignored
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy<2' scipy pyyaml pytest opencv-python-headless
```

- `scipy` is needed by all four suites (`scout_piper_jepa` uses it in its tests without declaring it); `pyyaml` reads the scene policy files; `opencv-python-headless` enables `plant_twin`'s outline test (skipped without it).
- Optional, torch (the two torch tests in `scout_piper_jepa`, and the E3 benchmark): `python -m pip install torch`. On Linux this installs the CUDA build (~4–5 GB); on a CPU-only machine use PyTorch's CPU index instead: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Optional: a C++17 compiler (`sudo apt-get install -y build-essential`) enables the C++ sampler check in `scout_piper_scene_repr` (skipped otherwise).
- Versions known to work here: torch 2.14.1 (PyPI, Python 3.13) and the conda-forge CPU build used for the E3 benchmark. The Jetson wheel source is part of the Path A TODO.

Run the tests from each package directory with `python -m pytest` (plain
`pytest`, or running from `Codes/`, fails with `ModuleNotFoundError`):

```bash
cd Codes/src
( cd scout_piper_bringup        && python -m pytest test -q )
( cd scout_piper_scene_repr     && python -m pytest test -q )
( cd plant_twin                 && python -m pytest test -q )
( cd scout_piper_jepa           && python -m pytest test -q )
( cd scout_piper_whole_body_mpc && PYTHONPATH=../scout_piper_scene_repr/python python -m pytest test -q )
```

Results at the time of writing (4-core x86_64, Python 3.13, numpy 1.26.4, scipy 1.17.1):

| Package | Result | Time |
|---|---|---|
| `scout_piper_bringup` | 9 passed, 9 skipped without `launch_ros` (18 passed with ROS) | <1 s |
| `scout_piper_scene_repr` | 19 passed | ~20 s |
| `plant_twin` | 21 passed, 1 failed (20 passed, 2 failed on a busy machine) | ~10–20 s |
| `scout_piper_jepa` | 15 passed, 2 skipped without torch; 17 passed with torch | ~5 s (~15–50 s with torch) |
| `scout_piper_whole_body_mpc` | 18 passed | ~50 s |

The `plant_twin` failures are known: `test_solver.py::test_linear_problem_solves_in_one_step`
reports `stalled` instead of `converged` with numpy 1.26.4 / scipy 1.17.1 (it
passes with numpy 2.x), and `test_jacobian.py::test_analytic_fit_matches_numeric_and_is_faster`
asserts a wall-clock speed-up, which fails when the machine is busy.

Synthetic benchmarks (offline; not robot evidence), still in `Codes/src`:

```bash
# Whole-body MPC reachability: W0 arm-only, W1 sequential, W3 unified on scenes R1-R3 and O1
( cd scout_piper_whole_body_mpc && PYTHONPATH=. python benchmarks/reachability.py --seeds 1 --steps 10 )   # quick smoke run
( cd scout_piper_whole_body_mpc && PYTHONPATH=. python benchmarks/reachability.py --seeds 3 )              # full run

# Piper-JEPA E3 predictor comparison (needs torch; ~1-1.5 h on a 4-core CPU). Keep the output outside the repository.
PYTHONPATH=scout_piper_jepa:scout_piper_whole_body_mpc \
  python scout_piper_jepa/benchmarks/e3_synthetic.py --steps 1500 --out /tmp/e3.json

# Piper-JEPA closed-loop visibility, C2 vs C3-oracle (~2 s per control step on CPU; --model needs torch)
PYTHONPATH=scout_piper_jepa:scout_piper_whole_body_mpc \
  python scout_piper_jepa/benchmarks/visibility_mpc.py --seeds 3
```

Compare the output with the tables in the
[`scout_piper_whole_body_mpc`](Codes/src/scout_piper_whole_body_mpc/README.md)
and [`scout_piper_jepa`](Codes/src/scout_piper_jepa/README.md) READMEs.
Offline tracking evaluation on recorded episodes is
`python -m scout_piper_jepa.episode eval ep1.npz --encoder color_patch`, run
from `Codes/src/scout_piper_jepa`; exporting episodes from a bag needs ROS 2,
and no sample episodes are in the repository yet.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Arm jumps to the zero pose at full speed | A command on the driver's input: with upstream `start_single_piper.launch.py` that is any `/joint_states` publisher (sliders, `view_robot`, MoveIt demo, a stub); with the bringup, anything on `/piper/joint_cmd` | Cut power. Start the arm only through `full_system.launch.py` and check `ros2 topic info /piper/joint_cmd` ([warning](#before-you-start-what-works-today)) |
| `arm_command_topic:=... would make the Piper driver execute joint states as commands` | A joint-state topic was given as the command topic | Use a dedicated topic (default `/piper/joint_cmd`) |
| RViz arm does not follow the real arm | Relay not receiving `/joint_states_single` | `ros2 topic hz /joint_states_single`; the relay logs a warning if `joint1..joint6` are missing |
| `ModuleNotFoundError: No module named 'piper_sdk'` | Piper driver's pip dependencies missing | `python3 -m pip install --user piper_sdk python-can` (3B.3; in the container see 3C.3) |
| `can_activate.sh`: `ethtool not detected` / `can-utils not detected` | Missing apt packages | `sudo apt-get install -y can-utils ethtool` |
| `can_activate.sh` asks for the USB hardware address | More than one CAN adapter plugged in | Pass the bus-info from `find_all_can_port.sh` as the third argument (9.1) |
| CAN bus contention | Both robots on one bus | Piper on `can0`, Scout on `can1` (set in `full_system.launch.py`) |
| `install_docker_nvidia.sh` prints `Do not run this as root` | It calls `sudo` itself | Run it as your normal user |
| `permission denied` on the Docker socket | `docker` group not active yet | Log out and back in, or `newgrp docker` |
| Unknown runtime `nvidia` | NVIDIA Container Toolkit missing | `./scripts/install_docker_nvidia.sh`, then the `nvidia-smi` check |
| RViz cannot open a display in the container; `~/.Xauthority` is a directory | The file did not exist at the first `run` | `sudo rmdir ~/.Xauthority`, `xauth extract ~/.Xauthority "$DISPLAY"`, run again (6C) |
| Packages installed in the container are gone | `run --rm` discards the container | Add them to `docker/Dockerfile.dev` and run `./docker/build_dev.sh` |
| `vcs: command not found` on the 20.04 host | pip `--user` scripts not on `PATH` | Prefix `PATH=$HOME/.local/bin:$PATH` |
| `ld: libgxf_core.so: file format not recognized; treating as linker script` | GXF LFS pointer files (`git-lfs` missing) | `./scripts/install_git_lfs_and_pull.sh`, `./scripts/patch_upstream.sh`, then `colcon build --symlink-install --packages-up-to isaac_ros_nvblox` |
| `patch_upstream.sh` stops in the NITROS patches | Upstream sources changed; the script exits before the LFS pull and submodule init | Check `repos.yaml` still pins `release-3.2`, then run `./scripts/install_git_lfs_and_pull.sh` and `git -C src/isaac_ros_nvblox submodule update --init --recursive` by hand |
| `scout_base` / `ugv_sdk` build errors | Patches not applied | `./scripts/patch_upstream.sh` |
| Missing dependencies after `./build_workspace.sh` | It skips `rosdep init` / `update` and hides `rosdep install` failures | Run the explicit commands in 6B |
| rosdep `ERROR ... could not have their rosdep keys resolved` | Expected for `nvblox_examples_bringup` only | Any other package listed is a real missing dependency (6B) |
| `realsense2_camera` configure error about the RealSense SDK | librealsense2 older than the imported branch needs (≥ 2.59.0) | Upgrade librealsense2 or pin `realsense-ros` in `repos.yaml` (9.2) |
| `nvcc not found` | CUDA not on `PATH` | `source /etc/bash.bashrc` (container) or `export PATH=/usr/local/cuda-12.4/bin:$PATH` |
| `find_package(vpi)` fails | VPI CMake config not found | ``export CMAKE_PREFIX_PATH=$(find /opt/nvidia -name 'vpiConfig.cmake' -printf '%h\n' \| head -1):$CMAKE_PREFIX_PATH`` |
| `find_package(CUDAToolkit)` errors | CMake too old | CMake ≥ 3.22 (the jammy apt version is fine) |
| `isaac_ros_image_pipeline` missing | Newer nvblox pulls in more Isaac ROS packages | Add it to `repos.yaml`, run `vcs import` and `patch_upstream.sh` again |
| `vcs import` cannot find an `isaac_ros_*` tag | Tag removed upstream | `release-3.2` is the validated pin; try another `release-3.x` only if it disappears |
| `./scripts/install_isaac_ros_apt.sh` exits with an error | Obsolete: no Isaac ROS apt packages for Humble | Use the source build (Steps 4–6) |
| `install_vpi.sh` refuses to run | Needs `/opt/ros/humble` and jammy | Not needed in the dev container (the image has VPI 4) |
| `_ARRAY_API not found` from `cv_bridge` / `sensor_msgs_py` | NumPy 2 in the ROS environment | `python3 -m pip install --user 'numpy<2'` |
| `colcon test` fails in `scout_piper_scene_repr` lint tests | `ament_lint_auto` (copyright, cpplint, …) | Expected for now; run the functional tests with the `--ctest-args -R` filter (Step 8) |
| Segmentation publishes empty masks | `yolo_model_path` missing, or settings not applied under `full_system.launch.py` | `stem_grasp.launch.py config:=` with a valid model path or `hsv_green` (10.9) |
| Topics under `/camera/camera/...` | Camera namespace not set | Launch through the bringup or `realsense_nvblox.launch.py` (both set `camera_namespace:=/`) |
| RealSense driver stuck | Device state | Relaunch with `initial_reset:=true`; a D405 can take seconds to re-enumerate |
| `nvblox_semantic.launch.py` produces nothing | Default `input_mode:=merged` has no producer, or no `odom` frame | `input_mode:=separate`, and make sure `odom` exists in TF |
| `move_group` refuses every plan with the Semantic plugin | No distance field, or no TF to `odom` | Run `class_demux_node` + `scene_query_node.py` with depth and TF (10.10.5); check `ros2 topic hz /scene_repr/distance_field` |
| `plant_twin` never initialises | No `/stem_grasp/target_mask` / leaf cloud outside `grounded_sam` mode | See 10.11 |
| Whole-body MPC publishes nothing | Missing `/odom`, `/joint_states` with `piper_joint*` names (the relay, 10.3), or a goal in `odom` | Check the inputs in 10.13 |
| JEPA node publishes nothing | No grounding mask yet, or state `lost` | Publish a mono8 mask on `/piper_jepa/init_mask` |
| `ModuleNotFoundError` for `scout_piper_jepa` / `plant_twin` in tests | Wrong directory, or plain `pytest` | `cd` into the package and use `python -m pytest test -q` |
| `pip install 'numpy<2'` compiles for minutes or fails | Python 3.13+ has no NumPy 1.x wheels | Use Python 3.10–3.12, or install a compiler and the Python headers (path D) |
| `python3 -m venv` reports `ensurepip is not available` | Ubuntu splits venv into its own package | `sudo apt-get install -y python3-venv` |
| `setup_env.sh` has no effect | Executed instead of sourced | `source setup_env.sh` from bash |
| Upstream launch file or xacro not found after `vcs import` | Upstream layout changed | Fix the references listed in Step 4; regenerate `_piper_arm.xacro` with `./scripts/fork_piper_arm.py` rather than editing it |

## Where to go next

- [`Codes/README.md`](Codes/README.md): workspace layout, upstream packages, decisions locked for Phase 0.
- [`Codes/PHASE0_CHECKLIST.md`](Codes/PHASE0_CHECKLIST.md): the full Phase 0 bring-up and porting checklist (calibration, servo wiring, ROS 1 regression, Nav2, sign-off).
- [`Codes/docker/README.md`](Codes/docker/README.md): dev container details.
- [`Codes/src/scout_piper_scene_repr/docs/PHASE1_RUNTIME.md`](Codes/src/scout_piper_scene_repr/docs/PHASE1_RUNTIME.md): nvblox source build and Phase 1 runtime.
- Package READMEs: [`stem_grasp`](Codes/src/stem_grasp/README.md), [`scout_piper_scene_repr`](Codes/src/scout_piper_scene_repr/README.md), [`plant_twin`](Codes/src/plant_twin/README.md), [`scout_piper_jepa`](Codes/src/scout_piper_jepa/README.md), [`scout_piper_whole_body_mpc`](Codes/src/scout_piper_whole_body_mpc/README.md).
- [`PROGRESS.md`](PROGRESS.md) for task status and blockers, [`ROADMAP.md`](ROADMAP.md) for the plan.
