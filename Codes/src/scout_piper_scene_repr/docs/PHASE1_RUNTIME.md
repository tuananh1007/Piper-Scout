# Phase 1 Runtime Setup

How to actually run the Phase 1 semantic-SDF stack on this machine.

The companion design doc is [`PHASE1_DESIGN.md`](PHASE1_DESIGN.md).

## What's needed

| Component | Where it runs | Status |
|---|---|---|
| `class_demux_node` (Python) | Our dev container | ☑ Built, fully testable today |
| Synthetic mask publisher (test) | Our dev container | ☑ For plumbing tests without nvblox |
| `nvblox_node` (Isaac ROS) | Our dev container, **built from source** | ◐ Source build path — see Step 2 below |
| `semantic_collision_plugin` (C++) | MoveIt 2's move_group | ☑ Built; needs nvblox ESDF API wiring (P1.4) |

## Why source build, not apt

NVIDIA dropped the Ubuntu 22.04 (jammy) + Humble apt builds for Isaac ROS
during Q4 2025. The only live apt repository now is `release-4.0` on Ubuntu
24.04 (noble) + ROS 2 Jazzy. Our dev container is Humble on jammy, so apt
install is not an option. We build from source instead.

The Dockerfile.dev now ships CUDA 12.4 dev libs + glog + gflags so the
nvblox source build can run inside our standard dev container.

## Step 1 — Plumbing smoke test (no nvblox needed)

Validates that synthetic mask → class_demux → per-class topics works.
**Do this first.**

```bash
# Host:
docker compose -f docker/compose.dev.yml run --rm dev

# Inside the container:
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select scout_piper_scene_repr
source install/setup.bash
ros2 launch scout_piper_scene_repr test_class_demux.launch.py
```

In another container shell, verify:

```bash
ros2 topic list | grep scene_repr
ros2 topic hz /scene_repr/mask/stem        # ~10 Hz
rqt &                                       # Image View → /scene_repr/mask/leaf
```

## Step 2 — Build nvblox from source

### 2a — Rebuild the dev container (CUDA 12.4 + nvblox build deps)

```bash
# Host:
cd ~/agilex/piper_ros/Piper_Scout_ws/Codes
git pull
./docker/build_dev.sh
```

This adds CUDA 12.4 toolkit (~1.5 GB pull) + libgflags-dev, libgoogle-glog-dev,
libsqlite3-dev, libbenchmark-dev, libgtest-dev, libgmock-dev. ~10 min.

### 2b — Pull the Isaac ROS source repos

```bash
cd ~/agilex/piper_ros/Piper_Scout_ws/Codes
PATH=$HOME/.local/bin:$PATH vcs import src < repos.yaml
./scripts/patch_upstream.sh
```

This adds three new source trees to `src/`:
- `nvblox` (~50 MB, the CUDA TSDF/ESDF library)
- `isaac_ros_nvblox` (~10 MB, the ROS 2 wrapper)
- `isaac_ros_common` (~5 MB, support utilities)

### 2c — Build inside the container

```bash
# Host:
docker compose -f docker/compose.dev.yml run --rm dev

# Inside:
source /opt/ros/humble/setup.bash
nvcc --version                # verify CUDA dev toolchain
colcon build --symlink-install --packages-up-to isaac_ros_nvblox
```

The build of nvblox (CUDA kernels) takes 15-30 min on a workstation. Plan
accordingly. After it completes:

```bash
source install/setup.bash
ros2 pkg list | grep nvblox
ros2 pkg executables isaac_ros_nvblox
```

### 2d — Smoke test nvblox

Use a sample dataset (one of the publicly available kitti / euroc bags) or
the included tests:

```bash
ros2 run isaac_ros_nvblox nvblox_node --ros-args -p use_static_tf:=true
```

Look for `/nvblox_node/static_esdf_layer` or similar topics appearing.

## Step 3 — Wire class_demux → 4× nvblox

Phase 1 v0 design (see `PHASE1_DESIGN.md` §5): instantiate four nvblox
nodes, each consuming the mask-gated depth from one class. Already wired in
[`launch/nvblox_semantic.launch.py`](../launch/nvblox_semantic.launch.py):

```bash
ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=merged
```

## Likely build issues + workarounds

| Symptom | Cause | Fix |
|---|---|---|
| `nvcc not found` after rebuild | PATH didn't pick up CUDA | `source /etc/bash.bashrc` or `export PATH=/usr/local/cuda-12.4/bin:$PATH` |
| isaac_ros_nvblox doesn't have a humble tag | release-3.x deleted | Try `release-3.0`, `release-3.1`, or the `humble` branch |
| nvblox cmake errors on `find_package(CUDAToolkit)` | CMake too old | Apt-installed cmake ≥ 3.22 should be fine; older systems may need cmake from pip |
| Missing `gtest` | Build deps incomplete | Already added in Dockerfile rev with libgtest-dev/libgmock-dev |
| isaac_ros_image_pipeline dep missing | Newer nvblox pulls more isaac deps | Add to repos.yaml, vcs import again |

## Phase 1 task tracking

Live in [`../../../../PROGRESS.md`](../../../../PROGRESS.md) under "Phase 1".
