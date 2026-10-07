# Phase 1 Runtime Setup

How to actually run the Phase 1 semantic-SDF stack on this machine.

The companion design doc is [`PHASE1_DESIGN.md`](PHASE1_DESIGN.md).

## What's needed

| Component | Where it runs | Status |
|---|---|---|
| `class_demux_node` (Python) | Our dev container | ☑ Built, fully testable today |
| Synthetic mask publisher (test) | Our dev container | ☑ For plumbing tests without nvblox |
| `nvblox_node` (Isaac ROS) | Our dev container, **built from source** | ☑ Built and validated with RealSense D405 |
| `semantic_collision_plugin` (C++) | MoveIt 2's move_group | ◐ Scaffold builds; reports no collision / infinite distance until wired to the semantic query or nvblox ESDF (P1.3.1 / P1.7.7) |

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
cd Codes          # from the repository root
git pull
./docker/build_dev.sh
```

This adds CUDA 12.4 toolkit (~1.5 GB pull) + libgflags-dev, libgoogle-glog-dev,
libsqlite3-dev, libbenchmark-dev, libgtest-dev, libgmock-dev. ~10 min.

### 2b — Pull the Isaac ROS source repos

```bash
cd Codes          # from the repository root
PATH=$HOME/.local/bin:$PATH vcs import src < repos.yaml
./scripts/patch_upstream.sh
```

This adds the Isaac ROS source trees pinned in `repos.yaml` (`release-3.2`) to `src/`:
- `isaac_ros_nvblox` (the ROS 2 wrapper; the nvblox CUDA TSDF/ESDF library, ~50 MB, is a submodule at `nvblox_ros/nvblox_core` that `patch_upstream.sh` initialises)
- `isaac_ros_common`, `isaac_ros_nitros`, `isaac_ros_gxf` (repo `gxf`) and `negotiated` (support and transport dependencies)

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
ros2 pkg executables nvblox_ros
```

### 2d — Smoke test nvblox

Use a sample dataset (one of the publicly available kitti / euroc bags) or
the included tests:

```bash
ros2 run nvblox_ros nvblox_node --ros-args -p use_tf_transforms:=false
```

This only verifies that the executable can start; the live reconstruction check
is Step 2e.

### 2e — RealSense D405 -> nvblox smoke test (P1.1.2)

This is the validated single-camera reconstruction path. It anchors the map in
`camera_link` so the test does not depend on VSLAM, odometry, Nav2, or the
Scout TF tree.

```bash
# Inside the dev container:
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch scout_piper_scene_repr realsense_nvblox.launch.py
```

Verify the camera input and nvblox outputs:

```bash
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic list | grep nvblox_node
```

Expected observations on this workstation:

- `/camera/aligned_depth_to_color/image_raw` runs around 29-30 Hz.
- nvblox publishes `/nvblox_node/tsdf_layer`, `/nvblox_node/mesh`,
  `/nvblox_node/static_esdf_pointcloud`, and related debug topics.
- nvblox logs show GPU TSDF block allocation after the first live depth frames.

Use `initial_reset:=true` only when the RealSense driver is stuck; after a
reset the D405 can take several seconds to re-enumerate.

## Step 3 — Wire class_demux -> 4x nvblox

Phase 1 v0 design (see `PHASE1_DESIGN.md` §5): instantiate four nvblox
nodes, each consuming the mask-gated depth from one class. Drafted in
[`launch/nvblox_semantic.launch.py`](../launch/nvblox_semantic.launch.py) but not yet run (P1.2.1). It still
launches `isaac_ros_nvblox`/`nvblox_node` with the older `depth/image`-style remaps, whereas the validated
Step 2e path uses `nvblox_ros`/`nvblox_node` with `camera_0/*` remaps:

```bash
ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=merged
# or, while segmentation_node still publishes /stem_grasp/mask + /stem_grasp/target_mask:
ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=separate
```

## Likely build issues + workarounds

| Symptom | Cause | Fix |
|---|---|---|
| `nvcc not found` after rebuild | PATH didn't pick up CUDA | `source /etc/bash.bashrc` or `export PATH=/usr/local/cuda-12.4/bin:$PATH` |
| `vcs import` cannot find an isaac_ros_* tag | Tag removed upstream | `repos.yaml` pins `release-3.2` (last Humble-targeted tag, validated in P1.1.1); try another `release-3.x` tag only if that one disappears |
| nvblox cmake errors on `find_package(CUDAToolkit)` | CMake too old | Apt-installed cmake ≥ 3.22 should be fine; older systems may need cmake from pip |
| Missing `gtest` | Build deps incomplete | Already added in Dockerfile rev with libgtest-dev/libgmock-dev |
| isaac_ros_image_pipeline dep missing | Newer nvblox pulls more isaac deps | Add to repos.yaml, vcs import again |

## Phase 1 task tracking

Live in [`../../../../PROGRESS.md`](../../../../PROGRESS.md) under "Phase 1".
