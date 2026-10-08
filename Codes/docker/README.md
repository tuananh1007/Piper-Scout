# Docker dev environment

The workstation runs Ubuntu 20.04, which has no `apt` packages for ROS 2 Humble.
This dev container provides a pre-baked Humble + MoveIt 2 + ros2_control + nav2 +
RealSense environment, plus the CUDA 12.4 / VPI 4 build deps for nvblox, so the
workspace can be built and run on the host machine.

## One-time setup

```bash
# 1. Install Docker + nvidia-container-toolkit (host, requires sudo)
cd Codes                     # from the repository root
./scripts/install_docker_nvidia.sh

# 2. Log out and back in (or `newgrp docker`) so your user picks up the docker group

# 3. Build the dev image (~3 GB download for the base image, plus the ROS/Python, CUDA 12.4 toolkit and VPI 4 layers)
./docker/build_dev.sh
```

## Daily use

```bash
cd Codes                     # from the repository root
docker compose -f docker/compose.dev.yml run --rm dev
# you are now inside the container, in /workspace

# First time: build the workspace (memory-limited, INSTALL.md Step 6)
./scripts/colcon_build_safe.sh --symlink-install
source install/setup.bash

# Then run the bringup or just visualize the URDF:
ros2 launch scout_piper_description view_robot.launch.py
ros2 launch scout_piper_bringup full_system.launch.py bringup_camera:=false bringup_base:=false
```

The container is configured for:
- GPU passthrough (RealSense + YOLO + servo math + nvblox).
- X11 forwarding (RViz works from inside the container).
- Host network mode (DDS auto-discovery with anything else on the LAN, incl. the
  Jetson Orin AGX once it's online).
- IPC=host (shared memory transport for high-rate topics).
- USB passthrough (RealSense + Piper CAN adapter + Scout CAN adapter).
- Workspace bind-mounted from the host — edit files on the host with your editor,
  the container sees changes immediately.

## Image notes

The Dockerfile pre-installs the heavy ROS 2 + Python deps so colcon build is fast.
Notably absent (install lazily): `groundingdino-py`, `segment-anything`. These
pull torch + a 2 GB checkpoint set; only needed when `mask_mode=grounded_sam`.

## Isaac ROS / nvblox (Phase 1)

NVIDIA no longer ships Isaac ROS apt packages for Humble on jammy, so nvblox is
built from source inside this container: the Dockerfile adds the CUDA 12.4
toolkit, VPI 4 and `magic_enum`, and `repos.yaml` pulls the `release-3.2` Isaac
ROS sources (run `./scripts/patch_upstream.sh` after `vcs import`). Build and
smoke-test steps: [`../src/scout_piper_scene_repr/docs/PHASE1_RUNTIME.md`](../src/scout_piper_scene_repr/docs/PHASE1_RUNTIME.md).
