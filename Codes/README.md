# Piper + Scout ROS 2 Workspace (Phase 0)

ROS 2 **Humble** colcon workspace integrating the AgileX Piper 6-DoF arm on a Scout 2.0 UGV, ported from the existing ROS 1 Noetic stack (`stem_grasp_ros1`, kept in the original `piper_ros` workspace on the lab machine; not part of this repository).

This is the **Phase 0 deliverable** of the [`../ROADMAP.md`](../ROADMAP.md); it also hosts the Phase 1
(`scout_piper_scene_repr`, `plant_twin`), Phase 2A (`scout_piper_jepa`) and Phase 3A
(`scout_piper_whole_body_mpc`) packages.

## Layout

```
Codes/
├── README.md               # this file
├── PHASE0_CHECKLIST.md     # step-by-step migration checklist
├── repos.yaml              # vcs-import manifest for upstream packages
├── build_workspace.sh      # rosdep + colcon build helper
├── setup_env.sh            # source to activate the workspace
├── docker/                 # dev container (Dockerfile.dev, compose.dev.yml, build_dev.sh)
├── scripts/                # host setup + post-import patches (patch_upstream.sh, fork_piper_arm.py, ...)
├── .gitignore
└── src/
    ├── plant_twin/                 # NEW: deformable leaf + stem twin fitted during grasps (Phase 1B)
    ├── scout_piper_bringup/        # NEW: integrated system launch + system config
    ├── scout_piper_description/    # NEW: unified URDF (Piper on Scout) + RViz
    ├── scout_piper_jepa/           # NEW: Piper-JEPA Stage A dense target memory (Phase 2A)
    ├── scout_piper_scene_repr/     # NEW: semantic scene — class demux, nvblox launch, CPU semantic map + distance query (Phase 1A)
    ├── scout_piper_whole_body_mpc/ # NEW: geometry-only whole-body MPC + safety filter, dry run by default (Phase 3A)
    └── stem_grasp/                 # NEW: ROS 2 port of stem_grasp_ros1 (rclpy)
```

`full_system.launch.py` includes `scout_piper_scene_repr` behind `bringup_scene_repr:=false`;
`plant_twin`, `scout_piper_jepa` and `scout_piper_whole_body_mpc` are started from their own launch files.

Upstream packages will be cloned into `src/` by `vcs import`:
- `piper_ros/` from `agilexrobotics/piper_ros@humble` (arm driver, ros2_control, MoveIt 2 config)
- `scout_ros2/` from `agilexrobotics/scout_ros2@humble` (Scout base CAN driver)
- `scout_nav2/` from `AIRLab-POLIMI/scout_nav2` (Nav2 stack tuned for Scout)
- `realsense-ros/` from `IntelRealSense/realsense-ros@4.58.4` (camera driver; matches ROS Humble's librealsense2 2.58)
- `ugv_sdk/` from `westonrobot/ugv_sdk@main` (CAN SDK used by `scout_base`)
- Phase 1 nvblox source build (`release-3.2`, the last Humble tags): `isaac_ros_common/`, `isaac_ros_nvblox/`,
  `isaac_ros_nitros/`, `isaac_ros_gxf/` (repo `NVIDIA-ISAAC-ROS/gxf`), plus `negotiated/` from `osrf/negotiated@master`

Run `./scripts/patch_upstream.sh` after every `vcs import`.

## Quick start

The workstation runs Ubuntu 20.04, which doesn't have apt packages for ROS 2
Humble. Everything below runs inside a Docker dev container (see
[`docker/README.md`](docker/README.md)).

```bash
cd Codes                     # from the repository root

# 1. (once) Install Docker + nvidia-container-toolkit on the host
./scripts/install_docker_nvidia.sh

# 2. (once) Pull upstream packages (no sudo, no ROS 2 needed)
PATH=$HOME/.local/bin:$PATH vcs import src < repos.yaml
./scripts/patch_upstream.sh   # post-import fixes; LFS assets need git-lfs (./scripts/install_git_lfs_and_pull.sh)

# 3. (once) Build the dev container
./docker/build_dev.sh

# 4. Drop into the container — workspace bind-mounted at /workspace
docker compose -f docker/compose.dev.yml run --rm dev

# Inside the container:
colcon build --symlink-install
source install/setup.bash
ros2 launch scout_piper_description view_robot.launch.py     # URDF viz
ros2 launch scout_piper_bringup full_system.launch.py        # hardware-free defaults; add bringup_arm/base/camera:=true on the rig
```

See [`PHASE0_CHECKLIST.md`](PHASE0_CHECKLIST.md) for the full migration plan.

## Decisions locked for Phase 0

| Choice | Value | Rationale |
|---|---|---|
| ROS 2 distro | **Humble** | AgileX official branch is `humble`; EOL May 2027 gives 12 months. |
| Vendoring | **vcs import** via `repos.yaml` | Standard ROS 2 way; cheap to update versions. |
| Build target | **Workstation first**, Jetson Orin AGX in Month 2 | Faster iteration; cross-build later via Docker or native rebuild. |
| Arm driver | `agilexrobotics/piper_ros@humble` | Official, ros2_control + MoveIt 2 included. |
| Base driver | `agilexrobotics/scout_ros2` + POLIMI `scout_nav2` | Official CAN driver + community Nav2 tuning. |
| Camera | `realsense-ros@4.58.4` (D405) | Upstream; no functional change vs ROS 1. Pinned to the release matching ROS Humble's librealsense2. |

## Cross-references

- ROS 1 source we are porting: `stem_grasp_ros1` (in the original `piper_ros` workspace on the lab machine; not part of this repository)
- Original launch (parameter source-of-truth): `stem_grasp_ros1/launch/stem_grasp_ros1.launch` (same workspace); its arguments are mirrored in [`src/scout_piper_bringup/config/system.yaml`](src/scout_piper_bringup/config/system.yaml)
- ROS 1 pipeline node: `stem_grasp_ros1/scripts/pipeline_node.py` (same workspace); ported to [`src/stem_grasp/stem_grasp/pipeline_node.py`](src/stem_grasp/stem_grasp/pipeline_node.py)
- Roadmap: [`../ROADMAP.md`](../ROADMAP.md)
