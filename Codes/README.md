# Piper + Scout ROS 2 Workspace (Phase 0)

ROS 2 **Humble** colcon workspace integrating the AgileX Piper 6-DoF arm on a Scout 2.0 UGV, ported from the existing ROS 1 Noetic stack at [`../../src/`](../../src/).

This is the **Phase 0 deliverable** of the [`../ROADMAP.md`](../ROADMAP.md).

## Layout

```
Codes/
├── README.md               # this file
├── PHASE0_CHECKLIST.md     # step-by-step migration checklist
├── repos.yaml              # vcs-import manifest for upstream packages
├── .gitignore
└── src/
    ├── scout_piper_bringup/        # NEW: integrated system launch + system config
    ├── scout_piper_description/    # NEW: unified URDF (Piper on Scout) + RViz
    └── stem_grasp/                 # NEW: ROS 2 port of stem_grasp_ros1 (rclpy)
```

Upstream packages will be cloned into `src/` by `vcs import`:
- `piper_ros/` from `agilexrobotics/piper_ros@humble` (arm driver, ros2_control, MoveIt 2 config)
- `scout_ros2/` from `agilexrobotics/scout_ros2@humble` (Scout base CAN driver)
- `scout_nav2/` from `AIRLab-POLIMI/scout_nav2` (Nav2 stack tuned for Scout)
- `realsense-ros/` from `IntelRealSense/realsense-ros@ros2-development` (camera driver)

## Quick start

The workstation runs Ubuntu 20.04, which doesn't have apt packages for ROS 2
Humble. Everything below runs inside a Docker dev container (see
[`docker/README.md`](docker/README.md)).

```bash
cd /home/sciarm/agilex/piper_ros/Piper_Scout_ws/Codes

# 1. (once) Install Docker + nvidia-container-toolkit on the host
./scripts/install_docker_nvidia.sh

# 2. (once) Pull upstream packages (no sudo, no ROS 2 needed)
PATH=$HOME/.local/bin:$PATH vcs import src < repos.yaml

# 3. (once) Build the dev container
./docker/build_dev.sh

# 4. Drop into the container — workspace bind-mounted at /workspace
docker compose -f docker/compose.dev.yml run --rm dev

# Inside the container:
colcon build --symlink-install
source install/setup.bash
ros2 launch scout_piper_description view_robot.launch.py     # URDF viz
ros2 launch scout_piper_bringup full_system.launch.py        # full bringup
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
| Camera | `realsense-ros@ros2-development` | Upstream; no functional change vs ROS 1. |

## Cross-references

- ROS 1 source we are porting: [`../../src/stem_grasp_ros1/`](../../src/stem_grasp_ros1/)
- Original launch (parameter source-of-truth): [`stem_grasp_ros1.launch`](../../src/stem_grasp_ros1/launch/stem_grasp_ros1.launch)
- ROS 1 pipeline node: [`pipeline_node.py`](../../src/stem_grasp_ros1/scripts/pipeline_node.py)
- Roadmap: [`../ROADMAP.md`](../ROADMAP.md)
