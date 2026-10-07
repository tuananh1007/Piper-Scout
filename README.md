# Piper-Scout

An autonomous mobile-manipulation platform for plant peduncle/branch grasping.
**AgileX Piper** 6-DoF arm mounted on **AgileX Scout 2.0** UGV, with **Intel
RealSense** depth camera, running on **NVIDIA Jetson Orin AGX**.

## Goal

Let a non-expert operator stand next to the robot, type or speak a
natural-language instruction ("grasp the third flower from the left"), and
have the system safely execute it — combining vision-language models for
understanding, GPU-parallel whole-body MPC for control, and a safety-bounded
visual servo for the final approach.

## Contents

| File / dir | Purpose |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | 18-month research & development plan (Phases 0–5) |
| [`PROGRESS.md`](PROGRESS.md) | Living per-phase task tracker — update as work ships |
| [`INSTALL.md`](INSTALL.md) | Step-by-step installation, setup and run guide |
| [`Codes/`](Codes/) | ROS 2 Humble colcon workspace (Phase 0 baseline + Phase 1, 2A, 3A and 3B packages) |
| [`research/`](research/) | Research tracks (semantic scene, plant twin, Piper-JEPA, whole-body MPC, active perception), shared platform facts and ID registry |

## Phase status

| Phase | Theme | Status |
|---|---|---|
| 0 | ROS 2 Humble migration + Scout integration | ◐ in progress (workspace, URDF, bringup and `stem_grasp` port done except plan→execute and iterative approach; `moveit_servo` wiring + hardware validation pending) |
| 1 | Semantic RGB-D scene (nvblox) + deformable plant state (`plant_twin`) | ◐ in progress (RealSense→nvblox validated; `plant_twin` core + node done; CPU semantic distance query and MoveIt semantic collision plugin done, not yet on hardware) |
| 2A | V-JEPA 2.1 dense temporal target state | ◐ in progress (Stage A target memory in `scout_piper_jepa` done; E1 dataset + V-JEPA runs pending) |
| 2B | Safety-bounded local MPPI visual servo | ☐ not started |
| 3A | Geometry-only whole-body GPU MPC (Scout + Piper) | ◐ in progress (`scout_piper_whole_body_mpc` CPU MPPI + safety filter + dry-run node, convergence near obstacles and W0/W1 baselines done; hardware and Orin runs pending) |
| 3B | Piper-JEPA predictive whole-body MPC (headline) | ◐ in progress (Stage B predictor + Stage C cost hook on a synthetic world; robot data, V-JEPA and closed-loop gain pending) |
| 4 | Uncertainty-driven active perception | ☐ not started |
| 5 | Language/VLM + operator GUI for non-experts | ☐ not started |

See [`PROGRESS.md`](PROGRESS.md) for task-level detail and current blockers.

## Installation and setup

Full step-by-step guide: **[`INSTALL.md`](INSTALL.md)** (prerequisites, build,
tests, CAN and camera setup, running each subsystem with pass checks,
troubleshooting). The short version:

| Path | Machine | Guide |
|---|---|---|
| A | Jetson Orin AGX on the robot | Not documented yet ([details](INSTALL.md#path-a-jetson-orin-agx-not-documented-yet)) |
| B | x86_64 Ubuntu 22.04, native ROS 2 Humble | [Steps 1–10](INSTALL.md#choose-your-path) |
| C | x86_64 Ubuntu 20.04 workstation, Docker dev container (the validated nvblox path) | [Steps 1–10](INSTALL.md#choose-your-path) |
| D | Any machine with Python 3.10–3.12, no ROS, no robot | [Research quick start](INSTALL.md#research-quick-start-without-ros-path-d) |

Build (path C; on path B install ROS 2 Humble and the packages of
[Step 3B](INSTALL.md#3b--native-ubuntu-2204-path-b), then run the same
`vcs`, patch and `colcon` steps natively after `rosdep install`):

```bash
sudo apt-get install -y git python3-pip         # host, once
git clone https://github.com/tuananh1007/Piper-Scout.git && cd Piper-Scout/Codes
./scripts/install_docker_nvidia.sh               # host, once: Docker + NVIDIA Container Toolkit (re-login after)
python3 -m pip install --user vcstool            # host tools for the next two lines
PATH=$HOME/.local/bin:$PATH vcs import src < repos.yaml
git lfs version || ./scripts/install_git_lfs_and_pull.sh   # once: Git LFS for the GXF binaries
./scripts/patch_upstream.sh                      # after every vcs import
./docker/build_dev.sh                            # dev image piper-scout-dev:humble
ls -l ~/.Xauthority || xauth extract ~/.Xauthority "$DISPLAY"   # must be a file before the first run
docker compose -f docker/compose.dev.yml run --rm dev
# inside the container, in /workspace:
colcon build --symlink-install && source install/setup.bash
ros2 launch scout_piper_description view_robot.launch.py      # URDF check, no hardware
```

Algorithms only (path D), from the repository root:

```bash
python3 -m venv .venv && . .venv/bin/activate   # Ubuntu: sudo apt-get install -y python3-venv first
python -m pip install 'numpy<2' scipy pyyaml pytest opencv-python-headless
cd Codes/src/scout_piper_jepa && python -m pytest test -q    # likewise in the other pure-Python packages
```

> **Before driving hardware** read
> [what works today](INSTALL.md#before-you-start-what-works-today): the robot
> stack is not validated on hardware, there is no validated software stop, and
> the Piper driver executes every message on its command input at full speed.
> Start the arm only through `full_system.launch.py`, which keeps those
> commands on `/piper/joint_cmd` instead of upstream's `/joint_states`.

## Hardware

- **Arm:** [AgileX Piper](https://global.agilex.ai/products/piper) — 6-DoF, ~1.5 kg payload
- **Base:** [AgileX Scout 2.0](https://global.agilex.ai/products/scout-2-0) — 4WD skid-steer, 50 kg payload, 1.5 m/s
- **Camera:** Intel RealSense D405, eye-in-hand on the Piper EE
- **Compute:** NVIDIA Jetson Orin AGX 64 GB

## License

MIT, except `scout_piper_jepa` and `scout_piper_whole_body_mpc`, which declare Apache-2.0 in their
`package.xml` — see individual package licenses for upstream code.
