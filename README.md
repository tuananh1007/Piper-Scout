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
| [`Codes/`](Codes/) | ROS 2 Humble colcon workspace (Phase 0 baseline + Phase 1, 2A and 3A packages) |
| [`research/`](research/) | Research tracks (semantic scene, plant twin, Piper-JEPA, whole-body MPC, active perception), shared platform facts and ID registry |

## Phase status

| Phase | Theme | Status |
|---|---|---|
| 0 | ROS 2 Humble migration + Scout integration | ◐ in progress (workspace, URDF, bringup and `stem_grasp` port done except plan→execute and iterative approach; `moveit_servo` wiring + hardware validation pending) |
| 1 | Semantic RGB-D scene (nvblox) + deformable plant state (`plant_twin`) | ◐ in progress (RealSense→nvblox validated; `plant_twin` core + node done; CPU semantic distance query and MoveIt semantic collision plugin done, not yet on hardware) |
| 2A | V-JEPA 2.1 dense temporal target state | ◐ in progress (Stage A target memory in `scout_piper_jepa` done; E1 dataset + V-JEPA runs pending) |
| 2B | Safety-bounded local MPPI visual servo | ☐ not started |
| 3A | Geometry-only whole-body GPU MPC (Scout + Piper) | ◐ in progress (`scout_piper_whole_body_mpc` CPU MPPI + safety filter + dry-run node, convergence near obstacles and W0/W1 baselines done; hardware and Orin runs pending) |
| 3B | Piper-JEPA predictive whole-body MPC (headline) | ☐ not started |
| 4 | Uncertainty-driven active perception | ☐ not started |
| 5 | Language/VLM + operator GUI for non-experts | ☐ not started |

See [`PROGRESS.md`](PROGRESS.md) for task-level detail and current blockers.

## Quick start (Phase 0)

Needs ROS 2 Humble (Ubuntu 22.04). On the Ubuntu 20.04 workstation, follow
[`Codes/README.md`](Codes/README.md), which runs these steps in the dev container.

```bash
cd Codes
source /opt/ros/humble/setup.bash
vcs import src < repos.yaml             # pull upstream packages
./scripts/patch_upstream.sh             # post-import fixes (ugv_sdk, Isaac ROS, nvblox submodule)
./build_workspace.sh                    # rosdep + colcon build
source install/setup.bash
ros2 launch scout_piper_description view_robot.launch.py   # visual sanity check
```

Full walkthrough: [`Codes/PHASE0_CHECKLIST.md`](Codes/PHASE0_CHECKLIST.md).

## Hardware

- **Arm:** [AgileX Piper](https://global.agilex.ai/products/piper) — 6-DoF, ~1.5 kg payload
- **Base:** [AgileX Scout 2.0](https://global.agilex.ai/products/scout-2-0) — 4WD skid-steer, 50 kg payload, 1.5 m/s
- **Camera:** Intel RealSense, eye-in-hand on the Piper EE (D435 in the URDF; the Phase 1 nvblox smoke test used a D405; model to be confirmed)
- **Compute:** NVIDIA Jetson Orin AGX 64 GB

## License

MIT, except `scout_piper_jepa` and `scout_piper_whole_body_mpc`, which declare Apache-2.0 in their
`package.xml` — see individual package licenses for upstream code.
