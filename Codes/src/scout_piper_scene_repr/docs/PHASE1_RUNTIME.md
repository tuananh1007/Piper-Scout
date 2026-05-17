# Phase 1 Runtime Setup

How to actually run the Phase 1 semantic-SDF stack on this machine.

The companion design doc is [`PHASE1_DESIGN.md`](PHASE1_DESIGN.md).

## What's needed

| Component | Where it runs | Status |
|---|---|---|
| `class_demux_node` (Python) | Our dev container | ☑ Built, fully testable today |
| Synthetic mask publisher (test) | Our dev container | ☑ For plumbing tests without nvblox |
| `nvblox_node` (Isaac ROS) | Our dev container, via apt | ◐ Apt repo install — see below |
| `semantic_collision_plugin` (C++) | MoveIt 2's move_group | ☑ Built; needs nvblox ESDF API wiring |

## Step 1 — Plumbing smoke test (no nvblox needed)

Validates that the synthetic mask → class_demux → per-class topics path
works. **Do this first.**

```bash
# Host:
docker compose -f docker/compose.dev.yml exec dev bash

# Inside the container:
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select scout_piper_scene_repr
source install/setup.bash
ros2 launch scout_piper_scene_repr test_class_demux.launch.py
```

In another container shell:

```bash
docker compose -f docker/compose.dev.yml exec dev bash
source /workspace/install/setup.bash

# Verify topics:
ros2 topic list | grep scene_repr
# Expect: /scene_repr/mask/{stem,branch,leaf,target}
#         /scene_repr/depth/{stem,branch,leaf,target}
#         /scene_repr/policy

ros2 topic hz /scene_repr/mask/stem       # ~10 Hz
ros2 topic hz /scene_repr/mask/target     # ~10 Hz
rqt &                                      # use Image View to confirm masks look right
```

Expected: four separate mask images (a vertical stem column, a circular
leaf blob, a small target disc, a branch nub). Each gated depth stream
is non-zero only where its class mask is non-zero.

## Step 2 — Install nvblox via NVIDIA's apt repo

NVIDIA moved Isaac ROS to apt-based distribution in 2024. You add their
repo to your sources, then `apt install ros-humble-isaac-ros-nvblox`.

```bash
# Inside the dev container:
/workspace/scripts/install_isaac_ros_apt.sh
```

The script:
- Verifies you're on Ubuntu 22.04 + ROS 2 Humble
- Adds the public Isaac ROS apt repo (no NGC key needed for `release-3.x`)
- Installs `ros-humble-isaac-ros-nvblox`
- Smoke-checks that the nvblox executable is on the ROS 2 package path

If the apt URLs have moved by the time you run this, the script prints
the canonical doc URL: [Isaac ROS Setup](https://nvidia-isaac-ros.github.io/getting_started/setup.html).

## Step 3 — Run nvblox standalone (smoke test)

Once nvblox is installed:

```bash
# In the dev container:
ros2 launch isaac_ros_nvblox isaac_ros_nvblox.launch.py    # example name — check the actual installed launch
```

Look for the `/nvblox_node/static_esdf_layer` topic (or similar) appearing.

## Step 4 — Wire class_demux → 4× nvblox

Phase 1 v0 design (see `PHASE1_DESIGN.md` §5): instantiate four nvblox
nodes, each consuming the mask-gated depth from one class. The launch
file is [`launch/nvblox_semantic.launch.py`](../launch/nvblox_semantic.launch.py).

```bash
ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py
```

(Adjust the `input_mode` argument if your segmentation node is producing
the legacy two-topic interface rather than a merged label image.)

## Fallback paths if the apt repo install fails

| Scenario | Workaround |
|---|---|
| Apt URLs moved | Check [official setup docs](https://nvidia-isaac-ros.github.io/getting_started/setup.html), edit `install_isaac_ros_apt.sh` accordingly |
| 403 / authentication required | The release-3.x repo is anonymous; for release-2.x or earlier you'd need an NGC API key. Use release-3.x or newer. |
| Package conflicts with our pinned numpy | `apt show ros-humble-isaac-ros-nvblox | grep Depends` — if it forces numpy>=2, we need to revisit the pinning strategy. |

## Phase 1 task tracking

Live in [`../../../../PROGRESS.md`](../../../../PROGRESS.md) under "Phase 1".
