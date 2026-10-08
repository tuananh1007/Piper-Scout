#!/usr/bin/env bash
# Convenience build script for the Piper+Scout ROS 2 workspace.
# Run from anywhere; resolves paths relative to this script. Extra arguments
# go to colcon build (e.g. --packages-select stem_grasp). The build runs
# through scripts/colcon_build_safe.sh, which limits parallel compiles to the
# free memory (a plain colcon build can freeze a 16 GB machine).
#
#   SKIP_ROSDEP=1 ./build_workspace.sh ...   skip the rosdep install step

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ERROR: ROS 2 ${ROS_DISTRO} not found at /opt/ros/${ROS_DISTRO}" >&2
  exit 1
fi

# shellcheck disable=SC1090
set +u; source "/opt/ros/${ROS_DISTRO}/setup.bash"; set -u

if [[ ! -d "src/piper_ros" ]]; then
  echo "Upstream packages not imported yet. Running vcs import..."
  vcs import src < repos.yaml
fi

if [[ -z "${SKIP_ROSDEP:-}" ]]; then
  echo "Installing rosdep dependencies..."
  rosdep install --from-paths src --ignore-src -r -y || true
fi

echo "Building workspace..."
"${SCRIPT_DIR}/scripts/colcon_build_safe.sh" --symlink-install "$@"

echo
echo "Done. Next:"
echo "  source ${SCRIPT_DIR}/install/setup.bash"
echo "  ros2 launch scout_piper_bringup full_system.launch.py"
