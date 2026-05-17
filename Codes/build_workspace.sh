#!/usr/bin/env bash
# Convenience build script for the Piper+Scout ROS 2 workspace.
# Run from anywhere; resolves paths relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ERROR: ROS 2 ${ROS_DISTRO} not found at /opt/ros/${ROS_DISTRO}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ ! -d "src/piper_ros" ]]; then
  echo "Upstream packages not imported yet. Running vcs import..."
  vcs import src < repos.yaml
fi

echo "Installing rosdep dependencies..."
rosdep install --from-paths src --ignore-src -r -y || true

echo "Building workspace..."
colcon build --symlink-install "$@"

echo
echo "Done. Next:"
echo "  source ${SCRIPT_DIR}/install/setup.bash"
echo "  ros2 launch scout_piper_bringup full_system.launch.py"
