# Source this (don't execute) to activate the workspace.
#   source setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [[ -f "${SCRIPT_DIR}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${SCRIPT_DIR}/install/setup.bash"
fi

export PIPER_SCOUT_WS="${SCRIPT_DIR}"
echo "Activated Piper+Scout ROS 2 workspace at ${PIPER_SCOUT_WS}"
