#!/usr/bin/env bash
# Installs the NVIDIA Isaac ROS apt repository inside a ROS 2 Humble container
# and pulls down ros-humble-isaac-ros-nvblox (the precompiled CUDA TSDF + ESDF
# package used by Phase 1).
#
# Run this INSIDE the dev container (or as part of Dockerfile.dev once we're
# happy it works). Requires sudo (the dev user has NOPASSWD configured).
#
# Usage:
#   ./scripts/install_isaac_ros_apt.sh
#
# Reference:
#   https://nvidia-isaac-ros.github.io/getting_started/setup.html

set -euo pipefail

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ERROR: ROS 2 Humble not found at /opt/ros/humble." >&2
  echo "Run this inside the piper-scout-dev container." >&2
  exit 1
fi

# Ubuntu 22.04 only — Isaac ROS doesn't ship binaries for 20.04.
. /etc/os-release
if [[ "${VERSION_ID}" != "22.04" ]]; then
  echo "ERROR: Isaac ROS apt requires Ubuntu 22.04 (got ${VERSION_ID})." >&2
  exit 1
fi

echo "==> Adding NVIDIA Isaac ROS apt key + repository"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends curl gnupg ca-certificates

# Public Isaac ROS apt repo (anonymous; no NGC key required for release-3.x).
curl -fsSL https://isaac.download.nvidia.com/isaac-ros/release-3/$(. /etc/os-release && echo "${ID}${VERSION_ID//./}")/isaac-ros.list \
  | sudo tee /etc/apt/sources.list.d/isaac-ros.list > /dev/null || {
    echo "WARN: failed to fetch release-3 list, falling back to manual form" >&2
    echo "deb [trusted=yes] https://isaac.download.nvidia.com/isaac-ros/release-3/ubuntu/$(lsb_release -cs) main" \
      | sudo tee /etc/apt/sources.list.d/isaac-ros.list > /dev/null
}

curl -fsSL https://isaac.download.nvidia.com/isaac-ros/release-3/isaac-ros.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/isaac-ros-archive-keyring.gpg 2>/dev/null || \
    echo "(Skipping signed-by key install; trusted=yes used as fallback.)"

echo "==> Updating apt cache"
sudo apt-get update -y || {
  echo "ERROR: apt-get update failed. The Isaac ROS repo URL may have changed." >&2
  echo "Visit https://nvidia-isaac-ros.github.io/getting_started/setup.html for the current URL." >&2
  exit 1
}

echo "==> Installing nvblox + visual SLAM (optional but useful for Scout Nav)"
sudo apt-get install -y \
  ros-humble-isaac-ros-nvblox \
  || {
    echo "ERROR: apt install failed. Try: apt list --installed 2>/dev/null | grep isaac" >&2
    exit 1
  }

echo
echo "==> Smoke test: locate the nvblox node executable"
ros2 pkg executables isaac_ros_nvblox 2>&1 | head -3 || \
  echo "WARN: ros2 pkg executables didn't find isaac_ros_nvblox; check the install."

echo
echo "Done. Verify with:"
echo "  source /opt/ros/humble/setup.bash"
echo "  ros2 pkg list | grep isaac_ros_nvblox"
echo "  ros2 run isaac_ros_nvblox nvblox_node --help    (or similar)"
