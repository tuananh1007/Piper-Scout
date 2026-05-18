#!/usr/bin/env bash
# Installs NVIDIA's Isaac ROS apt repo on Ubuntu 22.04 + ROS 2 Humble and
# pulls down `ros-humble-isaac-ros-nvblox` (the precompiled CUDA TSDF + ESDF
# package used by Phase 1).
#
# Canonical URL pattern (verified against the NVIDIA Isaac ROS docs,
# 2025-12 release-3.x line):
#   GPG key : https://isaac.download.nvidia.com/isaac-ros/repos.key
#   Source  : https://isaac.download.nvidia.com/isaac-ros/release-X.X jammy main
#
# Run INSIDE the piper-scout-dev container (the dev user has NOPASSWD sudo).
#
# Override the release via env var if needed:
#   ISAAC_ROS_RELEASE=release-3.1 ./scripts/install_isaac_ros_apt.sh
#
# Reference: https://nvidia-isaac-ros.github.io/getting_started/index.html

set -euo pipefail

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ERROR: ROS 2 Humble not found at /opt/ros/humble." >&2
  echo "Run this inside the piper-scout-dev container." >&2
  exit 1
fi

. /etc/os-release
if [[ "${VERSION_CODENAME}" != "jammy" ]]; then
  echo "ERROR: Isaac ROS apt requires Ubuntu 22.04 (jammy)." >&2
  echo "       Found ${VERSION_CODENAME} (${VERSION_ID}). Bail out." >&2
  exit 1
fi

ISAAC_ROS_RELEASE="${ISAAC_ROS_RELEASE:-release-3.2}"
KEYRING="/usr/share/keyrings/nvidia-isaac-ros.gpg"
SOURCES_LIST="/etc/apt/sources.list.d/nvidia-isaac-ros.list"

echo "==> Updating apt cache (preflight)"
sudo apt-get update -y >/dev/null

echo "==> Installing prerequisites"
sudo apt-get install -y --no-install-recommends \
  curl gnupg ca-certificates

echo "==> Adding NVIDIA Isaac ROS GPG key → ${KEYRING}"
sudo rm -f "${KEYRING}"
curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
  | sudo gpg --dearmor -o "${KEYRING}"
sudo chmod 644 "${KEYRING}"

echo "==> Writing apt source: ${SOURCES_LIST} (${ISAAC_ROS_RELEASE} / jammy)"
echo "deb [signed-by=${KEYRING}] https://isaac.download.nvidia.com/isaac-ros/${ISAAC_ROS_RELEASE} jammy main" \
  | sudo tee "${SOURCES_LIST}" > /dev/null

echo "==> Updating apt cache (with Isaac ROS)"
if ! sudo apt-get update -y; then
  echo >&2
  echo "ERROR: apt-get update failed AFTER adding the Isaac ROS repo." >&2
  echo "Most likely the release version is wrong. Try one of:" >&2
  echo "  ISAAC_ROS_RELEASE=release-3.1 $0" >&2
  echo "  ISAAC_ROS_RELEASE=release-3.0 $0" >&2
  echo "Or check the current release list at:" >&2
  echo "  https://nvidia-isaac-ros.github.io/getting_started/index.html" >&2
  exit 1
fi

echo "==> Installing ros-humble-isaac-ros-nvblox (pulls CUDA libs + TensorRT)"
sudo apt-get install -y ros-humble-isaac-ros-nvblox

echo "==> Sanity check — ROS 2 sees the package"
source /opt/ros/humble/setup.bash
ros2 pkg list | grep isaac_ros_nvblox || {
  echo "WARN: ros2 pkg list did not show isaac_ros_nvblox. The install likely" >&2
  echo "      succeeded but you may need to re-source /opt/ros/humble/setup.bash." >&2
}
ros2 pkg executables isaac_ros_nvblox 2>&1 | head -5 || true

echo
echo "Done. The nvblox node is at:"
echo "  \$(ros2 pkg prefix isaac_ros_nvblox)/lib/isaac_ros_nvblox/nvblox_node"
echo
echo "Next:"
echo "  ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py input_mode:=merged"
