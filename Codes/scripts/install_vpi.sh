#!/usr/bin/env bash
# Installs NVIDIA VPI 4.x for Ubuntu 22.04 x86_64.
#
# VPI (Vision Programming Interface) is required by isaac_ros_common
# (which is a build dep of nvblox_ros). NVIDIA distributes the x86_64
# VPI Debs via their Jetson apt repo (yes, the Jetson one — that's the
# canonical location per docs.nvidia.com/vpi/installation.html).
#
# Run INSIDE the dev container.

set -euo pipefail

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ERROR: ROS 2 Humble not found. Run inside the piper-scout-dev container." >&2
  exit 1
fi

. /etc/os-release
if [[ "${VERSION_CODENAME}" != "jammy" ]]; then
  echo "ERROR: VPI x86_64 requires Ubuntu 22.04 (jammy). Found ${VERSION_CODENAME}." >&2
  exit 1
fi

KEYRING="/usr/share/keyrings/nvidia-jetson-ota.gpg"
SOURCES_LIST="/etc/apt/sources.list.d/nvidia-vpi.list"

echo "==> Installing prereqs"
sudo apt-get update -y >/dev/null
sudo apt-get install -y --no-install-recommends curl gnupg ca-certificates software-properties-common

echo "==> Adding NVIDIA Jetson OTA GPG key → ${KEYRING}"
sudo rm -f "${KEYRING}"
curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
  | sudo gpg --dearmor -o "${KEYRING}"
sudo chmod 644 "${KEYRING}"

echo "==> Writing apt source: ${SOURCES_LIST}"
echo "deb [signed-by=${KEYRING}] https://repo.download.nvidia.com/jetson/x86_64/jammy r38.4 main" \
  | sudo tee "${SOURCES_LIST}" > /dev/null

echo "==> Updating apt cache"
sudo apt-get update -y

echo "==> Installing VPI 4 (libnvvpi4 + vpi4-dev)"
sudo apt-get install -y libnvvpi4 vpi4-dev

echo
echo "==> Verifying VPI install"
ls -la /opt/nvidia/vpi*/lib/x86_64-linux-gnu/ 2>&1 | head -5 || true
ls /usr/lib/x86_64-linux-gnu/libnvvpi*.so* 2>&1 | head -5 || true

# Make cmake find VPI. The vpi*-dev package installs a cmake config file
# under /opt/nvidia/vpi*/lib/x86_64-linux-gnu/cmake/vpi/ on most systems;
# some installs use /usr/lib/x86_64-linux-gnu/cmake/vpi/. Probe both.
echo
echo "==> Locating vpiConfig.cmake"
find /opt/nvidia -name 'vpiConfig.cmake' 2>/dev/null | head -3
find /usr/lib -name 'vpiConfig.cmake' 2>/dev/null | head -3

echo
echo "Done. Next:"
echo "  colcon build --symlink-install --packages-up-to isaac_ros_nvblox"
echo "If find_package(vpi) still fails, set:"
echo "  export CMAKE_PREFIX_PATH=\$(find /opt/nvidia -name 'vpiConfig.cmake' -printf '%h\\n' | head -1):\$CMAKE_PREFIX_PATH"
