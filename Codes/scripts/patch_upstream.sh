#!/usr/bin/env bash
# Post-vcs-import patches for upstream packages.
# Run after every `vcs import` to apply local-only fixes that aren't upstreamed yet.
#
# Usage:
#   ./scripts/patch_upstream.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../src" && pwd)"

# ----------------------------------------------------------------------------
# 1. ugv_sdk: declare build_type=cmake so colcon doesn't fall back to
#    ament_cmake. Upstream package.xml has an empty <export></export>.
# ----------------------------------------------------------------------------
UGV_PKG="${SRC_DIR}/ugv_sdk/package.xml"
if [[ -f "${UGV_PKG}" ]]; then
  if grep -q '<export></export>' "${UGV_PKG}"; then
    echo "Patching ${UGV_PKG} → adding <build_type>cmake</build_type>"
    sed -i 's|<export></export>|<export>\n    <build_type>cmake</build_type>\n  </export>|' "${UGV_PKG}"
  elif grep -q '<build_type>cmake</build_type>' "${UGV_PKG}"; then
    echo "ugv_sdk already patched. Skipping."
  else
    echo "WARNING: ugv_sdk/package.xml has an unexpected <export> shape; not patching."
  fi
else
  echo "ugv_sdk not found at ${UGV_PKG} — run vcs import first."
fi


# ----------------------------------------------------------------------------
# 2. isaac_ros_common@release-3.2 was built against VPI 3 and references
#    VPI_BACKEND_NVENC, which VPI 4 dropped. We have VPI 4 installed (the
#    only x86_64 jammy VPI NVIDIA still ships). Comment out the NVENC entry.
# ----------------------------------------------------------------------------
VPI_UTILS="${SRC_DIR}/isaac_ros_common/isaac_ros_common/src/vpi_utilities.cpp"
if [[ -f "${VPI_UTILS}" ]]; then
  if grep -q 'VPI_BACKEND_NVENC' "${VPI_UTILS}" && \
     ! grep -q '// NVENC removed in VPI 4' "${VPI_UTILS}"; then
    echo "Patching ${VPI_UTILS} → removing VPI_BACKEND_NVENC entry"
    sed -i 's|^\(\s*\){"NVENC", VPI_BACKEND_NVENC},$|\1// {"NVENC", VPI_BACKEND_NVENC},  // NVENC removed in VPI 4|' "${VPI_UTILS}"
  else
    echo "vpi_utilities.cpp NVENC patch already applied (or upstream changed). Skipping."
  fi
fi

# ----------------------------------------------------------------------------
# 3. isaac_ros_nvblox vendors the nvblox CUDA library as a submodule at
#    nvblox_ros/nvblox_core. vcs import doesn't init submodules; do it here.
# ----------------------------------------------------------------------------
NVBLOX_REPO="${SRC_DIR}/isaac_ros_nvblox"
if [[ -d "${NVBLOX_REPO}" ]]; then
  if [[ -f "${NVBLOX_REPO}/nvblox_ros/nvblox_core/CMakeLists.txt" ]]; then
    echo "isaac_ros_nvblox: nvblox_core submodule already populated. Skipping."
  else
    echo "Initing isaac_ros_nvblox submodule (nvblox CUDA core library, ~50 MB)"
    git -C "${NVBLOX_REPO}" submodule update --init --recursive
  fi
else
  echo "isaac_ros_nvblox not found at ${NVBLOX_REPO} — run vcs import first."
fi

echo
echo "Patches applied. You can now run colcon build."
