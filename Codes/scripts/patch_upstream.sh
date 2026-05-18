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
# 2. VPI 3→4 NVENC removal. Both isaac_ros_common and isaac_ros_nitros have
#    their own copy of vpi_utilities.cpp; both reference VPI_BACKEND_NVENC,
#    which VPI 4 dropped. Comment out the NVENC entry in each.
# ----------------------------------------------------------------------------
for VPI_UTILS in \
    "${SRC_DIR}/isaac_ros_common/isaac_ros_common/src/vpi_utilities.cpp" \
    "${SRC_DIR}/isaac_ros_nitros/isaac_ros_nitros/src/utils/vpi_utilities.cpp" ; do
  if [[ -f "${VPI_UTILS}" ]]; then
    if grep -q 'VPI_BACKEND_NVENC' "${VPI_UTILS}" && \
       ! grep -q '// NVENC removed in VPI 4' "${VPI_UTILS}"; then
      echo "Patching ${VPI_UTILS} → removing VPI_BACKEND_NVENC entry"
      sed -i 's|^\(\s*\){"NVENC", VPI_BACKEND_NVENC},$|\1// {"NVENC", VPI_BACKEND_NVENC},  // NVENC removed in VPI 4|' "${VPI_UTILS}"
    else
      echo "${VPI_UTILS}: NVENC patch already applied (or upstream changed). Skipping."
    fi
  fi
done

# ----------------------------------------------------------------------------
# 3. isaac_ros_nitros: CMakeLists.txt links to magic_enum::magic_enum and
#    includes negotiated/negotiated_publisher.hpp, but doesn't call
#    find_package() for either. ament_auto_find_build_dependencies in NITROS
#    SHOULD pick them up from package.xml automatically — but doesn't, for
#    reasons that look like an ament_auto bug. Add explicit find_package
#    calls next to the existing ones. Idempotent.
# ----------------------------------------------------------------------------
NITROS_CMAKE="${SRC_DIR}/isaac_ros_nitros/isaac_ros_nitros/CMakeLists.txt"
if [[ -f "${NITROS_CMAKE}" ]]; then
  for PKG in magic_enum negotiated; do
    if grep -q "find_package(${PKG} " "${NITROS_CMAKE}"; then
      echo "isaac_ros_nitros: ${PKG} find_package already present. Skipping."
    else
      echo "Patching ${NITROS_CMAKE} → add find_package(${PKG} REQUIRED)"
      sed -i "s|^\(find_package(vpi REQUIRED)\)\$|\1\nfind_package(${PKG} REQUIRED)|" \
        "${NITROS_CMAKE}"
    fi
  done
fi

# ----------------------------------------------------------------------------
# 4. isaac_ros_nvblox vendors the nvblox CUDA library as a submodule at
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
