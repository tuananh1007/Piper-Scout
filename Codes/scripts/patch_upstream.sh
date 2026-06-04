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

  # find_package(negotiated) alone isn't enough: ament_auto_add_library
  # doesn't propagate negotiated's exported include dirs to the target's
  # compile commands, so #include "negotiated/negotiated_publisher.hpp"
  # fails. Add an explicit ament_target_dependencies call after the
  # target_link_libraries(...) block.
  if grep -q "ament_target_dependencies(\${PROJECT_NAME} negotiated)" "${NITROS_CMAKE}"; then
    echo "isaac_ros_nitros: ament_target_dependencies(negotiated) already present. Skipping."
  else
    echo "Patching ${NITROS_CMAKE} → add ament_target_dependencies(\${PROJECT_NAME} negotiated)"
    # Insert immediately after the closing paren of the target_link_libraries
    # block that ends with "yaml-cpp\n)".
    python3 - "${NITROS_CMAKE}" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
insertion = (
    "\n# Patch: explicitly add ament_target_dependencies for negotiated so its\n"
    "# exported include dirs propagate to this target. ament_auto_add_library\n"
    "# should have picked it up from <depend>negotiated</depend> in package.xml,\n"
    "# but for reasons unclear, doesn't.\n"
    "ament_target_dependencies(${PROJECT_NAME} negotiated)\n"
)
pattern = re.compile(
    r"(target_link_libraries\(\$\{PROJECT_NAME\}\s*\n"
    r"\s*Eigen3::Eigen\s*\n"
    r"\s*vpi\s*\n"
    r"\s*yaml-cpp\s*\n"
    r"\))"
)
new_text, n = pattern.subn(r"\1" + insertion, text, count=1)
if n != 1:
    sys.stderr.write("patch_upstream: failed to locate target_link_libraries block in NITROS CMakeLists\n")
    sys.exit(1)
p.write_text(new_text)
PY
  fi

  # Without ament_export_dependencies(magic_enum negotiated), downstream
  # nitros_*_type packages fail CMake config because the isaac_ros_nitros
  # target's link interface references magic_enum::magic_enum but no
  # find_package(magic_enum) is called in the downstream context.
  if grep -q "ament_export_dependencies(magic_enum negotiated)" "${NITROS_CMAKE}"; then
    echo "isaac_ros_nitros: ament_export_dependencies(magic_enum negotiated) already present. Skipping."
  else
    echo "Patching ${NITROS_CMAKE} → add ament_export_dependencies(magic_enum negotiated)"
    sed -i "s|^\(ament_auto_package(INSTALL_TO_SHARE config)\)\$|ament_export_dependencies(magic_enum negotiated)\n\n\1|" \
      "${NITROS_CMAKE}"
  fi
fi

# ----------------------------------------------------------------------------
# 3b. VPI 3→4 also renamed VPIImagePlanePitchLinear::data (void*) to
#     VPIImagePlanePitchLinear::pBase (VPIByte*) + .offsetBytes. NITROS's
#     nitros_image.cpp uses the old field name in two places. Patch both.
# ----------------------------------------------------------------------------
NITROS_IMAGE_CPP="${SRC_DIR}/isaac_ros_nitros/isaac_ros_nitros_type/isaac_ros_nitros_image_type/src/nitros_image.cpp"
if [[ -f "${NITROS_IMAGE_CPP}" ]]; then
  if grep -q "// VPI 4 replaced \.data" "${NITROS_IMAGE_CPP}"; then
    echo "${NITROS_IMAGE_CPP}: VPI 4 .data→.pBase patch already applied. Skipping."
  else
    echo "Patching ${NITROS_IMAGE_CPP} → VPI 4 .data → .pBase + .offsetBytes"
    python3 - "${NITROS_IMAGE_CPP}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
old_write = "    img_data.buffer.pitch.planes[i].data = video_buff->pointer() + data_ptr_offset;"
new_write = (
    "    // VPI 4 replaced .data (void*) with .pBase (VPIByte*) + .offsetBytes.\n"
    "    img_data.buffer.pitch.planes[i].pBase =\n"
    "      reinterpret_cast<VPIByte *>(video_buff->pointer() + data_ptr_offset);\n"
    "    img_data.buffer.pitch.planes[i].offsetBytes = 0;"
)
old_read = "    src_ptr = output_data.buffer.pitch.planes[0].data;"
new_read = (
    "    // VPI 4: .data became .pBase + .offsetBytes.\n"
    "    src_ptr = reinterpret_cast<void *>(\n"
    "      output_data.buffer.pitch.planes[0].pBase +\n"
    "      output_data.buffer.pitch.planes[0].offsetBytes);"
)
n1 = text.count(old_write); n2 = text.count(old_read)
if n1 != 1 or n2 != 1:
    sys.stderr.write(f"patch_upstream: VPI .data sites not found (write={n1}, read={n2})\n")
    sys.exit(1)
p.write_text(text.replace(old_write, new_write).replace(old_read, new_read))
PY
  fi
fi

# ----------------------------------------------------------------------------
# 4. isaac_ros_nitros ships license-constrained GXF prebuilt .so binaries via
#    Git LFS (under isaac_ros_gxf/gxf/core/lib/gxf_x86_64_cuda_12_6/...). vcs
#    import doesn't trigger LFS, so without this step you get 132-byte pointer
#    files instead of real ELF libs, and NITROS fails to link with:
#      ld: libgxf_core.so: file format not recognized; treating as linker script
# ----------------------------------------------------------------------------
NITROS_REPO="${SRC_DIR}/isaac_ros_nitros"
NITROS_GXF_CORE="${NITROS_REPO}/isaac_ros_gxf/gxf/core/lib/gxf_x86_64_cuda_12_6/core/libgxf_core.so"
if [[ -d "${NITROS_REPO}/.git" ]]; then
  if [[ -f "${NITROS_GXF_CORE}" ]] && \
     [[ "$(stat -c%s "${NITROS_GXF_CORE}")" -gt 10000 ]]; then
    echo "isaac_ros_nitros: GXF prebuilt libs already populated. Skipping LFS pull."
  else
    if command -v git-lfs >/dev/null 2>&1; then
      echo "isaac_ros_nitros: fetching LFS objects (GXF prebuilts, ~50-100 MB)..."
      git -C "${NITROS_REPO}" lfs install
      git -C "${NITROS_REPO}" lfs pull
    else
      echo "WARNING: git-lfs not installed, but isaac_ros_nitros has LFS pointer files."
      echo "  Run: ./scripts/install_git_lfs_and_pull.sh  (one-time, needs sudo)"
      echo "  Then re-run this script."
    fi
  fi
fi

# ----------------------------------------------------------------------------
# 5. isaac_ros_nvblox vendors the nvblox CUDA library as a submodule at
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
