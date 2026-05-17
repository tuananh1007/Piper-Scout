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

echo
echo "Patches applied. You can now run colcon build."
