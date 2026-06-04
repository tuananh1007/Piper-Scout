#!/usr/bin/env bash
# One-shot: install git-lfs on the host and re-fetch LFS objects in
# src/isaac_ros_nitros (which ships license-constrained GXF prebuilt .so
# binaries via Git LFS — vcs import only pulled the pointer files).
#
# After this runs, the broken libgxf_core.so symlink under
# install/isaac_ros_gxf/lib/ will resolve to a real ELF shared library.
#
# Usage:
#   ./scripts/install_git_lfs_and_pull.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../src" && pwd)"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo ">> Installing git-lfs via apt..."
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends git-lfs
else
  echo ">> git-lfs already installed: $(git lfs version)"
fi

echo ">> Initializing git-lfs hooks..."
git lfs install --skip-repo

NITROS_REPO="${SRC_DIR}/isaac_ros_nitros"
if [[ -d "${NITROS_REPO}/.git" ]]; then
  echo ">> Pulling LFS objects in isaac_ros_nitros (~50-100 MB)..."
  git -C "${NITROS_REPO}" lfs install
  git -C "${NITROS_REPO}" lfs pull
  echo ">> Verifying libgxf_core.so..."
  LIB="${NITROS_REPO}/isaac_ros_gxf/gxf/core/lib/gxf_x86_64_cuda_12_6/core/libgxf_core.so"
  if [[ -f "${LIB}" ]]; then
    file "${LIB}"
    ls -lh "${LIB}"
  else
    echo "WARNING: ${LIB} not found after lfs pull"
  fi
else
  echo "ERROR: ${NITROS_REPO} is not a git repo — run vcs import first."
  exit 1
fi

echo
echo "Done. Rebuild with: colcon build --symlink-install --packages-up-to isaac_ros_nvblox"
