#!/usr/bin/env bash
# OBSOLETE — left in place so old shell history doesn't surprise anyone.
#
# NVIDIA dropped the Isaac ROS apt builds for Ubuntu 22.04 + ROS 2 Humble in
# Q4 2025. The only live apt repo is release-4.0 on noble + Jazzy, which
# doesn't match our dev container.
#
# The new path is to build nvblox from source. See scripts/build_nvblox.sh
# and the updated Dockerfile.dev (which now ships CUDA dev headers).

cat <<'EOF' >&2
This script is obsolete — NVIDIA's release-3.x apt repo for Humble is gone.

To get nvblox running, follow the source-build path:

  1. Rebuild the dev container (it now installs CUDA 12.4 dev libs):
       ./docker/build_dev.sh

  2. Pull the new source repos:
       vcs import src < repos.yaml

  3. Inside the dev container, build the workspace:
       colcon build --symlink-install

See docs/PHASE1_RUNTIME.md for the updated walkthrough.
EOF
exit 1
