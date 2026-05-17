#!/usr/bin/env bash
# Build the dev container for the Piper+Scout workspace.
# Run AFTER ./scripts/install_docker_nvidia.sh has completed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WS_DIR}"

USER_UID=$(id -u)
USER_GID=$(id -g)

echo "Building piper-scout-dev:humble (uid=${USER_UID} gid=${USER_GID})..."
USER_UID="${USER_UID}" USER_GID="${USER_GID}" \
  docker compose -f docker/compose.dev.yml build dev

echo
echo "Done. Drop into the dev container with:"
echo "  cd ${WS_DIR}"
echo "  docker compose -f docker/compose.dev.yml run --rm dev"
echo
echo "Inside, the workspace is at /workspace. Build with:"
echo "  colcon build --symlink-install"
