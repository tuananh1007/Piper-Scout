#!/usr/bin/env bash
# Installs Docker CE + NVIDIA Container Toolkit on Ubuntu 20.04.
# Run once on the workstation; requires sudo.
#
# Usage:
#   chmod +x scripts/install_docker_nvidia.sh
#   ./scripts/install_docker_nvidia.sh

set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Do not run this as root. It uses sudo internally." >&2
  exit 1
fi

echo "==> Updating apt cache"
sudo apt-get update -y

echo "==> Installing prerequisites"
sudo apt-get install -y ca-certificates curl gnupg lsb-release

echo "==> Adding Docker official GPG key + repo"
sudo install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
fi

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

echo "==> Adding $USER to the docker group (effective next login)"
sudo usermod -aG docker "$USER"

echo "==> Installing NVIDIA Container Toolkit"
distribution="ubuntu20.04"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit

echo "==> Wiring NVIDIA runtime into Docker"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo
echo "==> Smoke test (will fail until you log out + back in so group takes effect)"
echo "    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi"
echo
echo "Done. Log out and back in (or 'newgrp docker') before running docker as a non-root user."
