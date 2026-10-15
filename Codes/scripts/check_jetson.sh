#!/usr/bin/env bash
# Read-only checks for running this workspace on a Jetson AGX Orin
# (INSTALL.md, Path A). Changes nothing; run it as your normal user:
#
#   ./scripts/check_jetson.sh
#
# OK / WARN / FAIL lines compare the machine with what the pinned upstreams
# need: Isaac ROS release-3.2 is built for JetPack 6.1+ (L4T r36.4), CUDA
# 12.6 and VPI 3; ROS 2 Humble needs Ubuntu 22.04. Paste the output into
# INSTALL.md / PROGRESS.md when it changes (the JetPack version is a TODO there).

ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; }
have() { command -v "$1" > /dev/null 2>&1; }

echo "== System"
arch=$(uname -m)
[[ "$arch" == "aarch64" ]] && ok "architecture aarch64" || fail "architecture $arch (not a Jetson)"
if [[ -r /etc/nv_tegra_release ]]; then
  l4t=$(head -1 /etc/nv_tegra_release)
  if [[ "$l4t" =~ R36.*REVISION:\ (4|5|6) ]]; then ok "L4T: $l4t"
  else warn "L4T: $l4t (Isaac ROS 3.2 expects r36.4 = JetPack 6.1/6.2)"; fi
else
  fail "no /etc/nv_tegra_release (not L4T)"
fi
jp=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null)
[[ -n "$jp" ]] && ok "nvidia-jetpack $jp" || warn "nvidia-jetpack not installed (sudo apt install nvidia-jetpack: CUDA, cuDNN, TensorRT, VPI)"
. /etc/os-release 2>/dev/null
[[ "${VERSION_ID:-}" == "22.04" ]] && ok "Ubuntu $VERSION_ID" || fail "Ubuntu ${VERSION_ID:-?} (ROS 2 Humble needs 22.04)"
model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
[[ -n "$model" ]] && ok "model: $model"

echo "== Compute"
cuda=$( (nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null) | grep -o 'release [0-9.]*' | head -1)
[[ "$cuda" == "release 12.6" ]] && ok "CUDA ${cuda#release }" || warn "CUDA ${cuda#release }${cuda:-not found} (Isaac ROS 3.2 uses 12.6; add /usr/local/cuda/bin to PATH)"
vpi=$(ls -d /opt/nvidia/vpi[0-9]* 2>/dev/null | tr '\n' ' ')
case "$vpi" in
  *vpi3*) ok "VPI: $vpi(patch_upstream.sh keeps the upstream VPI 3 code)" ;;
  "")     warn "no VPI under /opt/nvidia (comes with nvidia-jetpack)" ;;
  *)      warn "VPI: $vpi(Isaac ROS 3.2 on Jetson expects VPI 3)" ;;
esac
trt=$(dpkg-query -W -f='${Version}' tensorrt 2>/dev/null)
[[ -n "$trt" ]] && ok "TensorRT $trt" || warn "TensorRT not installed (optional: YOLO .engine export)"
if have nvpmodel; then
  mode=$(nvpmodel -q 2>/dev/null | grep -i 'power mode' | head -1)
  [[ "$mode" == *MAXN* ]] && ok "${mode}" || warn "${mode:-power mode unknown} (MAXN: sudo nvpmodel -m 0; then sudo jetson_clocks)"
fi

echo "== Memory and storage"
mem_gb=$(awk '/^MemTotal:/ {printf "%.0f", $2 / 1048576}' /proc/meminfo)
ok "RAM ${mem_gb} GB (shared by CPU and GPU)"
swap=$(swapon --noheadings --show=NAME,SIZE 2>/dev/null | tr '\n' ' ')
[[ -n "$swap" ]] && ok "swap: $swap" || warn "no swap"
root_free=$(df -BG --output=avail / | tail -1 | tr -d ' G')
(( root_free >= 60 )) && ok "${root_free} GB free on /" || warn "${root_free} GB free on / (an NVMe SSD is recommended: workspace build, models, bags)"
systemctl is-active --quiet earlyoom 2>/dev/null && ok "earlyoom active" || warn "earlyoom not active (sudo apt install earlyoom)"

echo "== Robot I/O"
for m in gs_usb mttcan; do
  if modinfo "$m" > /dev/null 2>&1; then ok "kernel module $m available"
  elif [[ "$m" == gs_usb ]]; then fail "kernel module gs_usb missing (USB-CAN adapters of the Piper and Scout need it)"
  else ok "no $m (onboard CAN not built)"; fi
done
cans=$(ip -br link show type can 2>/dev/null | awk '{print $1 "(" $2 ")"}' | tr '\n' ' ')
[[ -n "$cans" ]] && ok "CAN interfaces: $cans" || warn "no CAN interfaces up (INSTALL.md 9.1)"
if lsmod 2>/dev/null | grep -q '^mttcan'; then
  warn "mttcan loaded: onboard CAN may hold can0/can1; pass piper_can_port:= / scout_can_port:= to full_system.launch.py or rename the USB adapters"
fi
if have lsusb; then
  rs=$(lsusb 2>/dev/null | grep -i '8086:' | head -1)
  [[ -n "$rs" ]] && ok "RealSense on USB: ${rs#*ID }" || warn "no RealSense on USB"
fi

echo "== ROS 2 and Python"
[[ -f /opt/ros/humble/setup.bash ]] && ok "ROS 2 Humble installed" || fail "/opt/ros/humble missing"
ok "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<default: rmw_fastrtps_cpp>}, ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
py=$(python3 - 2>/dev/null <<'PY'
import importlib
out = []
for mod in ("numpy", "scipy", "torch", "ultralytics", "piper_sdk"):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        if mod == "torch":
            v += f", cuda={m.cuda.is_available()}"
        out.append(f"{mod} {v}")
    except Exception as exc:  # noqa: BLE001
        out.append(f"{mod} MISSING ({type(exc).__name__})")
print("; ".join(out))
PY
)
echo "$py" | tr ';' '\n' | while read -r line; do
  case "$line" in
    *MISSING*)        warn "$line" ;;
    *"cuda=False"*)   fail "$line (install NVIDIA's Jetson torch wheel, not the PyPI one)" ;;
    "numpy 2"*)       fail "$line (ROS Humble needs numpy<2)" ;;
    *)                ok "$line" ;;
  esac
done
