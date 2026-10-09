#!/usr/bin/env bash
# Memory-safe `colcon build`: use it wherever INSTALL.md says `colcon build`.
#
#   ./scripts/colcon_build_safe.sh --symlink-install [more colcon build arguments]
#
# Plain `colcon build` runs one package per CPU thread and `make -j<threads>`
# inside each, so on a 16-thread CPU it can start well over a hundred
# compilers at once. Each needs up to 1.25 GB (rclcpp / MoveIt C++ ~0.9 GB,
# nvblox CUDA 1.0-1.25 GB; INSTALL.md Step 6), which overruns 16 GB of RAM:
# the machine swaps until it freezes. This script sizes the parallelism from the memory that is
# free when it starts and passes everything else to `colcon build` unchanged.
#
# Environment overrides:
#   BUILD_JOBS=N          total compilers at once (default: from free memory)
#   BUILD_RESERVE_GB=3    memory left for the desktop, browser and IDE
#   BUILD_GB_PER_JOB=1.5  memory budgeted per compiler (measured peak 1.25)
#   CUDA_ARCHS=86         GPU architectures for CUDA code (default: the GPUs
#                         nvidia-smi reports, 87 on a JetPack 6 Jetson; "default"
#                         keeps the packages' own)
#   DRY_RUN=1             print the colcon command instead of running it
set -euo pipefail

cpus=$(nproc)
avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
# inside a container with a memory limit (compose.dev.yml), /proc/meminfo
# still shows the host: use what is left under the cgroup limit if smaller
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [[ -r "$f" ]] || continue
  limit=$(cat "$f")
  [[ "$limit" =~ ^[0-9]+$ ]] || break                      # "max": no limit
  used_f=${f%/*}/memory.current
  [[ -r "$used_f" ]] || used_f=${f%/*}/memory.usage_in_bytes
  used=$(cat "$used_f" 2>/dev/null || echo 0)
  left_kb=$(( (limit - used) / 1024 ))
  (( left_kb < avail_kb )) && avail_kb=$left_kb
  break
done
reserve=${BUILD_RESERVE_GB:-3}
per_job=${BUILD_GB_PER_JOB:-1.5}

if [[ -n "${BUILD_JOBS:-}" ]]; then
  jobs=$BUILD_JOBS
else
  jobs=$(awk -v a="$avail_kb" -v r="$reserve" -v p="$per_job" \
    'BEGIN { j = int((a / 1048576 - r) / p); print (j < 1 ? 1 : j) }')
fi
if (( jobs > cpus )); then jobs=$cpus; fi
if (( jobs < 1 )); then jobs=1; fi

# A few packages at a time, two compilers each: packages with many files
# (nvblox, realsense2_camera) still compile in parallel, and the total stays
# at `jobs`.
if (( jobs >= 4 )); then
  workers=$(( jobs / 2 )); make_jobs=2
else
  workers=1; make_jobs=$jobs
fi
export MAKEFLAGS="-j${make_jobs} -l${cpus}"
export CMAKE_BUILD_PARALLEL_LEVEL=$make_jobs

# CUDA: build only for the GPU in this machine. Isaac ROS otherwise compiles
# for five architectures (89;86;80;75;70): same memory per file, 2-3x the time.
archs=${CUDA_ARCHS:-}
if [[ -z "$archs" ]] && command -v nvidia-smi > /dev/null 2>&1; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | tr -d ' .' | grep -E '^[0-9]+$' | sort -u | paste -sd ';' || true)
fi
# Jetson: JetPack 6 (L4T r36) runs only on Orin modules, compute capability 8.7
if [[ -z "$archs" ]] && grep -q '^# R36' /etc/nv_tegra_release 2>/dev/null; then
  archs=87
fi
args=("$@")
if [[ -n "$archs" && "$archs" != "default" && " ${args[*]} " != *CMAKE_CUDA_ARCHITECTURES* ]]; then
  # colcon keeps only the last --cmake-args, so add to the user's list if there is one
  merged=() found=0
  for a in "${args[@]}"; do
    merged+=("$a")
    if [[ "$a" == "--cmake-args" && $found == 0 ]]; then
      merged+=("-DCMAKE_CUDA_ARCHITECTURES=${archs}"); found=1
    fi
  done
  (( found )) || merged+=(--cmake-args "-DCMAKE_CUDA_ARCHITECTURES=${archs}")
  args=("${merged[@]}")
fi

printf 'colcon_build_safe: %.1f GB free, %d CPU threads -> %d package(s) at a time x make -j%d = %d compilers%s\n' \
  "$(awk -v a="$avail_kb" 'BEGIN { print a / 1048576 }')" "$cpus" "$workers" "$make_jobs" \
  "$(( workers * make_jobs ))" "${archs:+; CUDA architectures: $archs}"
cmd=(colcon build --executor parallel --parallel-workers "$workers" "${args[@]}")
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "MAKEFLAGS=\"$MAKEFLAGS\" ${cmd[*]}"
  exit 0
fi
exec "${cmd[@]}"
