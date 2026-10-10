#!/usr/bin/env bash
# Runs every hardware-free chain check in turn and prints a PASS/FAIL summary.
# Each check starts its own bringup with the fake arm and base (no CAN, no
# camera), runs the checker, and stops everything again.
#
#   cd Codes && source install/setup.bash
#   ./scripts/hardware_free_checks.sh                    # all checks, default MPC profile
#   ./scripts/hardware_free_checks.sh --profile orin     # on the Jetson AGX Orin
#   ./scripts/hardware_free_checks.sh --quick            # servo, MPC and one full grasp only
#   ./scripts/hardware_free_checks.sh --only 'effort|force'   # the force-estimate checks only
#
# Options: --profile default|orin   MPC profile (whole_body_mpc.launch.py profile:=)
#          --servo ibvs|mppi        stem_grasp servo controller for the grasp checks (Phase 2B)
#          --log-dir DIR            where logs go (default Codes/test_logs/hwfree_<date>_<time>,
#                                   which persists in the dev container too)
#          --quick                  skip the extra stem positions and the force-estimate checks
#          --only REGEX             run only the checks whose name matches (bash regex)
#
# Everything runs with ROS_LOCALHOST_ONLY=1 in its own ROS domain
# (HWF_DOMAIN, default 77), so it cannot reach a real robot on the network.
# Takes about 13 minutes (3 with --quick). Exit code 0 when every check passes.
set -uo pipefail

profile=default quick=0 servo=ibvs only=""
log_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/test_logs/hwfree_$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) profile="$2"; shift 2 ;;
    --log-dir) log_dir="$2"; shift 2 ;;
    --quick)   quick=1; shift ;;
    --servo)   servo="$2"; shift 2 ;;
    --only)    only="$2"; shift 2 ;;
    -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

if ! ros2 pkg prefix scout_piper_bringup > /dev/null 2>&1; then
  echo "scout_piper_bringup not found: build the workspace and 'source install/setup.bash' first" >&2
  exit 2
fi
mkdir -p "$log_dir"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="${HWF_DOMAIN:-77}"
ros2 daemon stop > /dev/null 2>&1; ros2 daemon start > /dev/null 2>&1

pids=()
start() {   # start <log> <command...>: run in its own process group
  local log="$1"; shift
  setsid "$@" > "$log" 2>&1 &
  pids+=("$!")
}
stop_all() {
  for p in "${pids[@]}"; do kill -INT -- -"$p" 2> /dev/null; done
  for _ in $(seq 20); do
    local alive=0
    for p in "${pids[@]}"; do kill -0 -- -"$p" 2> /dev/null && alive=1; done
    (( alive )) || break
    sleep 0.5
  done
  for p in "${pids[@]}"; do kill -9 -- -"$p" 2> /dev/null; done
  pids=()
  sleep 2
}
trap 'stop_all; exit 130' INT TERM

results=()
bringup_extra=()   # extra full_system.launch.py arguments for the next checks
run_check() {   # run_check <name> <with_base 0|1> <with_mpc 0|1> <pipeline args or -> <checker command...>
  local name="$1" base="$2" mpc="$3" pipe="$4"; shift 4
  [[ -n "$only" && ! "$name" =~ $only ]] && return
  local dir="$log_dir/$name"
  mkdir -p "$dir"
  echo "== $name"
  local bringup=(ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true
                 bringup_servo:=true bringup_rviz:=false bringup_pipeline:=false
                 ${bringup_extra[@]+"${bringup_extra[@]}"})
  (( base )) && bringup+=(bringup_base:=true fake_base:=true)
  start "$dir/bringup.log" "${bringup[@]}"
  (( mpc )) && start "$dir/mpc.log" ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py \
    execute:=true profile:="$profile"
  if [[ "$pipe" != "-" ]]; then
    # shellcheck disable=SC2086
    start "$dir/pipeline.log" ros2 run stem_grasp pipeline_node --ros-args \
      -p reach_executor:=whole_body_mpc -p servo_controller:="$servo" $pipe
  fi
  sleep 12                                            # let everything come up
  timeout 300 "$@" > "$dir/check.out" 2>&1
  local rc=$?
  stop_all
  grep -E '^(PASS|FAIL|REFUSED|info)' "$dir/check.out" | sed 's/^/   /' | cut -c1-160
  if (( rc == 0 )); then results+=("PASS  $name"); else results+=("FAIL  $name (exit $rc, see $dir)"); fi
}

run_check servo_chain 0 0 - ros2 run scout_piper_bringup servo_chain_check.py
run_check mpc_chain   1 1 - ros2 run scout_piper_bringup mpc_chain_check.py 1.0 0.2 0.4
run_check grasp_full_0.95_0.15 1 1 "-p approach_enabled:=true -p grasp_close_gripper:=true" \
  ros2 run scout_piper_bringup grasp_chain_check.py 0.95 0.15 --release
if (( ! quick )); then
  run_check mpc_chain_far 1 1 - ros2 run scout_piper_bringup mpc_chain_check.py 1.4 -0.2 0.35
  run_check grasp_servo_only_0.8_0.3 1 1 "" \
    ros2 run scout_piper_bringup grasp_chain_check.py 0.8 0.3 --servo-seconds 15
  run_check grasp_approach_1.1_-0.1 1 1 "-p approach_enabled:=true" \
    ros2 run scout_piper_bringup grasp_chain_check.py 1.1 -0.1
  run_check grasp_full_1.25_0.0 1 1 "-p approach_enabled:=true -p grasp_close_gripper:=true" \
    ros2 run scout_piper_bringup grasp_chain_check.py 1.25 0.0 --release
  # Contact force from the fake arm's joint efforts: estimate, calibration, then the
  # pipeline's force abort with that calibration (no F/T sensor on the platform)
  cal="$log_dir/effort_calibration.json"
  bringup_extra=(bringup_force_estimate:=true)
  run_check effort_chain 0 0 - ros2 run scout_piper_bringup effort_chain_check.py --calibrate "$cal"
  bringup_extra=(bringup_force_estimate:=true effort_calibration:="$cal")
  run_check grasp_force_abort_0.95_0.15 1 1 "-p force_tare_service:=/effort_force_estimator/tare" \
    ros2 run scout_piper_bringup grasp_chain_check.py 0.95 0.15 --push-force 3.0
  bringup_extra=()
fi
ros2 daemon stop > /dev/null 2>&1

echo
echo "Summary (MPC profile: $profile; servo: $servo; logs: $log_dir)"
printf '  %s\n' "${results[@]}" | tee "$log_dir/summary.txt"
grep -h "MPPI solves overrun" "$log_dir"/*/mpc.log 2> /dev/null | head -3 | sed 's/^/  warning: /'
! printf '%s\n' "${results[@]}" | grep -q '^FAIL'
