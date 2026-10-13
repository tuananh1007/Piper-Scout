#!/usr/bin/env bash
# Runs every package's Python unit tests, one pytest per package, without a
# colcon install (sibling packages via PYTHONPATH), and prints a summary.
# GitHub Actions runs it (.github/workflows/unit-tests.yml); locally:
#
#   Codes/ci/run_unit_tests.sh                      # all packages
#   Codes/ci/run_unit_tests.sh stem_grasp plant_twin
#
# Tests skip what the environment lacks: ROS (launch tests), torch, Node +
# Playwright (the operator page), a C++ compiler (the field snapshot driver).
# Exit code 0 when no package has a failing test.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
PKGS=(stem_grasp scout_piper_bringup scout_piper_scene_repr plant_twin scout_piper_jepa
      scout_piper_whole_body_mpc scout_piper_active_perception scout_piper_language)
(( $# )) && PKGS=("$@")
export PYTHONPATH=".:$SRC/scout_piper_whole_body_mpc:$SRC/scout_piper_scene_repr/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
summary=() rc=0
for p in "${PKGS[@]}"; do
  echo "::group::$p"
  out=$(cd "$SRC/$p" && python3 -m pytest test -q -p no:cacheprovider 2>&1)
  r=$?
  echo "$out"
  echo "::endgroup::"
  last=$(echo "$out" | grep -E "passed|failed|error|no tests ran" | tail -1)
  summary+=("$( ((r == 0 || r == 5)) && echo PASS || echo FAIL )  $p: $last")
  ((r == 0 || r == 5)) || rc=1
done
echo
printf '%s\n' "${summary[@]}"
exit $rc
