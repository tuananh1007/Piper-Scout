#!/usr/bin/env bash
# Records the topics a dataset needs into Codes/bags/<preset>_<date>_<time>/
# (rosbag2, default storage). Stop with Ctrl-C.
#
#   ./scripts/record_bag.sh scene  [name]   # P1.4.1 plant scenes for the semantic-scene benchmark
#   ./scripts/record_bag.sh e1     [name]   # P2A.5 Piper-JEPA tracking episodes (E1)
#   ./scripts/record_bag.sh e3     [name]   # P3B.8 Piper-JEPA prediction episodes (E3: robot motion)
#
# Every preset records colour, aligned depth, both camera_info topics, TF,
# joint states and odometry; "scene" adds the segmentation outputs, "e3" adds
# the commands that moved the robot. A JSON note with the preset, the git
# commit and the topic list is written next to the bag.
set -euo pipefail

preset="${1:-}"
name="${2:-}"
common=(/camera/color/image_raw /camera/color/camera_info
        /camera/aligned_depth_to_color/image_raw /camera/aligned_depth_to_color/camera_info
        /tf /tf_static /joint_states /joint_states_single /odom)
case "$preset" in
  scene) topics=("${common[@]}" /stem_grasp/mask /stem_grasp/semantic_label /stem_grasp/target_mask
                 /stem_grasp/pipeline_state) ;;
  e1)    topics=("${common[@]}" /piper_jepa/target_state /piper_jepa/init_mask) ;;
  e3)    topics=("${common[@]}" /cmd_vel /servo_node/delta_joint_cmds /servo_node/delta_twist_cmds
                 /whole_body_mpc/status /piper_jepa/target_state /piper_jepa/init_mask) ;;
  *) echo "usage: $0 scene|e1|e3 [name]" >&2; exit 2 ;;
esac
if ! command -v ros2 > /dev/null; then echo "source the ROS environment first" >&2; exit 2; fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/bags"
mkdir -p "$root"
out="$root/${preset}_$(date +%Y%m%d_%H%M%S)${name:+_$name}"
commit=$(git -C "$root/.." rev-parse --short HEAD 2>/dev/null || echo unknown)
printf '{"preset": "%s", "name": "%s", "commit": "%s", "topics": [%s]}\n' "$preset" "$name" "$commit" \
  "$(printf '"%s",' "${topics[@]}" | sed 's/,$//')" > "$out.json"
echo "recording ${#topics[@]} topics to $out (Ctrl-C to stop)"
exec ros2 bag record -o "$out" "${topics[@]}"
