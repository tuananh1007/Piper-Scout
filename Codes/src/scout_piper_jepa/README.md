# scout_piper_jepa — Piper-JEPA Stage A (dense target memory)

Keeps the exact operator-selected flower/peduncle identified across camera
motion, plant sway and temporary occlusion, and publishes a compact target
state. Research plan: [`research/piper_jepa/`](../../../research/piper_jepa/).

## Pieces

| Module | Role |
|---|---|
| `encoder.py` | `DenseEncoder` interface. `VJepaEncoder` (V-JEPA 2/2.1 via `torch.hub`, lazy torch import) and `ColorPatchEncoder` (numpy reference used by the tests; **not** a V-JEPA result) |
| `target_memory.py` | descriptor from the init mask; per frame: cosine similarity → gated softmax → image mean/covariance, entropy, confidence, `tracking`/`occluded`/`lost`; 3-D estimate from aligned depth when valid |
| `metrics.py` | E1 metrics: ID retention, false-switch rate, centre error, jitter, occlusion recovery |
| `action.py` | Stage B action embedding Γ(x,u) = [Δs_b, Δθ_b, Δp_ee, Δr_ee, Δg] in the base frame at t |
| `episode.py` | rosbag2 → `.npz` export and offline evaluation (`jepa_episode eval …`) |
| `target_state_node.py` | ROS 2 node |

## Identity rule

Appearance alone cannot tell the selected flower from an identical neighbour.
The memory therefore gates the search to a window around the
constant-velocity prediction, coasts through occlusion with a bounded-speed
growing gate, and once `lost` it does **not** re-acquire by itself: publish a
new mask on `/piper_jepa/init_mask` to re-ground. Setting `prior_sigma_px <= 0`
disables prior and gate (the ablation).

## ROS interface

```text
in   /camera/color/image_raw                   sensor_msgs/Image
in   /camera/aligned_depth_to_color/image_raw  sensor_msgs/Image (must be aligned)
in   /camera/color/camera_info                 sensor_msgs/CameraInfo
in   /piper_jepa/init_mask                     sensor_msgs/Image mono8  (re)ground
out  /piper_jepa/target_state                  std_msgs/String (JSON)
out  /piper_jepa/target_point                  geometry_msgs/PointStamped (valid depth only)
out  /piper_jepa/target_visible                std_msgs/Bool
out  /piper_jepa/debug_similarity              sensor_msgs/Image (publish_debug)
```

```bash
ros2 launch scout_piper_jepa target_state.launch.py
```

To use V-JEPA, set `encoder: vjepa` and `vjepa_hub_entry` in
`config/target_memory.yaml` from the `facebookresearch/vjepa2` README for the
checkpoint you deploy (ViT-B/16 384 first), and retune `visible_similarity`:
cosine scales differ between encoders.

## Tests

```bash
cd Codes && python -m pytest src/scout_piper_jepa/test -q
```

Synthetic two-flower scene with an identical distractor and an occluder:
identity kept (>90 % retention, no switches, recovery on the first visible
frame), the no-prior ablation is measurably more ambiguous, `lost` needs
re-grounding, metric 3-D only with valid depth, action embedding, image codec
and episode evaluation.

## Not done yet

- V-JEPA inference has not been run here (no torch/GPU in this container);
  the hub entry point is configuration until verified on the Orin.
- Touching/merged instances are not handled specially.
- Stage B predictor training (`action.py` is only the action embedding).
