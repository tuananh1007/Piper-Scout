# scout_piper_jepa — Piper-JEPA (Stage A target memory, Stage B prediction, Stage C cost)

Stage A keeps the exact operator-selected flower/peduncle identified across
camera motion, plant sway and temporary occlusion, and publishes a compact
target state. Stage B predicts how candidate robot motions change the dense
features (and so the target's location, visibility and identity); Stage C
turns those predictions into a cost for the whole-body MPC. Research plan:
[`research/piper_jepa/`](../../../research/piper_jepa/).

## Pieces

| Module | Role |
|---|---|
| `encoder.py` | `DenseEncoder` interface. `VJepaEncoder` (V-JEPA 2/2.1 via `torch.hub`, lazy torch import), `DinoV2Encoder` (DINOv2 patch tokens, the E1 baseline T2) and `ColorPatchEncoder` (numpy reference used by the tests; **not** a V-JEPA result) |
| `target_memory.py` | descriptor from the init mask; per frame: cosine similarity → gated softmax → image mean/covariance, entropy, confidence, `tracking`/`occluded`/`lost`; 3-D world estimate from aligned depth when valid and the camera pose is known |
| `metrics.py` | E1 metrics: ID retention, false-switch rate, centre error, jitter, occlusion recovery |
| `action.py` | action embedding Γ = [Δs_b, Δθ_b, Δp_ee, Δr_ee, Δg] in the base frame at t, from a command (`action_embedding`) or, batched, from two whole-body states (`action_from_states`, used for training data and MPC rollouts) |
| `readout.py` | §12 read-outs from (predicted) features: similarity, gated softmax, location û, entropy, identity cos(r̂, r), visibility; `readout_sequence` gates each predicted step around the previous one |
| `predictor.py` | `DensePredictor` interface, `PersistencePredictor`, `StateConditionedPredictor` (MPC state rollouts → Γ → predictor), `OracleStatePredictor` (synthetic upper bound) |
| `torch_predictor.py` | learned predictor (lazy torch): patch tokens + one token for the action and the joint angles (P_φ(Z, a, s), §10), pre-norm transformer, residual head (zero-initialised = persistence); P0 action- and state-free, P2 unweighted, P3 target-weighted (§11) by configuration; optional projection of the encoder features onto their top principal directions (stored in the checkpoint); `train_predictor` with L_TF + λ_R L_roll |
| `prediction_metrics.py` | E3 metrics per horizon: target error, visibility F1 and AUROC, identity accuracy / switch rate, target-region vs global latent error |
| `predictive_cost.py` | Stage C `JepaVisibilityCost`, an `ExtraTerm` for the whole-body MPC: J_vis (centring, entropy, FoV barrier, soft loss of visibility) + J_id, with a geometry anchor (below) |
| `synthetic.py` | synthetic dense-feature world: z-buffered surfel renderer for an eye-in-hand camera, flower + identical twin + occluding leaf + stems + wall, random-motion episodes |
| `episode.py` | rosbag2 → `.npz` export (frames, aligned depth, intrinsics, camera poses from the bag's TF, whole-body states from `/odom` + `/joint_states`, framewise masks, merged labels; two passes, TF first) and offline evaluation (`jepa_episode eval …`) |
| `annotate.py` | target / distractor / plant masks: PNG import from any tool, keyframe polygons with vertex interpolation (OpenCV window), `info` (`jepa_annotate`) |
| `e1.py` | E1 comparison T0 (framewise segmentation), T1 (Lucas–Kanade), T2 (DINOv2 + memory), T3 / T4 (V-JEPA 2 / 2.1 + memory) from `config/e1_methods.yaml`, and the H1 go/no-go rule (`jepa_e1`) |
| `train.py` | P3B.9: features per episode (cached), Γ and joint angles from states, masks to the grid, optional feature projection (`--proj-dim`), P0 / P2 / P3 training on CUDA, E3 scores vs persistence, checkpoints with the training frame interval (`jepa_train`; `--synthetic N` for a check without data) |
| `latency.py` | P3B.10: predictor rollout and predictive-cost time per sample count over the MPC horizon, fp16 option, untrained models of any feature size (`--grid`, `--feat-dim`, `--input-dim`) (`jepa_latency`) |
| `predictive_mpc_node.py` | C3 (P3B.11): the whole-body MPC node with the target memory, the learned predictor and the visibility cost in one process |
| `ground.py` | publish a target mask (PNG, box or circle) on `/piper_jepa/init_mask` (`jepa_ground`) |
| `image_codec.py` | `sensor_msgs/Image` ↔ numpy without cv_bridge; shared by the node and the bag exporter |
| `target_state_node.py` | ROS 2 node |

## Identity rule

Appearance alone cannot tell the selected flower from an identical neighbour.
The memory therefore gates the search to a window around the
constant-velocity prediction, coasts through occlusion with a bounded-speed
growing gate, and once `lost` it does **not** re-acquire by itself: publish a
new mask on `/piper_jepa/init_mask` to re-ground. Setting `prior_sigma_px <= 0`
disables prior and gate (the ablation).

## Stage B — action-conditioned prediction

```text
Z_{t-K+1:t} (dense features)  +  Γ(x_t, x_{t+1}) (action)
        └── P_φ (transformer over patch tokens + action token) ──▶ Ẑ_{t+1}, then autoregressively Ẑ_{t+2..t+H}
                     └── read-outs (§12) ──▶ û, Ĥ, visible, identity per step
```

P0 / P2 / P3 differ only in `ACPredictorConfig` (`.p0()`, `.p2()`, `.p3()`);
P3 weights the L1 loss by 1 + λ_T M_target + λ_P M_plant (λ_T = 20, λ_P = 2
by default) because a peduncle covers a handful of patches. P1 (the V-JEPA
2-AC baseline) needs the real encoder and is not implemented.

Training data are episodes of features, actions from consecutive states, and
target / plant masks (`synthetic.synthetic_episodes` builds them for the
synthetic world; on the robot they come from recorded bags + annotation,
P3B.8). `benchmarks/e3_synthetic.py` runs the E3 comparison end to end.

## Stage C — predictive cost for the whole-body MPC

`JepaVisibilityCost(predictor, image_hw, camera_pose=..., K=..., stride=...)`
is appended to `WholeBodyCost.extra`; each control cycle call
`set_context(Z_hist, r, u_now, p_world)` from the Stage A memory. Without a
context it adds nothing (the controller is C2).

**Geometry anchor (deviation from §12).** Appearance cannot separate the
selected flower from an identical twin: with the plain read-out, a motion
that hides the flower behind a leaf while bringing its twin into view scored
as "visible, identity ≈ 1" (unit test `test_geometry_anchor_catches_the_identical_twin`).
When the memory has a metric target position, the cost projects it into each
planned camera pose and reads the predicted features only there (16 px gate);
behind the camera or outside the image counts as out of view. The anchor is
the grounding position, refreshed only by agreeing depth estimates — a
tracker that slipped to the twin must not drag it along. Without depth the
cost falls back to the sequential read-out.

## Synthetic results (offline, 4-core x86 CPU; not robot evidence)

**E3** (`benchmarks/e3_synthetic.py`, defaults: 1,500 training / 300 test
random-motion episodes of 14 frames, 1,500 training steps per model,
predictor step 0.2 s, one seed; 416 test windows; flower visible in 47 % of
frames):

| Horizon 4 steps (0.8 s) | Persistence | P0 action-free | P2 unweighted | P3 target-weighted |
|---|---|---|---|---|
| Target error E_target(4) | 18.0 px | 21.6 px | 21.9 px | **13.0 px** |
| Visibility AUROC | 0.46 | 0.61 | 0.58 | **0.73** |
| Visibility F1 (threshold 0.6) | 0.66 | 0.00 | 0.06 | **0.71** |
| Identity accuracy | 0.19 | 0.16 | 0.16 | **0.43** |
| Target-region latent L1 | 4.42 | 3.63 | 3.58 | **2.21** |
| Global latent L1 | 3.35 | 2.39 | **2.24** | 2.39 |

| E_target (px) at 1 / 4 / 8 steps | Persistence | P0 | P2 | P3 |
|---|---|---|---|---|
| | 9.6 / 18.0 / 23.9 | 11.5 / 21.6 / 26.8 | 13.1 / 21.9 / 27.2 | **7.9 / 13.0 / 17.2** |

P3 against P2 shows the signature the plan predicts (§10.7): at 4 steps the
target-region error is 38 % lower while the global error is 7 % higher, i.e.
capacity moved from the background to the flower. P0 and P2 model the scene
better than persistence (lower global error) but locate the flower worse:
deterministic L1 training blurs a 1–2-cell target, which also collapses their
fixed-threshold F1. P3 is the only predictor that beats persistence on every
target metric, at every horizon. Caveats: synthetic features, one seed, a
small CPU model; P1 (V-JEPA 2-AC) is not in the comparison.

**Closed loop (C2 vs C3 with the oracle predictor), `benchmarks/visibility_mpc.py`:
inconclusive.** Pre-grasp goal between the flower and its twin, 2 seeds,
60 steps, MPPI with 64 samples:

| Method | Flower in view | Tracker ends on | Goal error |
|---|---|---|---|
| C2 geometry-only | 90–93 % | twin (2/2) | 0.3–1.0 cm |
| C3-oracle, sequential read-out | 75–87 % | twin (2/2) | 3.2–4.4 cm |
| C3-oracle, anchor refreshed by the tracker | 25–87 % | flower 1/2, twin 1/2 | 3.6–4.7 cm |
| C3-oracle, anchor from grounding | 72–78 % | twin / elsewhere | 8.7–17 cm |

The cost does what it is built to do on candidate rankings (unit tests), but
in closed loop it trades goal accuracy against a view of the flower that this
pre-grasp pose may not offer at all, and 64 samples do not find a better
compromise. Before claiming H4 this needs: a check that a view-preserving
reachable path exists in the test scene, visibility as a near-goal constraint
or a scheduled weight instead of a constant cost, more samples (GPU), and the
learned predictor in the loop (P3B.11).

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
cd Codes/src/scout_piper_jepa && python -m pytest test -q    # torch tests skip without torch
```

Synthetic two-flower scene with an identical distractor and an occluder:
identity kept (>90 % retention, no switches, recovery on the first visible
frame), the no-prior ablation is measurably more ambiguous, `lost` needs
re-grounding, metric 3-D only with valid depth, action embedding, image codec
and episode evaluation. Stage B / C (`test_stage_b.py`): z-buffered occlusion
and projection in the synthetic world, read-outs need the spatial prior
against the twin, Γ from states matches Γ from commands, strided predictor
actions, E3 metrics rank the true future above persistence, the visibility
cost prefers keeping the flower in view and the geometry anchor catches the
twin; with torch: the untrained network is exactly persistence and survives a
save/load round trip, and a trained P2 uses the action (beats persistence and
the action-free P0 on a shift world).

## Datasets, E1, training, latency, C3 (2026-10-09)

The path from robot to results, each step a command (MODULE_TASKS.md has the order):

```text
record_bag.sh e1|e3  →  jepa_episode export  →  jepa_annotate  →  jepa_e1 run (H1 go/no-go)
                                                               →  jepa_train (P0 / P2 / P3)  →  jepa_latency (Orin)
                                                                                             →  predictive_mpc_node (C3)
```

**Predictor step.** A checkpoint records the time between its training frames
(`step_s`); the MPC steps the predictor every `jepa_stride` MPC periods and
derives `jepa_stride` from `step_s` when it is 0 (the default). Export E3
episodes with a frame interval that is a multiple of the MPC period: with the
colour stream at 30 fps and the MPC at 10 Hz, `--stride 6` gives 0.2 s
(`jepa_stride` 2); the default `--stride 1` (0.033 s) matches no MPC step and
the node warns.

**Feature size.** The cost predicts samples × predictor steps × cells ×
channels features every MPPI iteration: with V-JEPA 2 ViT-L at 256 px (16 × 16
cells, 1024 channels), 256 samples and 10 steps that is 2.7 GB. `jepa_train
--proj-dim 64` projects the features onto their 64 main directions (fitted on
the training episodes, uncentred so cosines inside the subspace are kept,
stored in the checkpoint; the node, the cost and the target descriptor use it
automatically). `results.json` reports the energy kept (`proj_energy`).
On the development CPU, 32 samples × 10 steps took 1270 ms with 1024 channels
and 440 ms projected to 64 (`jepa_latency --grid 16 16 --feat-dim 1024 --state`
vs `--input-dim 1024 --feat-dim 64 --state`); MODULE_TASKS.md A8 / B4 measure
the GPU.

Checked here without a GPU: the export on a bag recorded from the fake robot
(37 frames, states for 34, camera poses for 35), annotation and E1 runner on
synthetic episodes (T0 / T1 / reference memory; V-JEPA and DINOv2 not
loaded), training on synthetic episodes (CPU smoke run only), latency on the
CPU (64 samples × 4 predictor steps ≈ 250 ms: a GPU is needed for 10 Hz), and
the predictive MPC node on the fake robot with a synthetic camera (cost
active in 216 of 216 cycles once grounded; persistence predictor).

## Not done yet

- V-JEPA and DINOv2 inference have not been run yet (the tests use only the
  numpy `ColorPatchEncoder`); the hub entry points in `config/e1_methods.yaml`
  are to be verified on the workstation (MODULE_TASKS.md A6).
- Touching/merged instances are not handled specially.
- Stage B/C run only on the synthetic world: no recorded E3 episodes (P3B.8),
  no V-JEPA features, no GPU timing (P3B.10), and the closed-loop C3 result
  above is inconclusive (P3B.11).
- Gripper state Δg is not wired (always 0).
