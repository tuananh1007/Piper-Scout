# Module tasks: semantic scene, Piper-JEPA, whole-body MPC

The three research modules are implemented to the point where the remaining
work needs a GPU, the Jetson, the robot or recorded data. This file lists that
work as tasks to run, in order: **Part A on the workstation (PC) first, Part B
on the Jetson AGX Orin, Part C on the robot, Part D back on the PC with the
recorded data.** Each task names the module, the PROGRESS.md items it closes,
the commands, what to expect and what to record.

**Before starting:** [`TEST_PROCEDURE.md`](TEST_PROCEDURE.md) W1–W4 passed on
the PC (build, unit tests, hardware-free chains). Part B needs J1–J6, Part C
needs R0–R8 of the same file.

**Contents:** [Conventions](#conventions) · [What is implemented](#what-is-implemented) ·
[Part A — Workstation](#part-a--workstation-rtx-3060) · [Part B — Jetson AGX Orin](#part-b--jetson-agx-orin) ·
[Part C — Robot](#part-c--on-the-robot) · [Part D — Analysis on the PC](#part-d--analysis-on-the-pc-recorded-data) ·
[Results log](#results-log)

## Conventions

- Run everything from `Codes/` (`/workspace` in the dev container) with the
  workspace sourced: `source install/setup.bash`. Rebuild after pulling:
  `./scripts/colcon_build_safe.sh --symlink-install`.
- Logs: one folder per day, as in TEST_PROCEDURE.md:

  ```bash
  export LOG=$PWD/test_logs/$(date +%Y-%m-%d) && mkdir -p $LOG
  ```

  Save each command's output with `| tee $LOG/<task>_<name>.txt` and copy
  the [results log](#results-log) into `$LOG/MODULE_RESULTS.md`.
- Benchmarks are plain Python scripts; with the workspace sourced they need no
  `PYTHONPATH`. They print one JSON line per run and a table at the end.
- Stop at the first unexpected failure, keep its output, and continue with a
  different module only if the failure does not concern it.

## What is implemented

| Module | Implemented and checked here (hardware-free, CPU) | Needs GPU / Orin / robot / data |
|---|---|---|
| Whole-body MPC | numpy MPPI + safety filter; **torch backend** (same model and cost on CUDA, `profile:=gpu` / `orin_gpu`); **W2 reactive QP** baseline; **30-scene plant benchmark** (W3 reaches 21/22 arm-unreachable targets: P3.3.2 gate passed, synthetic); **calibration tools** (slip, TCP pivot, hand-eye); semantic field from a topic; unknown-space policy `no_entry` | GPU / Orin timing (A2, B2), calibration on the robot (C2), runs on the robot (C3, C9) |
| Semantic scene | CPU map + query + MoveIt plugin; **merged label image** from stem_grasp (P1.1.4); **`other` class** and **finger self-filter** in the demux / CPU map (P1.7.8); **nvblox ESDF bridge** → `/scene_repr/distance_field` (P1.3.1, P3.2.2); **target attractor** → `/scene_repr/target_goal` (P1.3.3); **RViz config** (P1.2.3); **semantic vs occupancy** benchmark, synthetic 18/20 vs 17/20 and an offline mode for recorded scenes (P1.4.2); `record_bag.sh scene` (P1.4.1) | first run of the per-class nvblox stack and the bridge (A4, B3), Orin rate (B3), recorded scenes (C5, D1), robot runs (C4) |
| Piper-JEPA | Stage A memory, Stage B predictors, Stage C cost; **DINOv2 encoder** (T2); **E1 runner T0–T4** with the H1 go/no-go rule (P2A.6); **bag → episode export** with depth, poses, states, segmentation, labels; **annotation tool** (P2A.5, P3B.8); **training command** with GPU support (P3B.9); **latency benchmark** (P3B.10); **predictive MPC node C3** (P3B.11) | V-JEPA / DINOv2 inference (A6), GPU training and latency (A7, A8, B4), datasets (C6, C7), H1 decision (D2), E3 training (D3), C3 on the robot (C8) |
| Contact force | no F/T sensor: **joint-effort force estimate** (`dynamics/effort.py`, `effort_force_node` on `/ft_sensor/raw`, `~/tare`), **`calibrate_effort`** (move to a checked centre, multi-sine, fit, held-out residual, 3-σ threshold); relay passes driver efforts; fake driver simulates efforts and a TCP force; stem_grasp `force_topic`, tare on servo start, stale-force stop, thresholds into the MPPI servo; hardware-free: 0.04 N error on 3 N, calibration gains 1.00, 3 N push aborts the servo in 0.2 s | real driver efforts and noise, calibration on the robot (C2 step 4) |
| Phase 2B visual servo | arm-only **MPPI visual servo** (`visual_servo.py`, P2.1–P2.2) in stem_grasp `servo_controller: mppi`: image, view, joint, manipulability, clearance, smoothness, force and approach costs; projective prediction or online image Jacobian; JointJog out; full hardware-free grasp passes; **semantic clearance** from `/scene_repr/distance_field` with the target stem released (`exclude_target`: obstacle voxels connected to the grasp point; neighbours stay hard), hardware-free grasp with a synthetic field passes | CUDA / Orin cycle time (A13, B7), servo comparison on the robot (C9) |

---

## Part A — Workstation (RTX 3060)

### A1 — GPU check for torch

**Module:** all. **Steps:**

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee $LOG/A1_torch.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a $LOG/A1_torch.txt
```

**Pass:** `True` and `NVIDIA GeForce RTX 3060`. **If it prints `False`:**
install a CUDA build: `python3 -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu124`
and repeat (in the dev container this is lost when it exits: add it to
`docker/Dockerfile.dev` once it works).

### A2 — Whole-body MPC timing, numpy vs torch CPU vs torch CUDA (P3A.8 workstation half)

**Steps:**

```bash
python3 src/scout_piper_whole_body_mpc/benchmarks/timing.py --backends numpy,torch-cpu,torch-cuda \
  --samples 128,256,512,1024,2048,4096 --solves 20 --out $LOG/A2_timing.jsonl | tee $LOG/A2_timing.txt
```

**Expect:** a table of median / p95 ms per backend, geometry (none, plant
field) and sample count. Reference (4-core x86, no GPU, one thread): numpy 256
samples 93 ms without geometry and 232 ms with the plant field; torch-cpu 27 ms
and 68 ms.
**Then:** set `samples` in `src/scout_piper_whole_body_mpc/config/whole_body_mpc_gpu.yaml`
to the largest torch-cuda value whose p95 with the plant field is under 90 ms
(record it). **Record:** the table, the chosen sample count.

### A3 — Hardware-free chains on the GPU profile

**Steps:** `./scripts/hardware_free_checks.sh --profile gpu 2>&1 | tee $LOG/A3_hwfree_gpu.txt`
**Pass:** ten `PASS` lines; no `MPPI solves overrun` warning.
**Record:** the two `MPPI solve median / max` lines.

### A4 — Per-class nvblox and the distance-field bridge (P1.2.1, P1.3.1, first run)

**Module:** semantic scene. **Goal:** the five class mappers run, the bridge
merges their ESDFs, and the planner-facing topics publish. Needs the D405 on
the PC and a green object in view (a stand-in stem).

**Steps**, one terminal each:

1. `ros2 launch scout_piper_scene_repr realsense_nvblox.launch.py` (camera; its single map is not used here)
2. `ros2 run stem_grasp segmentation_node --ros-args -p mask_mode:=hsv_green`
3. `ros2 run tf2_ros static_transform_publisher --frame-id odom --child-frame-id camera_link`
4. `ros2 run tf2_ros static_transform_publisher --frame-id odom --child-frame-id base_link --x -0.6`
   (stands in for the robot base: the target goal's approach axis comes from it)
5. `ros2 launch scout_piper_scene_repr nvblox_semantic.launch.py classes:=stem,target,other rviz:=true 2>&1 | tee $LOG/A4_semantic.txt`
6. After 30 s:

   ```bash
   ros2 topic echo --once /stem_grasp/semantic_label --field encoding
   ros2 topic hz /scene_repr/depth/stem | tee $LOG/A4_depth_hz.txt                       # 10 s
   ros2 topic echo --once --field data /scene_repr/bridge_status std_msgs/msg/String | tee $LOG/A4_bridge.txt
   ros2 topic echo --once /scene_repr/target_goal | tee $LOG/A4_target_goal.txt          # needs a target mask
   nvidia-smi --query-gpu=memory.used --format=csv | tee $LOG/A4_gpu.txt
   free -m | tee -a $LOG/A4_gpu.txt
   ```

**Pass:** the label image is `mono8`; `bridge_status` shows `"ok"` for each
running class and `known_voxels` > 0; RViz (`semantic_scene.rviz`) shows the
stem mesh where the green object is; the PC keeps at least 2 GB of RAM free.
**Record:** the bridge status (`merge_ms`, `request_s`, the answer sizes in
`answers`), GPU memory with three class mappers, free RAM. With the default
0.6 m box the 3 mm stem and target answers are about 8 M voxels (32 MB) each
per request; `request_s` well below 1 s keeps the field within the MPC's
`field_max_age_s` (2.5 s). `target_goal` publishes only with a target mask (`grounded_sam`
mode or a multi-class YOLO model): note whether it was tested.
**If it fails:** `nvblox_msgs not found` → nvblox is not built (INSTALL.md
10.10 item 2). `"no service"` for every class → the mappers did not start
(see `A4_semantic.txt`); `"failed"` → the AABB is outside the map (check
the TF from step 3).

### A5 — MPC on the semantic field (P3.2.2, dry run)

**Module:** whole-body MPC + semantic scene. With A4 still running and the
hardware-free bringup in another terminal
(`ros2 launch scout_piper_bringup full_system.launch.py bringup_arm:=true fake_arm:=true bringup_base:=true fake_base:=true bringup_pipeline:=false bringup_rviz:=false`;
stop A4's step 3 and 4 first, the bringup publishes odom and the camera TF):

```bash
ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py profile:=gpu \
  field_topic:=/scene_repr/distance_field 2>&1 | tee $LOG/A5_mpc.txt
```

Send a goal (`ros2 run scout_piper_bringup print_tcp.py --forward 0.1`, run the
printed line) and save `ros2 topic echo --field data /whole_body_mpc/status std_msgs/msg/String | tee $LOG/A5_status.txt`.
**Pass:** status messages carry a numeric `geometry_age_s` (the field is used)
and `min_clearance_m` once spheres are near mapped geometry; no `watchdog`
after the first field (the MPC stops until one arrives, and when the newest
is older than `field_max_age_s`, 2.5 s).

### A6 — Encoders: V-JEPA 2, V-JEPA 2.1, DINOv2 (P2A.1 check)

**Module:** Piper-JEPA. **Steps:** load each hub entry once (downloads the weights):

```bash
python3 - <<'EOF' 2>&1 | tee $LOG/A6_encoders.txt
import numpy as np, time
from scout_piper_jepa.encoder import make_encoder
clip = [np.random.randint(0, 255, (480, 640, 3), np.uint8)] * 2
for kind, kw in [("dinov2", {"hub_entry": "dinov2_vits14"}),
                 ("vjepa", {"hub_entry": "vjepa2_vit_large", "image_size": 256})]:
    enc = make_encoder(kind, device="cuda", **kw)
    Z = enc.encode(clip); t = time.perf_counter()
    for _ in range(10): enc.encode(clip)
    print(kind, kw, Z.shape, f"{100 * (time.perf_counter() - t):.0f} ms per frame")
EOF
```

**Pass:** both print a feature grid (e.g. `(32, 32, 384)` and `(16, 16, 1024)`).
**Then:** find the V-JEPA 2.1 entry point in the facebookresearch/vjepa2
README and put it into `src/scout_piper_jepa/config/e1_methods.yaml` (T4,
`hub_entry`, `image_size`); rerun this task with it. **Record:** entry
points, grid shapes, ms per frame. **If a hub entry fails:** the name differs
in the installed release; take it from the README and record it.

### A7 — E3 on the synthetic world, full size on the GPU (P3B.5 rerun)

```bash
python3 -m scout_piper_jepa.train --synthetic 1500 --steps 1500 --device cuda \
  --out-dir $LOG/A7_e3_synthetic 2>&1 | tee $LOG/A7_e3.txt
```

**Expect:** the table of persistence / P0 / P2 / P3; P3 should again have the
lowest `E_target@4` and `target_L1@4` (CPU reference: P3 13.0 px vs 18.0 px
for persistence). **Record:** the table and the training time per model.

### A8 — Predictor and predictive-cost latency on the GPU (P3B.10, PC half)

```bash
python3 -m scout_piper_jepa.latency --device cuda --samples 32,64,128,256,512 --out $LOG/A8_latency.jsonl | tee $LOG/A8_latency.txt
python3 -m scout_piper_jepa.latency --device cuda --fp16 --samples 64,256 | tee -a $LOG/A8_latency.txt
# V-JEPA-sized features (ViT-L at 256 px: 16 x 16 x 1024), raw and projected to 64 channels (D3 --proj-dim 64)
python3 -m scout_piper_jepa.latency --device cuda --grid 16 16 --feat-dim 1024 --state --samples 32,64,128,256 \
  --out $LOG/A8_latency.jsonl | tee -a $LOG/A8_latency.txt
python3 -m scout_piper_jepa.latency --device cuda --grid 16 16 --input-dim 1024 --feat-dim 64 --state \
  --samples 32,64,128,256,512 --out $LOG/A8_latency.jsonl | tee -a $LOG/A8_latency.txt
```

**Expect:** the largest sample count whose `cost` p95 stays under ~40 ms
(two MPPI iterations per 100 ms step). The rollouts cover the 20-step MPC
horizon at a predictor step of 2 MPC steps (10 predictor steps; with
`--model`, the stride comes from its training frame interval). CPU reference:
64 samples × 4 predictor steps took 250 ms; 32 samples × 10 steps of
16 × 16 features took 1270 ms with 1024 channels, 440 ms projected to 64.
**Record:** the table per feature size; the chosen `samples` for the
predictive MPC; whether raw V-JEPA features fit at all (if not, D3 needs
`--proj-dim`).

### A9 — Closed loop C2 vs C3 on the synthetic world (P3B.7 rerun)

```bash
python3 src/scout_piper_jepa/benchmarks/visibility_mpc.py --seeds 3 --samples 256 \
  --methods C2,C3-oracle,C3c-oracle,C3c-learned --model $LOG/A7_e3_synthetic/P3.pt 2>&1 | tee $LOG/A9_c2_c3.txt
# starts where geometry-only motion loses the flower (C2 screened on 30 random starts)
python3 src/scout_piper_jepa/benchmarks/visibility_mpc.py --hard 30 --hard-k 5 --samples 256 \
  --methods C2,C3c-oracle,C3c-learned --model $LOG/A7_e3_synthetic/P3.pt 2>&1 | tee $LOG/A9_c2_c3_hard.txt
```

**Expect:** the feasibility line (share of goal poses that see the flower),
then rows per method. `C3-*` adds the visibility cost to the goal cost (the
2026-10-07 setup, which gave up 3–15 cm of goal error); `C3c-*` lets it cost
at most 1 cm (`--tol`). The question: does C3c keep the flower in view more
often, and the tracker on it, without a larger goal error, above all on the
hard starts. CPU reference (64 / 128 samples, default start): C3c keeps the
goal error at 1.3–1.8 cm but does not see the flower more than C2 (C2 already
keeps it in view 81–95 % there). **Record:** visible fraction, tracker end
state and goal error per method and start; samples.

### A10 — E1 pilot with a hand-held camera (P2A.5 / P2A.6 rehearsal)

**Module:** Piper-JEPA. Rehearses the whole E1 pipeline before robot time.
With the D405 on the PC and two identical flowers (or two red balls on
sticks) and a leaf that can be moved in front of one:

1. Camera and segmentation as in A4 steps 1–2; record while moving the camera
   by hand around the scene, hiding the chosen flower behind the leaf twice:
   `./scripts/record_bag.sh e1 pilot` (Ctrl-C after ~30 s).
2. Export: `python3 -m scout_piper_jepa.episode export bags/e1_*_pilot $LOG/A10_pilot.npz --stride 2`
3. Annotate the chosen flower on every 5th frame (hidden frames: `o`):
   `python3 -m scout_piper_jepa.annotate polygons $LOG/A10_pilot.npz --every 5`;
   then the twin as a distractor (run `polygons` with a copy, or import PNG
   masks from another tool with `--key distractors`).
4. `python3 -m scout_piper_jepa.annotate info $LOG/A10_pilot.npz`
5. `python3 -m scout_piper_jepa.e1 run $LOG/A10_pilot.npz --device cuda --out $LOG/A10_e1.json | tee $LOG/A10_e1.txt`

**Pass:** the export lists `frames, stamps, depth, K` (no states without the
robot); the E1 table prints a row per method (T4 only once A6 set its entry).
**Record:** the table and the time annotation took per 100 frames.

### A11 — Plant-scene benchmarks on the PC (P3.3, P1.4.2 synthetic reruns)

```bash
python3 src/scout_piper_whole_body_mpc/benchmarks/plant_scenes.py --scenes 30 --jobs 6 \
  --out $LOG/A11_plant_scenes.jsonl | tail -12 | tee $LOG/A11_plant_scenes.txt
python3 src/scout_piper_whole_body_mpc/benchmarks/semantic_vs_occupancy.py --scenes 20 --jobs 6 \
  --out $LOG/A11_sem_occ.jsonl | tail -8 | tee $LOG/A11_sem_occ.txt
```

**Expect** (reference, 4-core x86): plant scenes W0 8/30, W1 30/30, W2 28/30,
W3 29/30, exit gate 95 % → PASS; semantic vs occupancy 18/20 vs 17/20. Runs
take about 10 and 5 minutes with 6 processes; keep ≥ 2 GB free (each process
holds ~0.5 GB).

### A12 — Calibration tools on the fake robot (rehearsal of C2)

```bash
ros2 run scout_piper_bringup fake_scout_base.py --ros-args -p slip_k_v:=0.9 -p slip_k_omega:=0.75 &
ros2 run scout_piper_whole_body_mpc calibrate_slip --execute --out $LOG/A12_slip.json | tee $LOG/A12_slip.txt
kill %1
```

**Pass:** `k_v` ≈ 0.9 and `k_omega` ≈ 0.75 (within 0.03): the fake base's
odometry plays the external reference here. **Record:** the fitted values.

Joint-effort calibration and the force estimate on the fake arm (INSTALL.md 10.14):

```bash
./scripts/hardware_free_checks.sh --only 'effort|force' --log-dir $LOG/A12_force | tee $LOG/A12_force.txt
```

**Pass:** `PASS  effort_chain` and `PASS  grasp_force_abort_0.95_0.15`.
**Record:** the estimate errors, the calibration's gains, held-out residual and
3-σ threshold, the abort delay.

### A13 — MPPI visual servo: cycle time and the hardware-free grasp (P2.1.1, P2.2.2)

```bash
python3 src/scout_piper_whole_body_mpc/benchmarks/visual_servo_timing.py --samples 256,512,1024 \
  --out $LOG/A13_vs_timing.jsonl | tee $LOG/A13_vs_timing.txt
./scripts/hardware_free_checks.sh --servo mppi 2>&1 | tee $LOG/A13_hwfree_mppi.txt
```

**Expect:** torch-CUDA median under 10 ms at 512 samples (CPU reference:
numpy ~50 ms at 256, torch-CPU ~27 ms at 512); all grasp checks `PASS` with
the MPPI servo. **Record:** the timing table; servo approach time and the
"gripper axis through the stem" lines; with a GPU, rerun the grasp checks with
`-p mppi_backend:=torch -p mppi_samples:=512` added to the pipeline (edit
`hardware_free_checks.sh` or run `grasp_chain_check.py` by hand).

---

## Part B — Jetson AGX Orin

Prerequisite: TEST_PROCEDURE.md J1–J6 passed on the Orin.

### B1 — torch with CUDA on the Orin

`python3 -c "import torch; print(torch.cuda.is_available())"` prints `True`
(TEST_PROCEDURE.md J3). Record torch version.

### B2 — MPC timing on the Orin (P3A.8 / WE7)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
python3 src/scout_piper_whole_body_mpc/benchmarks/timing.py --backends numpy,torch-cpu,torch-cuda \
  --samples 64,128,256,512,1024 --solves 20 --out $LOG/B2_timing.jsonl | tee $LOG/B2_timing.txt
```

**Then:** choose the profile for the robot: `orin_gpu` (torch on the GPU) if its
p95 with the plant field is under 90 ms at ≥ 256 samples, else `orin` (numpy,
128 samples). Set `samples` in `config/whole_body_mpc_orin_gpu.yaml` (or
`_orin.yaml`) accordingly and run `./scripts/hardware_free_checks.sh --profile <chosen>`
(ten PASS). **Record:** the table, the chosen profile and samples.

### B3 — nvblox rate and memory on the Orin (P1.1.3, P1.7.6 timing)

As A4 on the Orin with all five classes
(`nvblox_semantic.launch.py` default classes) and `tegrastats` running.
nvblox prints its rates to the console every 10 s (`print_rates_to_console`).
**Pass (P1.1.3):** depth integration of each mapper < 33 ms (≥ 30 Hz input
keeps up). **Record:** integration ms per class, bridge `merge_ms`, RAM and
GPU load from `tegrastats`, number of classes that fit.

### B4 — Predictor latency on the Orin (P3B.10)

As A8 with `--device cuda` on the Orin, with the A7 model
(`--model .../P3.pt`) and the two V-JEPA-sized runs (raw and projected); later
with the D3 model. **Record:** the table; the sample count for C3 on the
Orin (cost p95 under ~40 ms) per feature size.

### B5 — Encoder speed on the Orin

As A6 on the Orin (V-JEPA entry chosen in A6, `fp16`). **Record:** ms per frame;
whether 10 Hz is reachable (the C3 node encodes every colour frame it gets).

### B6 — Predictive MPC node, hardware-free on the Orin

With the hardware-free bringup on the Orin, a camera stream (D405) and the A7
model:

```bash
ros2 run scout_piper_jepa predictive_mpc_node --ros-args \
  --params-file $(ros2 pkg prefix scout_piper_whole_body_mpc)/share/scout_piper_whole_body_mpc/config/whole_body_mpc.yaml \
  -p backend:=torch -p samples:=<B4 choice> -p encoder:=color_patch
```

Ground the target with a mask the size of the colour image, e.g. a circle
around it (pixel centre and radius from RViz):
`ros2 run scout_piper_jepa jepa_ground --circle <u> <v> <r>` (or `--box`, or
`--png mask.png`); send a goal; watch `ros2 topic echo /piper_jepa/mpc_context`.
**Pass:** `cost_active: true` while the target is visible, MPC `solve_ms`
under 90. (A model trained on `color_patch` features is needed for `-p jepa_model:=`;
the A7 synthetic model has a different grid, so run without `jepa_model` here.)

### B7 — MPPI visual servo cycle time on the Orin (P2.2.2)

As A13 on the Orin: `visual_servo_timing.py --samples 256,512` and
`hardware_free_checks.sh --profile orin --servo mppi`. **Pass:** torch-CUDA
median under 10 ms at the chosen sample count. **Record:** the table; the
`mppi_samples` / `mppi_backend` for the robot.

---

## Part C — On the robot

Prerequisite: TEST_PROCEDURE.md R0–R8 passed (CAN, arm, servo, gripper,
base, all three, whole-body MPC). Safety rules R0 apply to every task.

### C1 — Self-filter check (P1.7.8)

With the R7 bringup, the CPU scene node and the gripper in the camera view:

```bash
ros2 run scout_piper_scene_repr scene_query_node.py
ros2 topic echo --once --field data /scene_repr/map_status std_msgs/msg/String
```

**Pass:** `self_filtered_px` > 0 while the fingers are in view, and in RViz
(`/scene_repr/voxels`) no `other` voxels at the fingers. **If not:** adjust the
boxes (`self_filter_boxes`, "frame:cx,cy,cz,sx,sy,sz" in metres, link7 /
link8 frames) until the finger voxels disappear; record the final values.

### C2 — Calibration (P3A.7 / WE1, P0.2.6)

1. **TCP:** fix a cone tip (or a marked point) in the workspace; with the
   bridge disabled and the arm moved by hand or by servo, touch it with the
   gripper tip from at least 4 orientations ≥ 30° apart:
   `ros2 run scout_piper_whole_body_mpc calibrate_tcp --out $LOG/C2_tcp.json`.
   Expect `tcp_offset_m` near 0.14 and an RMS of a few mm; put it into
   `whole_body_mpc.yaml` (`tcp_offset_m`) and stem_grasp `tcp_offset_m`.
2. **Hand-eye:** print a 4×4 ArUco marker (DICT_4X4_50, measure its side),
   fix it in front of the robot, move the arm to ≥ 8 views:
   `ros2 run scout_piper_whole_body_mpc calibrate_hand_eye --aruco --marker-length <m> --out $LOG/C2_hand_eye.json`.
   It prints the `<origin>` for the camera joint and the difference to the
   current URDF; put it into `scout_piper_description` and rebuild.
   Without OpenCV's aruco module use an AprilTag / ArUco detector node and
   `--board-topic`.
3. **Slip:** needs an external pose reference (motion capture, or a fixed
   camera tracking an AprilTag on the base, publishing PoseStamped). Area clear
   (the plan stays within ~1 m of the start):
   `ros2 run scout_piper_whole_body_mpc calibrate_slip --execute --pose-topic <topic> --pose-type pose --out $LOG/C2_slip.json`.
   Put `k_v`, `k_omega` into `whole_body_mpc.yaml`. With `/odom` only the fit
   is ≈ 1 and says nothing (the tool warns).
4. **Joint efforts → contact force** (INSTALL.md 10.14): first check that the
   driver reports efforts, `ros2 topic echo /joint_states_single --field effort --once`
   (seven numbers, changing when the arm is pushed lightly by hand with the
   bridge disabled). Nothing near the arm, gripper free, servo started and the
   bridge enabled; dry run, then
   `ros2 run scout_piper_whole_body_mpc calibrate_effort --execute --center 0,0.8,-1.2,0,0.45,0 --payload-kg <camera + mount> --out $LOG/C2_effort.json`.
   Expect joints 2–5 identified with gains of one sign, held-out residual close
   to the in-sample one (a much larger one means the model misses something:
   cable forces, a payload, inertia), and `threshold_3sigma_n`. Then launch
   with `bringup_force_estimate:=true effort_calibration:=$LOG/C2_effort.json`,
   tare (`ros2 service call /effort_force_estimator/tare std_srvs/srv/Trigger`)
   and hang a known mass from the gripper (e.g. 200 g ≈ 1.96 N): the estimate's
   z in `/ft_sensor/raw` should read about −1.96 N. Set stem_grasp
   `contact_threshold_n`, `max_force_n` and plant_twin `contact_threshold_n`
   above `threshold_3sigma_n`.

**Record:** all four results and the residuals; for the force, the 3-σ
threshold and the known-mass reading.

### C3 — MPC with the chosen profile on the robot

Repeat TEST_PROCEDURE.md R8 with `profile:=<B2 choice>` after C2.
**Record:** as R8; compare with the first R8 run.

### C4 — Semantic scene on the robot (P1.6.9-adjacent, P3.2.2)

1. R7 bringup; `stem_grasp.launch.py` with the test config of R9 (segmentation
   publishes `/stem_grasp/semantic_label`); `nvblox_semantic.launch.py rviz:=true`
   on the Orin.
2. Check: `/scene_repr/bridge_status` ok per class, stems and leaves in RViz
   where they are, fingers not mapped.
3. MPC dry run on the field (A5 on the robot), then with `execute:=true` toward
   a goal next to the plant: the robot keeps its clearance from stems
   (`min_clearance_m` in the status) and the leaf penetration stays below 2 cm.
4. Target goal: with a target mask, `ros2 topic echo /scene_repr/target_goal`;
   let the MPC follow it directly (dry run first):
   `ros2 launch scout_piper_whole_body_mpc whole_body_mpc.launch.py profile:=<B2> field_topic:=/scene_repr/distance_field goal_pose_topic:=/scene_repr/target_goal`.

**Record:** bridge rates, clearances, a screenshot per step.

### C5 — Record the 20 plant scenes (P1.4.1)

For each of 20 arrangements (1–3 plants, leaves around the target, some
targets out of the arm's reach): move the camera over the plant with servo or
the MPC for ~20 s while recording:
`./scripts/record_bag.sh scene s01` (… `s20`). Keep the bags (`Codes/bags/`,
not in git) and a photo of each arrangement.

### C6 — Record the E1 dataset (P2A.5)

Episodes as in PIPER_JEPA_EXPERIMENTS.md §6 (identical neighbours, leaf
occlusion, camera motion through the robot): `./scripts/record_bag.sh e1 <name>`.
Export each with `jepa_episode export` and annotate (A10 steps 2–4).

### C7 — Record the E3 dataset (P3B.8)

Arm-only, base-only and combined motions with the target in view and behind
leaves: `./scripts/record_bag.sh e3 <name>`, driving the robot with the MPC or
servo. Export with `jepa_episode export ... --stride 6` (states and camera
poses come from `/odom`, `/joint_states` and TF), annotate the target (and
`--key plant_masks` if plant masks are wanted for P3). The stride sets the
predictor step: with the colour stream at 30 fps (`ros2 topic hz
/camera/color/image_raw`), stride 6 gives frames 0.2 s apart, two periods of
the 10 Hz MPC; the predictive MPC reads this interval from the checkpoint
(`jepa_stride` 2).

### C8 — C3 on the robot (P3B.11)

After D3 (a model trained on the robot's features): `predictive_mpc_node`
with `-p jepa_model:=<D3 P3.pt> -p encoder:=vjepa -p hub_entry:=<A6>`, ground
the flower with `jepa_ground`, dry run, then executed toward pre-grasp goals
next to an identical neighbour, against C2 (the plain MPC) on the same goals. **Record:** target in view (fraction of
steps), tracker on the right flower at the end, goal error, per method.

---

### C9 — MPPI visual servo vs IBVS on the robot (P2.3, first runs)

After TEST_PROCEDURE.md R-steps for the grasp (servo and approach validated
with the IBVS): the same stems, `servo_controller:=ibvs` and
`servo_controller:=mppi` (`mppi_backend`, `mppi_samples` from B7; with the
semantic stack of C4 running, also `scene_field_topic:=/scene_repr/distance_field`), servo only
first (`approach_enabled:=false`), then the stepwise approach. Hand on the
E-stop; `mppi_qd_max` 0.3 for the first runs. **Record:** per run, image error
over time (`/stem_grasp/servo_status`), approach time, final axis miss and
distance, aborts; contact or leaf displacement by eye, and the joint-effort
force (`/ft_sensor/raw`) once C2 step 4 has calibrated it (no F/T sensor). The
P2.3 exit numbers (≥ 20 % success, ≥ 30 % force) need the 50-trial protocol and
that calibrated force estimate.

---

## Part D — Analysis on the PC (recorded data)

### D1 — Semantic vs occupancy on the recorded scenes (P1.4.2)

```bash
for b in $(ls -d bags/scene_*/); do
  b=${b%/}; python3 -m scout_piper_jepa.episode export $b ${b}.npz --stride 3
done
python3 src/scout_piper_whole_body_mpc/benchmarks/semantic_vs_occupancy.py --episode bags/scene_*.npz \
  --jobs 4 --out $LOG/D1_sem_occ.jsonl | tail -8 | tee $LOG/D1_sem_occ.txt
```

Episodes without a target class need `--goal x y z` (run those one by one).
**Pass (P1.4.2):** ≥ 30 % fewer failures with the semantic representation.

### D2 — E1 comparison T0–T4 and the H1 decision (P2A.6)

```bash
python3 -m scout_piper_jepa.e1 run e1_episodes/*.npz --device cuda --out $LOG/D2_e1.json | tee $LOG/D2_e1.txt
```

**Record:** the table and the `h1` line (go / no-go). The rule: T4 at least
0.05 higher ID retention than the best of T0–T2, or ≥ 1 frame faster occlusion
recovery without lower retention.

### D3 — Train the predictors on robot features (P3B.9)

```bash
python3 -m scout_piper_jepa.train e3_episodes/*.npz --encoder vjepa --hub-entry <A6 entry> \
  --image-size 256 --device cuda --steps 3000 --proj-dim 64 --out-dir $LOG/D3_e3_vjepa | tee $LOG/D3_train.txt
```

`--proj-dim 64` keeps the predicted features small enough for the control
loop (A8); omit it, or try 128, if A8 shows raw features fit. P2 / P3 also take
the joint angles as input (`--no-state` for the ablation).

Features are cached next to each episode (first run is the slow one).
**Record:** the E3 table (persistence / P0 / P2 / P3), training time, and
`setup.step_s` from `results.json` (0.2 s with the C7 export stride) and
`setup.proj_energy` (the feature energy the projection keeps); keep
`P3.pt` for C8 and B4.

### D4 — Update PROGRESS.md

Fill in the measured values (A2, A4, B2–B5, C2, D1–D3) in PROGRESS.md and the
module READMEs, and mark the items done.

---

## Results log

Copy into `$LOG/MODULE_RESULTS.md`.

| Task | Date | Result | Numbers / notes |
|---|---|---|---|
| A1 torch GPU | | | version, GPU |
| A2 MPC timing PC | | | samples for `gpu` profile |
| A3 hardware-free (gpu) | | | MPPI median / max |
| A4 per-class nvblox + bridge | | | merge ms, GPU MB, free RAM |
| A5 MPC on the field | | | geometry_age_s, clearance |
| A6 encoders | | | entry points, grids, ms/frame |
| A7 E3 synthetic (GPU) | | | E_target@4 per method |
| A8 predictor latency PC | | | samples for C3 |
| A9 C2 vs C3 synthetic | | | visible fraction, goal error |
| A10 E1 pilot | | | table, annotation time |
| A11 plant benchmarks | | | W0–W3, semantic vs occupancy |
| A12 slip on the fake base | | | k_v, k_omega |
| A12 force estimate (fake arm) | | | estimate error, gains, 3-σ, abort s |
| A13 MPPI servo timing + grasp | | | median ms per backend, approach s |
| B2 MPC timing Orin | | | chosen profile, samples |
| B3 nvblox on the Orin | | | integration ms per class, RAM |
| B4 predictor latency Orin | | | samples |
| B5 encoder on the Orin | | | ms per frame |
| B6 C3 node hardware-free | | | cost active, solve ms |
| B7 MPPI servo on the Orin | | | median ms, chosen samples |
| C1 self-filter | | | boxes, filtered px |
| C2 calibration | | | tcp offset, hand-eye origin, k_v, k_omega, effort σ, 3-σ force, known-mass reading |
| C3 MPC with profile | | | time, error |
| C4 semantic scene on robot | | | bridge rates, clearances |
| C5 20 scenes recorded | | | bag names |
| C6 E1 recorded | | | episodes, frames |
| C7 E3 recorded | | | episodes, frames |
| C8 C3 vs C2 on robot | | | in view, tracker, goal error |
| C9 MPPI servo vs IBVS | | | error vs time, approach s, axis miss |
| D1 semantic vs occupancy (real) | | | failures per representation |
| D2 E1 + H1 | | | go / no-go |
| D3 E3 training | | | E_target@4 per method |
