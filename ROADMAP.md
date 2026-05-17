# Piper + Scout Research & Development Roadmap

**Platform:** AgileX Piper 6-DoF arm + RealSense camera mounted on AgileX Scout 2.0 UGV
**Compute:** NVIDIA Jetson Orin AGX 64 GB (on-robot) + operator laptop (GUI)
**Application domain:** Autonomous plant manipulation — peduncle/branch grasping for pollination and selective harvesting
**Document owner:** TBD
**Last updated:** 2026-05-16

---

## 0. End-state vision

A non-expert operator stands next to the Scout, opens a laptop GUI, and types or speaks something like:

> *"Go to the second row of plants on your left, find the flower with the longest peduncle, and grasp it just below the bud."*

The system:

1. **Parses the instruction** with a Vision-Language model running on the laptop (with Orin as fallback).
2. **Navigates the Scout** to the row using Nav2 + nvblox.
3. **Approaches the plant** using whole-body MPC that coordinates arm + base when the arm alone can't reach.
4. **Acquires geometry** with a learned active-perception loop (multi-view → feed-forward Gaussian).
5. **Selects the grasp** via a VLA-guided perception module that resolves "longest peduncle" against the scene.
6. **Executes the grasp** with an MPC visual servo that handles foliage occlusion and force feedback.
7. **Reports back** in plain language: *"Grasped the third flower. Force at contact: 0.4 N. Confidence 0.92. Want me to try another?"*

The roadmap below is the engineering path to that end state, structured so that **every phase produces a publishable contribution AND a deployable capability** — no dead-end research.

---

## 1. Architecture overview

```
                           ┌────────────────────────────────────┐
                           │   OPERATOR LAPTOP (GUI + LLM/VLM)  │
                           │   • Natural-language input         │
                           │   • Live scene view + grasp picks  │
                           │   • Confirmation / abort           │
                           └──────────────┬─────────────────────┘
                                          │ ROS 2 DDS (Wi-Fi)
                                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                       JETSON ORIN AGX 64GB                       │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐  │
│  │ Perception       │  │ Reasoning                             │  │
│  │ • YOLO/SAM seg   │──▶ VLA target picker (π0/OpenVLA-class) │  │
│  │ • VGGT 3D recon  │  │ VLM grounding for open-vocab labels  │  │
│  │ • nvblox sem.SDF │  │                                       │  │
│  └────────┬─────────┘  └──────────────┬───────────────────────┘  │
│           │                            │                          │
│           ▼                            ▼                          │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Whole-body MPC (cuRobo + MPPI extensions)                │    │
│  │ • 8-DoF (arm 6 + base 2)                                 │    │
│  │ • Semantic-class-aware cost                              │    │
│  │ • Visibility + manipulability + joint-limit constraints  │    │
│  └────────┬─────────────────────────────────────────────────┘    │
│           │                                                       │
│           ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Low-level ros2_control                                    │    │
│  │ • Piper joint trajectory controller (CAN via piper_sdk)  │    │
│  │ • Scout differential-drive velocity controller            │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Current baseline (what we're replacing)

| Component | Current | Pain point |
|---|---|---|
| Middleware | ROS 1 Noetic | EOL May 2025; blocks cuMotion, moveit_servo2, nvblox |
| Planner | MoveIt 1 + OMPL (8 attempts × 3 s budget) | Slow, non-deterministic, KDL IK fails near singularities |
| World model | Octomap from raw point cloud | No semantics; thin stems vanish in voxels |
| Servo | Custom IBVS (`core.py`), publishes TwistStamped to nowhere | No consumer; no horizon; no constraint handling |
| Mobile base | None | 6-DoF arm reach is the hard limit |
| Operator interface | RViz + terminal | Expert-only |

---

## 3. Phased plan

The phases are sequenced so that each one **unblocks the next** and **delivers an independent demo**.

### Phase 0 — ROS 2 Humble migration + Scout integration (Months 1–2)

**Goal:** Get the existing pipeline running on ROS 2 Humble with the Scout 2.0 as the mobile base, with no functional regression.

**Why first:** Every downstream contribution (cuMotion, nvblox, moveit_servo, recent VLA stacks) is ROS 2 only. AgileX already ships [piper_ros on a `humble` branch](https://github.com/agilexrobotics/piper_ros/tree/humble) with ros2_control + MoveIt 2, so the driver risk is low.

**Technical approach**
- Adopt `agilexrobotics/piper_ros@humble` as the arm driver baseline.
- Adopt [scout_nav2](https://github.com/AIRLab-POLIMI/scout_nav2) (ROS 2 Humble, Nav2-ready) as the base driver.
- Port [stem_grasp_ros1](../src/stem_grasp_ros1) → `stem_grasp` (ROS 2):
  - `rospy` → `rclpy`; node lifecycle managed by `Node`/`LifecycleNode`.
  - tf1 → tf2_ros (mostly mechanical).
  - service/action calls → `rclpy.action.ActionClient`.
- Build a unified URDF `piper_on_scout.urdf.xacro` with a 2-DoF planar virtual joint between `scout_base_link` and `piper_base_link`.
- Re-do hand-eye calibration on the integrated rig (existing [calibration_transform.py](../calibration_transform.py) logic ports cleanly; switch sample collection to ROS 2 actions).
- Stand up Nav2 on Scout with a basic 2D costmap so the base can be driven from a goal pose.

**Deliverables**
- New colcon workspace at `Piper_Scout_ws/` with `src/{piper, piper_description, piper_moveit, piper_msgs, scout_base, scout_nav2, stem_grasp, scout_piper_bringup}`.
- `scout_piper_bringup/launch/full_system.launch.py` brings up arm + base + camera + Nav2 + MoveIt 2.
- Regression test: existing scan/plan/servo pipeline produces the same grasp candidates as the ROS 1 version on a static plant.
- Updated CLAUDE.md / README for the new workspace.

**Success criteria**
- Full system boots and runs to "candidate grasp selected" on a real plant within 10 s.
- Scout drives 2 m to a goal pose with arm folded, then unfolds without self-collision.
- TF tree validated end-to-end (`scout_odom → scout_base_link → piper_base_link → camera_link`).

**Risks**
- Driver behavioral parity gaps between Noetic and Humble branches → mitigate with smoke tests on day 1 of each port.
- CAN bus contention between Piper SDK and Scout SDK → use distinct interfaces; verified by [piper_sdk](https://github.com/agilexrobotics/piper_sdk) usage of named CAN routes.

**Publications:** None expected from this phase — it's enabling work. (Possible "Open-source agricultural mobile-manipulation platform" tech report at end.)

---

### Phase 1 — Semantic 3D scene representation (Idea 4, Months 3–4)

**Goal:** Replace MoveIt's Octomap with a GPU-accelerated **per-class signed distance field** that distinguishes stem / branch / leaf / pot / target, fed by the existing YOLO+SAM segmentation.

**Why now:** Solves the open TODO in your codebase ("add leaf as collision object" from commit `e8cdfc7`), is a *prerequisite* for whole-body MPC in Phase 3 (the cost terms need semantic SDFs), and delivers an immediate demo improvement.

**Technical approach**
- Deploy [isaac_ros_nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox) on Orin AGX with the RealSense feed.
- Feed semantic masks (your existing segmentation node, ported to ROS 2) to nvblox's semantic channel.
- Maintain four parallel TSDFs: `stem`, `branch`, `leaf`, `target`.
- Expose a custom MoveIt 2 collision plugin that:
  - Treats `stem` + `branch` as **hard collision**.
  - Treats `leaf` as **soft cost** (allow brushing, log contact force budget).
  - Treats `target` as an **attractor** for goal generation.
- Add a configurable **inflation per class** (e.g., 5 mm padding around stems, 0 mm around leaves).

**Deliverables**
- `stem_grasp/scene_repr/` ROS 2 package wrapping nvblox + custom plugin.
- `stem_grasp/config/semantic_classes.yaml` declarative class → behavior map.
- Comparative benchmark: planning success rate on the same 20 hand-picked cluttered scenes, Octomap vs. semantic SDF.

**Success criteria**
- ≥ 30 % reduction in "no plan found" failures in cluttered scenes.
- Per-frame nvblox update < 33 ms on Orin AGX.
- Leaf-aware planning visibly avoids treating leaves as solid obstacles.

**Risks**
- nvblox's semantic channel API may need a custom fork to support 4 classes — budget 1 week for that.
- Segmentation latency could drag down nvblox update rate → run segmentation at half-rate, fuse temporally.

**Publications (target):** Workshop paper at ICRA Agri-Robotics or CASE — *"Semantic Signed Distance Fields for Thin-Structure Agricultural Manipulation."*

**Key references:** [isaac_ros_nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox), [Isaac ROS cuMotion + nvblox integration](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/index.html), [Self-Supervised Robotic Leaf Manipulation (2025)](https://arxiv.org/html/2505.03702v3).

---

### Phase 2 — MPPI visual-predictive servo (Idea 2, Months 5–7)

**Goal:** Replace the current adaptive IBVS in [core.py](../src/stem_grasp_ros1/src/stem_grasp_ros1/core.py) with a **sampling-based MPC over image-plane features** that handles visibility, manipulability, joint-limit, and force constraints jointly.

**Why now:** It's an isolated module (arm-only, single-frame inputs/outputs) — low integration risk. Validates MPPI on the platform before scaling to whole-body in Phase 3. Already plugs into the Phase 1 semantic SDF as a soft cost term.

**Technical approach**
- Re-implement [MPPI-VS](https://arxiv.org/abs/2104.04925) on Jetson Orin in CUDA + PyTorch.
- State: joint config q, EE pose, current image features (stem centroid, leaf bounding boxes).
- Inputs: 6-DoF joint velocity.
- Horizon: 20 steps × 50 ms = 1 s lookahead.
- Cost terms:
  - Image-plane error to desired stem centroid (existing).
  - Visibility (penalize leaving FoV).
  - Manipulability (Yoshikawa index, penalize singularities).
  - Joint-limit barrier.
  - Force prediction (use linearized arm dynamics + F/T history).
  - Soft-cost from Phase 1 leaf SDF.
- Sample 512 trajectories × 20 steps in parallel; converge in ~5 ms/step on Orin.
- Output `JointJog` to MoveIt 2 Servo (which now actually has a subscriber, unlike the ROS 1 stack).

**Deliverables**
- `stem_grasp/servo/mppi_vs/` package with CUDA kernels + Python wrapper.
- A/B test harness vs. current adaptive IBVS on 50 scripted approach trials.
- Public benchmark dataset (anonymized) of approach trajectories with force traces.

**Success criteria**
- ≥ 20 % improvement in approach success rate vs. baseline IBVS.
- ≥ 30 % reduction in average max-force at contact.
- Maintains 100 Hz outer loop on Orin AGX.

**Risks**
- MPPI tuning (sampling covariance, temperature) is finicky → reserve 2 weeks for tuning.
- Image-Jacobian linearization can be inaccurate at close range → fall back to your online-Jacobian estimator as a residual.

**Publications (target):** IROS or RA-L — *"Constraint-Aware MPPI Visual Servoing for Foliage-Rich Manipulation."*

**Key references:** [MPPI-VS](https://arxiv.org/abs/2104.04925), [Real-Time Constrained Visual Servoing for Agricultural Harvesting](https://www.mdpi.com/2673-2688/7/4/124), [Visual Predictive Control for Mobile Manipulator](https://www.sciencedirect.com/science/article/abs/pii/S0921889024001386).

---

### Phase 3 — Whole-body GPU MPC for Piper + Scout (Idea 1, Months 8–11) ★ headline contribution

**Goal:** Treat the 6-DoF arm + 2-DoF differential-drive base as a **unified 8-DoF system** under one MPC, so the robot can drive *while* the arm approaches and recover from local-minima that an arm-only planner can't (the "can't go around the obstacle" pain point).

**Why now:** Phases 0–2 have de-risked the building blocks (ROS 2, semantic SDFs, MPPI on Orin). This phase ties them together and is the strongest standalone publication.

**Technical approach**
- Extend [cuRobo](https://curobo.org/) / [cuMotion](https://github.com/nvidia-isaac/cumotion) to include the base degrees of freedom in its kinematic chain.
  - Approach 1: add a planar virtual joint at the chain root and let cuRobo's existing optimizer treat it as 2 additional DoF.
  - Approach 2: fork cuRobo's CUDA kernels to add a wheeled-base layer with non-holonomic constraint penalties — cleaner, more publishable.
- Custom cost terms (built on Phase 1 semantic SDFs):
  - Reach objective (Phase 2 image cost transplanted).
  - Manipulability of arm subset (avoid driving when the arm could solve it cleanly).
  - Base motion penalty (gentle preference for "arm first, base only if needed").
  - Visibility of target throughout the trajectory (so we don't drive past the plant).
  - Leaf soft-cost, stem/branch hard cost.
- 200 Hz whole-body replan target on Orin AGX.
- Safety filter: clip whole-body command through a Phase 2 short-horizon MPPI as a final layer.

**Comparison baselines**
- Arm-only MoveIt 2 + cuMotion (no base motion).
- Sequential planner: navigate base to fixed pose, then plan arm.
- Holistic QP (Haviland & Corke's NEO, reimplemented).
- Ours: whole-body MPPI.

**Deliverables**
- `stem_grasp/whole_body_mpc/` package.
- Quantitative benchmark on 30 cluttered plant scenes, measuring: time-to-grasp, success rate, % of "arm-unreachable" targets converted to "reachable via base motion."
- Open-source release of the wheeled-base cuRobo extension (the *upstream-able* contribution).

**Success criteria**
- ≥ 50 % of previously-unreachable targets become reachable via coordinated base motion.
- End-to-end time-to-grasp ≤ baseline.
- ≤ 0 % regression on previously-reachable targets.

**Risks**
- Differential-drive non-holonomy is awkward inside cuRobo's gradient framework — MPPI handles it natively, so lean MPPI if cuRobo fights us.
- Localization quality on Scout in outdoor / cluttered scenes — depend on nvblox + Nav2 fusion from Phase 0.

**Publications (target):** ICRA / IROS full paper — *"Whole-Body GPU-Parallel MPC for Mobile Manipulation in Thin-Structure Agricultural Settings."* Possibly T-RO extended version.

**Key references:** [Haviland & Corke holistic mobile manipulation](https://jhavl.github.io/holistic/), [EHC-MM](https://arxiv.org/html/2409.08527), [RMMI](https://arxiv.org/html/2408.16206), [cuRobo report](https://curobo.org/reports/curobo_report.pdf), [Industrial Motion Planning with GPUs](https://arxiv.org/html/2508.04146v2).

---

### Phase 4 — Active perception with feed-forward Gaussians (Idea 3, Months 11–13)

**Goal:** Replace the existing fixed "multi-view capture ring" with a **next-best-view loop** driven by feed-forward 3D reconstruction, cutting the number of captures needed by 3–5× and improving stem reconstruction quality near occlusions.

**Why now:** Phase 3 makes the platform mobile; mobile-rig views unlock NBV's value. By this point we have a stable whole-body controller, so positioning for an NBV viewpoint is "free."

**Technical approach**
- Integrate [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025 Best Paper) for sub-second posed 3D reconstruction from 3–8 views.
- Implement [FisherRF-style NBV](https://arm.stanford.edu/next-best-sense) but with a **task-aware information gain**:
  - Standard NBV maximizes generic surface coverage entropy.
  - Ours maximizes variance reduction *on the skeleton joint nearest the user-specified target* — the only geometry that matters for grasp selection.
- Closed loop: VGGT → skeletonize → score candidate viewpoints → command Phase 3 whole-body MPC to fly to the best one → re-VGGT.

**Deliverables**
- `stem_grasp/active_perception/` package.
- Ablation: random viewpoints vs. fixed ring vs. ours, measured on reconstruction quality at the grasp site + downstream grasp success.

**Success criteria**
- Reach equivalent reconstruction quality (Chamfer distance at stem) in 3 views vs. 9 fixed-ring views.
- Total perception time per plant ≤ 10 s.

**Risks**
- VGGT memory budget on Orin (need careful sparse-view batching).
- NBV optimization can get stuck in equally-good local minima → add small random jitter.

**Publications (target):** RA-L or T-RO short — *"Task-Aware Next-Best-View for Thin-Structure Reconstruction with Feed-Forward 3D Gaussians."*

**Key references:** [VGGT](https://github.com/facebookresearch/vggt), [Next Best Sense (Stanford ARM)](https://arm.stanford.edu/next-best-sense), [ActiveSplat (RA-L 2025)](https://li-yuetao.github.io/ActiveSplat/ActiveSplat.pdf).

---

### Phase 5 — VLA + operator GUI for non-expert use (Months 13–18) ★ application capstone

**Goal:** The end-state application. A laptop GUI lets a non-expert user issue natural-language instructions ("grasp the third flower from the left"), the system grounds them against the live scene, and the underlying Phase 1–4 stack executes safely.

**Why last:** VLAs are the *least* mature layer and benefit most from a deterministic, safety-bounded controller underneath. By Phase 5 we have that — the VLA can hallucinate freely; Phase 3's MPC won't let it crash the arm.

**Technical approach**

**5a — VLA integration on Orin (Months 13–14)**
- Pick the deployment model based on Orin AGX budget (~8 GB GPU headroom from prior phases):
  - **Default:** π₀ (3 B params, ~6 GB after INT8/TRT-LLM) — runs at ~5–8 Hz on Orin.
  - **Lighter fallback:** Octo-Small (30 M params), if π₀ pushes Orin too hard.
  - **Hybrid option:** Run the 7B-class VLA on the laptop, stream sub-goals to Orin over DDS at 2–5 Hz.
- Frame the VLA as a **sub-goal generator**, not a low-level controller:
  - Input: scene image + natural-language instruction.
  - Output: structured target (SE(3) pose + class label + textual rationale).
  - The whole-body MPC remains the safety-bounded executor.
- Add a [PhysVLM](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_PhysVLM_Enabling_Visual_Language_Models_to_Understand_Robotic_Physical_Reachability_CVPR_2025_paper.pdf)-style reachability check: gate every VLA-proposed target through a forward-kinematics-aware filter before the MPC sees it.

**5b — Open-vocabulary perception (Month 14)**
- Replace the fixed YOLO "branch/stem" prompt with a VLM-grounded segmentation (e.g., Grounding-DINO + SAM-2) so the system can handle new plant species ("the yellow flower," "the wilted leaf") without retraining.

**5c — Operator GUI on laptop (Months 15–16)**
- Tauri or Electron app (Tauri preferred for size + native feel).
- Tabs:
  - **Drive**: live camera, click-to-go-here on the costmap (drives Scout).
  - **Inspect**: live 3D Gaussian preview of the current plant from VGGT, with candidate grasp poses overlaid.
  - **Command**: text/voice prompt box; the VLA's parsed interpretation shown back to the user ("I think you mean: the flower at position X, do you confirm?").
  - **Watch**: live force/joint readouts, abort button, plain-language status feed.
- ROS 2 ↔ GUI communication: `rclpy` bridge or [Foxglove Studio](https://foxglove.dev) as the underlying transport (fast track for prototyping).
- Voice input via whisper-cpp on the laptop (no cloud).
- Voice output via Piper TTS (the *other* Piper, ironically) for status reports.

**5d — Confirmation loop + safety affordances (Month 17)**
- Every VLA decision surfaces in the GUI with a 2-second "are you sure?" countdown the operator can override.
- Plain-language explanations of why a target was rejected ("That flower is behind a leaf I can't safely brush through. Try another?").
- Hot-key e-stop wired to both the GUI and the existing hotkey watcher.

**5e — User study (Month 18)**
- 10 non-expert participants, 3 task families: identify a flower, grasp it, abort mid-grasp.
- Metrics: task success, time-to-grasp, NASA-TLX cognitive load, operator trust score.

**Deliverables**
- `stem_grasp/vla_bridge/` package (model serving + sub-goal grounding).
- `piper_scout_gui/` desktop app (Tauri).
- Public dataset of language-grounded plant manipulation episodes.
- User study report.

**Success criteria**
- A first-time user can issue and confirm a successful grasp instruction within 5 minutes of meeting the system.
- ≥ 80 % language-to-grasp success rate on a 100-trial test set.
- All VLA-proposed targets are reachability-filtered; 0 unsafe commands reach the arm.

**Risks**
- VLA latency on Orin — mitigation: run on laptop, sub-goals only.
- Operator trust collapse on a single failure — mitigation: confirmation loop + plain-language explanations.
- Open-vocabulary segmentation drift on new plant species — mitigation: keep YOLO fallback for known classes.

**Publications (target):**
- HRI or RO-MAN — *"Plain-Language Mobile Manipulation: Bridging VLA Reasoning and Safety-Bounded Control for Agricultural Robots."*
- Application paper at IROS Agri-Robotics workshop with the user study.

**Key references:** [Vision-Language-Action Models Survey](https://arxiv.org/html/2505.04769v1), [Large VLM-based VLA Models Survey](https://arxiv.org/html/2508.13073v1), [PhysVLM](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_PhysVLM_Enabling_Visual_Language_Models_to_Understand_Robotic_Physical_Reachability_CVPR_2025_paper.pdf), [Vision-Guided Robotic Pollination](https://arxiv.org/html/2510.06146).

---

## 4. Timeline at a glance

```
Month  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
P0    ████
P1          ████
P2                ██████
P3                          ████████████
P4                                      ██████
P5                                            ████████████
```

| Phase | Months | Lead milestone | Paper venue |
|---|---|---|---|
| 0 — ROS 2 + Scout | 1–2 | Full system on Humble | Tech report |
| 1 — Semantic SDF | 3–4 | ≥30 % plan-failure reduction | ICRA-W / CASE |
| 2 — MPPI-VS | 5–7 | ≥20 % servo-success gain | IROS / RA-L |
| 3 — Whole-body MPC | 8–11 | ≥50 % unreachable→reachable | ICRA / IROS / T-RO |
| 4 — Active perception | 11–13 | 3× fewer views | RA-L / T-RO |
| 5 — VLA + GUI | 13–18 | Novice-user study | HRI / RO-MAN |

**Total: 6 publishable contributions over 18 months on a single coherent platform.** Each phase is independently demonstrable — if the program ends early, every completed phase still ships a usable capability.

---

## 5. Compute budget on Jetson Orin AGX (64 GB)

Steady-state during Phase 5 operation:

| Component | GPU mem | GPU SM | Latency | Source |
|---|---|---|---|---|
| nvblox semantic SDF @ 30 Hz | ~2 GB | 1 SM | < 33 ms | Phase 1 |
| YOLO + SAM segmentation (TRT) | ~1 GB | 2 SMs | ~10 ms | existing |
| VGGT sparse 4-view recon (on-demand) | ~3 GB | 4 SMs | ~800 ms | Phase 4 |
| MPPI visual servo @ 100 Hz | ~0.5 GB | 1 SM | ~5 ms | Phase 2 |
| Whole-body MPC @ 200 Hz | ~1 GB | 2 SMs | ~5 ms | Phase 3 |
| π₀ VLA inference (sub-goal) | ~6 GB | 4 SMs (intermittent) | ~150 ms | Phase 5 |
| **Total peak** | **~13.5 GB** | **~10 SMs / 16** | — | — |

64 GB unified memory headroom is comfortable; SM contention is the real constraint and is staggered (VLA fires once per instruction; MPC runs continuously).

---

## 6. Cross-phase risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| AgileX `piper_ros@humble` branch becomes unmaintained | Medium | Fall back to [Reimagine-Robotics/piper_ros](https://github.com/Reimagine-Robotics/piper_ros) (MIT) or wrap the bare [piper_sdk](https://github.com/agilexrobotics/piper_sdk) directly. |
| Scout 2.0 outdoor localization drift | Medium | Fuse wheel odom + IMU + nvblox ICP + (optional) GPS in Nav2. |
| cuRobo can't be extended cleanly to wheeled base | Medium | MPPI-only fallback for Phase 3; loses some optimality but ships. |
| VLA hallucinations reach the arm | Low (by design) | PhysVLM-style reachability filter + safety MPC + operator confirm. |
| Schedule slip on any one phase | High | Each phase has a standalone "publishable scope" floor and a "full scope" ceiling — ship the floor, defer the ceiling. |
| ROS 2 Humble EOL (May 2027) during Phase 5 | Low-medium | Plan a Jazzy upgrade in Month 18 if needed; AgileX is likely to track. |

---

## 7. Open-source strategy

Each phase produces upstream-able artifacts. Suggested release schedule:

| Artifact | Phase | License | Notes |
|---|---|---|---|
| `piper_on_scout` URDF + bringup | 0 | Apache 2.0 | Lowers bar for other Piper+Scout users |
| Semantic SDF MoveIt collision plugin | 1 | Apache 2.0 | Useful beyond agriculture |
| MPPI-VS CUDA kernels | 2 | BSD | Reusable for any IBVS user |
| Wheeled-base cuRobo extension | 3 | Apache 2.0 | Pitch to NVIDIA Isaac team upstream |
| Task-aware NBV with VGGT | 4 | Apache 2.0 | — |
| VLA bridge + operator GUI | 5 | MIT (GUI), Apache 2.0 (bridge) | — |

---

## 8. References (consolidated)

### Motion planning / GPU
- [cuRobo project](https://curobo.org/) · [cuRobo technical report](https://curobo.org/reports/curobo_report.pdf)
- [cuMotion (Isaac ROS)](https://github.com/nvidia-isaac/cumotion) · [Isaac ROS cuMotion docs](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/index.html)
- [NVIDIA: CUDA-Accelerated Robot Motion Generation](https://developer.nvidia.com/blog/cuda-accelerated-robot-motion-generation-in-milliseconds-with-curobo/)
- [Industrial Robot Motion Planning with GPUs (extended cuRobo)](https://arxiv.org/html/2508.04146v2)
- [Black Coffee Robotics — cuRobo + ROS 2](https://www.blackcoffeerobotics.com/blog/curobo-nvidia-and-ros2-for-motion-planning)

### Scene representation
- [isaac_ros_nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox)
- [nvblox on Jetson AGX Orin](https://wiki.seeedstudio.com/deploy_nvblox_jetson_agx_orin/)

### Whole-body mobile manipulation
- [Haviland & Corke — Holistic Mobile Manipulation](https://jhavl.github.io/holistic/)
- [A Holistic Approach to Reactive Mobile Manipulation (arXiv 2109.04749)](https://www.arxiv-vanity.com/papers/2109.04749/)
- [NEO algorithm](https://www.researchgate.net/publication/348970144_NEO_A_Novel_Expeditious_Optimisation_Algorithm_for_Reactive_Motion_Control_of_Manipulators)
- [EHC-MM: Embodied Holistic Control](https://arxiv.org/html/2409.08527)
- [RMMI: Reactive Mobile Manipulation with Implicit Neural Map](https://arxiv.org/html/2408.16206)
- [Local Reactive Control for Mobile Manipulators (2025)](https://arxiv.org/html/2501.02815v1)
- [Reactive Whole-body Locomotion-integrated Manipulation (Springer 2025)](https://link.springer.com/article/10.1007/s11633-024-1538-9)

### Visual servoing / MPC
- [MPPI-VS (arXiv 2104.04925)](https://arxiv.org/abs/2104.04925)
- [MPC-Guided RL Visual Servoing for Tomato Harvesting (MDPI AI 2024)](https://www.mdpi.com/2673-2688/7/4/124)
- [Visual Predictive Control for Mobile Manipulator (RAS 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0921889024001386)
- [U-MPPI (T-RO 2025)](https://dl.acm.org/doi/10.1109/TRO.2025.3526078)

### Active perception
- [VGGT (CVPR 2025 Best Paper)](https://github.com/facebookresearch/vggt)
- [Next Best Sense (Stanford ARM, ICRA 2025)](https://arm.stanford.edu/next-best-sense)
- [ActiveSplat (RA-L 2025)](https://li-yuetao.github.io/ActiveSplat/ActiveSplat.pdf)

### Agricultural manipulation
- [Self-Supervised Robotic Leaf Manipulation (2025)](https://arxiv.org/html/2505.03702v3)
- [Autonomous Selective Harvesting Review — JFR 2024](https://onlinelibrary.wiley.com/doi/full/10.1002/rob.22230)
- [Vision-Guided Robotic Pollination (2025)](https://arxiv.org/html/2510.06146)
- [Peduncle Collision-Free Grasping with DRL (Compag 2023)](https://dl.acm.org/doi/10.1016/j.compag.2023.108488)
- [Key Technologies of Robotic Arms in Unmanned Greenhouse (MDPI 2025)](https://www.mdpi.com/2073-4395/15/11/2498)

### VLA / language-conditioned robotics
- [Vision-Language-Action Models Survey (arXiv 2505.04769)](https://arxiv.org/html/2505.04769v1)
- [Large VLM-based VLA Models Survey (arXiv 2508.13073)](https://arxiv.org/html/2508.13073v1)
- [PhysVLM (CVPR 2025) — physical reachability for VLMs](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_PhysVLM_Enabling_Visual_Language_Models_to_Understand_Robotic_Physical_Reachability_CVPR_2025_paper.pdf)
- [Foundation Models in Robotics — Comprehensive Review (2025)](https://arxiv.org/html/2507.10087v1)

### Hardware / platforms
- [AgileX Piper](https://global.agilex.ai/products/piper) · [piper_sdk (Python)](https://github.com/agilexrobotics/piper_sdk) · [piper_ros @ humble](https://github.com/agilexrobotics/piper_ros/tree/humble) · [Reimagine-Robotics/piper_ros (alt ROS 2 driver)](https://github.com/Reimagine-Robotics/piper_ros)
- [AgileX Scout 2.0](https://global.agilex.ai/products/scout-2-0) · [scout_ros (ROS 1)](https://github.com/agilexrobotics/scout_ros) · [scout_nav2 (ROS 2)](https://github.com/AIRLab-POLIMI/scout_nav2)
- [MoveIt 2 Realtime Servo Tutorial](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html) · [Manipulation in ROS 2 — 2025 discourse thread](https://discourse.openrobotics.org/t/manipulation-in-ros2-2025-what-s-everyone-using-these-days/43683)
