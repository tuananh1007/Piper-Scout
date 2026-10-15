# scout_piper_language — instructions and operator GUI (Phase 5)

An operator who is not a roboticist says or types what to do ("grasp the red
flower on the left"), sees what the robot understood on the camera image,
and has two seconds to cancel before anything moves. The language layer only
chooses *which* detected object is meant; the existing stack (stem_grasp,
whole-body MPC, servo, hard gates) does the motion. ROADMAP.md Phase 5: map ℓ
to q_ℓ = (class, ordinal, spatial relation, attribute, action), "without
making the VLM the motor controller".

## Pieces

| Module | What it does |
|---|---|
| `parser.py` | Rule-based q_ℓ: action (grasp, inspect, release, stop, scan; cut / prune / pull refused), class (flower, stem, branch, leaf, fruit, bud), ordinal with counting direction ("second from the right"), superlative (leftmost, tallest, nearest, biggest, middle), attributes (colour, size, ripe), relation to a reference phrase (left / right of, above, below, behind, in front of, next to); unknown words are kept, never guessed |
| `grounding.py` | Picks the detection: class aliases (YOLO names such as "potted plant" → stem), colour and size filters, image left/right, height and distance from the robot when the 3-D point is known; `ambiguous` → a question listing the candidates by position, `resolve_followup` reads the answer ("the left one", "the second") |
| `reachability.py` | Every proposed target, before anything moves: `no_depth`, `too_high` / `too_low` (heights a near-horizontal approach reaches from the top plate), `too_thick` (stem width from the outline vs the 7 cm gripper), `crowded` (another hard obstacle within half the open gripper + 1 cm, target stem released with `exclude_target`), `too_far` / `no_pose` (no collision-free whole-body pose on the pre-grasp point within the drive range); advisory `occluded` ("It is behind a leaf; I will look from another side first") hands over to active perception. Plain-language messages |
| `vlm.py` | `RuleProposer` (default) and `HttpProposer`: a VLM behind HTTP names one detection; the answer is validated (index exists, class matches) or the rules answer, with the reason |
| `session.py` | Clarify → propose → 2-s countdown → commit, `confirm` / `cancel`; "stop" is immediate and never counted down |
| `safety_stop.py` | stem_grasp's hot-key stop extended: zero twist and joint jog to moveit_servo, zero `/cmd_vel`, MPC cancel, servo bridge disabled (confirmation tracked) |
| `instruction_node.py` | ROS node around the session (below) |
| `gui_server.py` | Web server, ROS-free: page, JSON API, server-sent events, MJPEG; `FakeBackend` for trying the page and for the browser test |
| `operator_gui_node.py` | The GUI's robot side: topics in, instruction / confirm / cancel out, STOP here as well (works without the instruction node), hold-to-move drive, optional whisper-cpp / Piper TTS |
| `web/` | The page: tabs Command (camera with numbered detections, instruction box, microphone, proposal card with countdown, questions with option buttons, message log with optional speech), Inspect (detection list with Grasp / Inspect buttons), Drive (enable, hold-to-move base and gripper pads, speed), Watch (pipeline, MPC, uncertainty meters, force, plant twin); STOP always on screen, Esc or Space |

## Run

```bash
ros2 launch scout_piper_language operator.launch.py            # http://127.0.0.1:8080/
ros2 launch scout_piper_language operator.launch.py host:=0.0.0.0   # tablet on the robot's closed network
# the pipeline grasps the stem the operator chose:
ros2 run stem_grasp pipeline_node --ros-args ... -p operator_target_topic:=/operator/target
# the page without a robot:
ros2 run scout_piper_language operator_gui_demo --port 8080
```

| Node | In | Out |
|---|---|---|
| `instruction_node` | `/operator/instruction`, `/operator/confirm`, `/operator/cancel`, `/operator/abort`, `/stem_grasp/detections`, `/scene_repr/distance_field` (optional), camera_info, `/odom`, `/joint_states`, TF | `/operator/say`, `/operator/proposal` (events), `/operator/state`, `/operator/target` (PointStamped, odom), `/piper_jepa/init_mask`; calls pipeline `~/scan`, `~/release`, `/active_perception/plan_view` |
| `operator_gui_node` | `/operator/say`, `/operator/proposal`, `/operator/state`, `/stem_grasp/pipeline_state`, `/whole_body_mpc/status`, `/active_perception/status`, `/plant_twin/state`, `/ft_sensor/raw`, `/stem_grasp/detections`, camera image | `/operator/instruction`, `/operator/confirm`, `/operator/cancel`, `/operator/abort`, STOP (`safety_stop.py`), `/cmd_vel` and `/servo_node/delta_twist_cmds` (manual drive) |

Manual drive: enabled only while the pipeline is not moving on its own; each
command holds 0.3 s (`hold_s`) and then zeros, so a dropped connection or a
released button stops; speeds capped (0.2 m/s, 0.4 rad/s, 5 cm/s at the
gripper). A STOP disables the drive.

Dead-man switch: the page posts a heartbeat four times a second; the node
publishes `/operator/heartbeat` while the page's are fresh (0.75 s). The
safety supervisor (scout_piper_bringup `safety_supervisor.py`, launched with
`operator_heartbeat:=true`) stops all motion when it stops. The page shows
`/safety/state` as a red banner with **Reset safety stop** (`/api/safety_reset`
→ `/safety_supervisor/reset`); drive is refused while tripped. Browsers slow
the timers of background tabs, so the page must stay in front while the
robot moves under the heartbeat.

Voice: `whisper_cmd` (e.g. `whisper-cli -m /models/ggml-base.en.bin -nt -f {wav}`)
and `tts_cmd` (e.g. `piper --model /models/en_US-lessac-medium.onnx --output_file {wav}`)
in `config/operator.yaml`; without them the page is typed and speaks with the
browser's speech synthesis.

The server binds to 127.0.0.1 by default and has no authentication: open it to
the network only on the robot's closed network.

## Tests

```bash
cd Codes/src/scout_piper_language && python3 -m pytest test -q
```

Parser cases, grounding (unique, ambiguous with follow-up, none, relations,
superlatives, aliases), the session (countdown, clarification, rejection by
reachability, stop during a countdown, refusals), the VLM answer validation
and fallback, reachability reasons (including a crowded target in a
synthetic field), the server's API and events, and the page in headless
Chromium (Playwright, `test/gui_e2e.mjs`): proposal and countdown, cancel,
commit, question and answer, hold-to-move with a stop on release, STOP button
and Esc, the drive disabled after a stop, heartbeats, the safety banner with
the drive refused and the reset, the Watch tab, phone width without
horizontal scrolling, no page errors. On the fake robot:
`scout_piper_bringup/scripts/operator_chain_check.py` (`hardware_free_checks.sh`,
check `operator`).

## Open

- **VLA / VLM on the Orin** (P5.1.1, P5.1.2): model choice and quantisation;
  `HttpProposer` is the interface.
- **Open-vocabulary perception** (P5.2): Grounding-DINO + SAM-2 in place of the
  fixed YOLO classes; the segmentation node's Grounded-SAM path exists but
  does not publish per-instance detections yet.
- **Tauri shell** (P5.3.1): the page runs in any browser; a Tauri wrapper is
  packaging only.
- **User study** (P5.4): ethics approval, participants, NASA-TLX.
