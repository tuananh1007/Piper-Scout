# plant_twin — deformable leaf + stem digital twin

Per-frame reconstruction of one leaf and its stem while the Piper gripper
touches and pulls it, in the spirit of the "Original + tracks / Reconstruction /
3D pan" demo: the leaf is a textured, curved triangle mesh with an explicit
outline and holes; the stem is a 3D centreline with circular thickness.

The maths lives in pure numpy/scipy (`leaf.py`, `stem.py`, `fitting.py`) and is
tested without ROS. `twin_node.py` is the ROS 2 wrapper for the Piper arm.

## Models

### Leaf (`LeafModel`)

| Part | Representation |
|---|---|
| Rest shape | 2D outline polygon + hole polygons in the leaf plane, triangulated once on a regular lattice (triangles outside the outline or touching a hole are dropped). |
| Rigid motion | rotation vector (3) + translation (3). |
| Bending | `grid`² scalar heights on a lattice → smooth height field via Gaussian RBF, applied along the leaf normal **before** the rigid transform, so rigid and bending parameters are separated. |

Residual blocks (`LeafModel.residuals`, weights in `LeafWeights`):

- **data** — point-to-plane + light point-to-point ICP, both directions
  (obs→mesh and mesh→obs). Point-to-plane is what lets the pose settle: a
  pure nearest-vertex term stalls half a lattice cell from the truth.
- **stretch** — mesh edge lengths vs rest lengths (leaves bend, don't stretch).
- **prior** — bending heights pulled toward zero (rest shape).
- **temporal** — parameters pulled toward previous frame.
- **contact** — the gripper fingertip must lie on the surface while touching.

### Stem (`StemModel`)

K control points of a Catmull-Rom centreline plus a fixed radius.

- **base** — first control point anchored to the initial base position.
- **tip** — last control point attached to the leaf's petiole point
  (`LeafModel.tip_point`).
- **length** — total curve length ≈ rest length.
- **smooth** — second differences of the control points.
- **temporal** — vs previous frame.
- **stationary** — vs rest shape, active only while `pulling=False`.

### Fitter (`PlantTwinFitter`)

Alternates a leaf solve and a stem solve per frame (stem uses the leaf's
tip as its attachment target). Both models expose `evaluate(params, ...)`
returning the stacked residual **and** its analytic Jacobian in one pass
(shared KD-tree queries and normals; ICP correspondences and normals are held
fixed per evaluation — Gauss-Newton ICP). The leaf's per-vertex derivative
uses the SO(3) right Jacobian for the rotation vector; the stem is linear in
its control points (`curve = M @ ctrl`).

The solver (`solver.py`) is a small Marquardt-damped Gauss-Newton on the
normal equations: the problems are thousands of residuals by ~20 unknowns, so
one fused evaluation plus a P×P Cholesky solve per step is far cheaper than
scipy TRF's per-step SVD. `PlantTwinFitter(solver=...)` picks `'lm'`
(default), `'trf'` (scipy + analytic Jacobian) or `'fd'` (scipy +
finite differences, for debugging). Each frame is warm-started from a
constant-velocity prediction and stops at a 1e-3 relative cost drop (a
tracking tolerance; most frames take 3–5 steps).

Measured on a 4-core laptop-class CPU, 600 leaf points, `mesh_res=20`,
tracking 5 mm/frame in-plane motion:

| fitter | ms/frame | steady error |
|---|---|---|
| numerical Jacobian, TRF, 40 evals (v1) | ~130 | 22 mm (lagged) |
| analytic Jacobian, TRF, 10 evals | ~60 | 3.7 mm |
| fused evaluate + LM, `ftol=1e-3` | **~28** | 1.6 mm |

Expect roughly 2–3× slower on the Orin's CPU cores; `max_iter` and `ftol`
are the knobs.

## ROS 2 node

```
ros2 launch plant_twin plant_twin.launch.py
```

| Direction | Topic | Type | Role |
|---|---|---|---|
| in | `/stem_grasp/leaf_filtered_cloud` | PointCloud2 (xyz+rgb) | leaf points, from `stem_grasp/pointcloud_node`; TF'd into the planning frame |
| in | `/stem_grasp/filtered_cloud` | PointCloud2 | stem points |
| in | `/stem_grasp/target_mask` | Image mono8 | leaf mask → outline + holes |
| in | `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info` | Image, CameraInfo | back-project the mask contour (depth must be aligned to colour) |
| in | `/ft_sensor/raw` | WrenchStamped | \|F\| > `contact_threshold_n` ⇒ touching; no F/T sensor on the platform: the joint-effort estimate (INSTALL.md 10.14), so the threshold must exceed its 3-σ noise |
| in | `/joint_states` | JointState | `piper_joint7` < `gripper_closed_m` ⇒ grasped |
| in (TF) | `piper_base_link → piper_link7` | | fingertip = contact point |
| out | `/plant_twin/markers` | MarkerArray | TRIANGLE_LIST leaf, LINE_STRIP stem (RViz) |
| out | `/plant_twin/leaf_tip` | PointStamped | stem/leaf attachment |
| in | `/camera/color/image_raw` | Image | colour for the UV texture |
| out | `/plant_twin/state` | String (JSON) | fit quality of the frame: `confidence` (0–1), `valid`, `leaf_rms_mm` / `stem_rms_mm`, inlier fractions, point counts, `touching`, `pulling`, `force_n`, `cloud_age_s`, `stamp`, `frame`, `silhouette_iou`, `reinit_count`, `texture_coverage` |
| srv | `~/export` | Trigger | OBJ + MTL + PNG of the current leaf into `export_dir` |

`pulling` = touching **and** gripper closed; before that the stem's
stationary prior holds it at rest.

On the first frame with clouds, mask, depth and intrinsics, the node
(`outline.py`):

1. fits a PCA plane to the leaf points, normal facing the camera;
2. takes the mask's largest outer contour and its hole contours
   (`cv2.RETR_CCOMP`), simplifies them, back-projects with the nearest valid
   depth and projects into the plane → `LeafModel(outline, holes)`;
3. sets the petiole to the outline vertex nearest the lowest stem point and
   textures each mesh vertex with the nearest cloud point's colour
   (`LeafModel.texture_from_cloud`), published as per-vertex marker colours;
4. initialises the stem as a straight line from that base to the petiole.

### Fit quality

`fit_quality` scores each frame from the data alone: the leaf points'
point-to-plane distance to the fitted mesh (normal of the nearest vertex) and
the stem points' distance to the centreline minus the radius. Per part it
reports the RMS and the fraction within `inlier_m` (5 mm); the confidence is
`min(inlier fractions) · exp(−max(RMS) / 5 mm)`, valid at ≥ 0.5 with at least
20 points per part. Phase 4's active perception reads `1 − confidence` as the
deformation uncertainty U_twin; a large `cloud_age_s` means the twin is
coasting on old clouds.

### Re-initialisation, texture, export (`texture.py`)

Every frame the fitted mesh is rasterised into the camera and compared with
the leaf mask (`silhouette_iou`). Five frames in a row below 0.6
(`reinit_min_iou`, `reinit_frames`) without contact re-run the first-frame
set-up from the current mask: the leaf turned, folded, grew in the mask or
merged with a second leaf, and the rest shape no longer explains it. Frames
with contact neither count nor reset the run, since the fingers occlude the
leaf then.

`UVTexture` bakes the colour image into a texture over the rest plane
(`texture_size`², every `texture_every_n` frames): each texel is lifted by
the bending field, moved by the frame's pose, projected into the camera and
coloured where it is visible (inside the image, facing the camera, on the
observed depth within 1 cm), blended over frames with weight cos²(view
angle). Holes and the outside stay transparent. `export_obj` (service
`~/export`) writes the mesh with `LeafModel.uv` as texture coordinates, the
MTL and the PNG, for a render outside RViz. RViz still shows the per-vertex
colours.

### Torch evaluation (`leaf_torch.py`)

`TorchLeafEvaluator` is `LeafModel.evaluate` in torch: the per-vertex
Jacobian and its contractions as tensor operations, nearest neighbours by
`torch.cdist`, so a GPU does the whole evaluation and the LM solver only gets
the arrays back. In float64 on the CPU it matches numpy to rounding
(`test_leaf_torch.py`). `leaf_backend: torch`, `device: cuda` in the config
(float32 on a GPU). Timed on the GPU and the Orin: not yet (MODULE_TASKS.md).

## Tests

```bash
cd Codes/src/plant_twin
python -m pytest test -q
```

Synthetic checks: rigid pose recovery from a noisy surface sample, bending
toward a contact point without stretching, stem base/length/tip constraints,
stem stationarity before a pull, and a six-frame pull sequence with smooth
motion and stem–leaf attachment.

## What is still stubbed

- **Re-initialisation starts over.** A re-initialised leaf gets a new rest
  shape and stem, and its texture starts empty; nothing is carried across
  (a merge of the old texture into the new UV square is possible, as both
  are in the leaf plane).
- **Speed on the Jetson.** ~28 ms/frame here is ≈35 Hz; on the Orin CPU
  it will be nearer 10–15 Hz. Each evaluation is ~3.7 ms of which the two
  KD-tree queries are ~1 ms and the (V,3,P) Jacobian assembly the rest. The
  torch port runs it on the GPU; its speed there is unmeasured.
- **Untested on hardware** — only the numpy core is covered by tests (no ROS needed).
