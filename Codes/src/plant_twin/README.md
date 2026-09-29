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
| in | `/camera/depth/image_rect_raw`, `/camera/color/camera_info` | Image, CameraInfo | back-project the mask contour |
| in | `/ft_sensor/raw` | WrenchStamped | \|F\| > `contact_threshold_n` ⇒ touching |
| in | `/joint_states` | JointState | `piper_joint7` < `gripper_closed_m` ⇒ grasped |
| in (TF) | `piper_base_link → piper_link7` | | fingertip = contact point |
| out | `/plant_twin/markers` | MarkerArray | TRIANGLE_LIST leaf, LINE_STRIP stem (RViz) |
| out | `/plant_twin/leaf_tip` | PointStamped | stem/leaf attachment |

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

## Tests

```bash
cd Codes
python -m pytest src/plant_twin/test -q
```

Synthetic checks: rigid pose recovery from a noisy surface sample, bending
toward a contact point without stretching, stem base/length/tip constraints,
stem stationarity before a pull, and a six-frame pull sequence with smooth
motion and stem–leaf attachment.

## What is still stubbed

- **Outline is fixed after the first frame.** Re-initialisation when the mask
  changes substantially (occlusion by the gripper, a second leaf) is not
  handled; the rest shape stays whatever the first frame gave.
- **Texture is per-vertex**, not a UV image; fine for RViz, coarse for a
  render. `LeafModel.uv` is there for a proper texture map.
- **Speed on the Jetson.** ~28 ms/frame here is ≈35 Hz; on the Orin CPU
  it will be nearer 10–15 Hz. Each evaluation is ~3.7 ms of which the two
  KD-tree queries are ~1 ms and the (V,3,P) Jacobian assembly the rest — a
  CuPy/torch port of `evaluate` is the next lever if that is not enough.
- **Untested on hardware** — no ROS in the dev container; only the numpy core
  is covered by tests.
