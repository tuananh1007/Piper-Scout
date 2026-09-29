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

Alternates two `scipy.optimize.least_squares` (TRF) solves per frame: leaf
first, then stem with the leaf's tip as the attachment target. Both use
**analytic Jacobians** (`LeafModel.jacobian`, `StemModel.jacobian`): ICP
correspondences and normals are held fixed per evaluation (Gauss-Newton
ICP); the leaf's per-vertex derivative uses the SO(3) right Jacobian for the
rotation vector, and the stem is linear in its control points
(`curve = M @ ctrl`). Each frame runs ~10 evaluations warm-started from a
constant-velocity prediction; on a laptop CPU that is ~60 ms/frame for a
600-point leaf (was ~130 ms with numerical Jacobians and 40 evaluations), with
~4 mm steady error while tracking 5 mm/frame motion. `export()` gives
`leaf_vertices`, `leaf_faces`, `stem_curve`, `stem_radius`.

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
- **Speed.** ~60 ms/frame on CPU (≈16 Hz). The remaining cost is split
  between residual/Jacobian evaluation (~4 ms each) and TRF's SVD per
  iteration; a custom Gauss-Newton with normal equations, or a GPU port, is
  the route to 30 Hz.
- **Untested on hardware** — no ROS in the dev container; only the numpy core
  is covered by tests.
