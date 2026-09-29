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
first, then stem with the leaf's tip as the attachment target. `export()` gives
`leaf_vertices`, `leaf_faces`, `stem_curve`, `stem_radius`.

## ROS 2 node

```
ros2 launch plant_twin plant_twin.launch.py
```

| Direction | Topic | Type | Role |
|---|---|---|---|
| in | `/stem_grasp/leaf_cloud` | PointCloud2 | leaf points, planning frame |
| in | `/stem_grasp/stem_cloud` | PointCloud2 | stem points |
| in | `/ft_sensor/raw` | WrenchStamped | \|F\| > `contact_threshold_n` ⇒ touching |
| in | `/joint_states` | JointState | `piper_joint7` < `gripper_closed_m` ⇒ grasped |
| in (TF) | `piper_base_link → piper_link7` | | fingertip = contact point |
| out | `/plant_twin/markers` | MarkerArray | TRIANGLE_LIST leaf, LINE_STRIP stem (RViz) |
| out | `/plant_twin/leaf_tip` | PointStamped | stem/leaf attachment |

`pulling` = touching **and** gripper closed; before that the stem's
stationary prior holds it at rest.

On the first frame with both clouds, the node initialises the leaf plane by
PCA of the leaf points (disc outline of `leaf_outline_radius_m`) and the stem
as a straight line from the lowest stem point to the leaf tip.

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

- **Outline from the mask.** The node starts from a disc; replacing it with the
  segmentation mask's contour (+ hole contours from `cv2.findContours`
  hierarchy) projected into the PCA plane is the next step.
- **Texture.** `export()` returns geometry only; UV = `LeafModel.uv`, so
  texturing is a matter of sampling the RGB image at the projected rest
  vertices on the first frame.
- **Per-class clouds.** `/stem_grasp/leaf_cloud` and `/stem_grasp/stem_cloud`
  are expected from the Phase 1 `class_demux_node`; `pointcloud_node` currently
  publishes a single masked cloud.
- **Speed.** Numerical Jacobians; ~50–100 ms per frame at `mesh_res=20` on a
  laptop CPU. Analytic Jacobians or a GPU port are needed for 30 Hz.
