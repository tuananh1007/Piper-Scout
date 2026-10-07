# Snapshot of the CPU semantic distance field (P1.7.7) for planners that do not
# embed the Python query, i.e. the MoveIt semantic collision plugin. Published by
# scene_query_node from scout_piper_scene_repr_py.field.export_field().
#
# Grid: voxel (i, j, k) spans origin + [i, i+1) * voxel_size (likewise j, k);
# values live at voxel centres. Arrays hold size[0]*size[1]*size[2] entries in
# C order over (i, j, k), k fastest. Distances are signed (> 0 outside), metres.

std_msgs/Header header          # frame_id: grid frame; stamp: last integration (ages refer to it)
geometry_msgs/Point origin      # grid-frame coordinates of the (0, 0, 0) voxel corner
float32 voxel_size
uint32[3] size

# Hard classes: min over classes of (signed distance - padding).
# +inf everywhere when no hard voxel is occupied.
string[] hard_classes
float32[] hard_distance
uint8[] hard_class              # index into hard_classes of the class nearest each voxel

# Soft class (leaf): signed distance; penetration deeper than soft_max_penetration
# is a hard violation. soft_distance is empty when the policy has no soft class.
string soft_class
float32 soft_max_penetration
float32[] soft_distance

# Observation age at header.stamp in 0.1 s units; 65535 = never observed (unknown).
uint16[] age_ds
