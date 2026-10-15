"""Add static collision objects (config/scene_objects.yaml) to MoveIt's planning scene.

Runs once move_group is up: adds the boxes and the allowed-collision entries
in a single planning-scene diff, reads the scene back to confirm, then stays up
to latch what it applied on /scene_objects/info (std_msgs/String, JSON) so
recorded episodes carry it. Exits non-zero (and quickstart.launch.py shuts
everything down) if move_group never appears or the objects don't stick: you
should never plan without the obstacles you think are there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import rclpy
import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix, CollisionObject, PlanningScene
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String


# --------------------------------------------------------------------- config
def load_config(path: Path) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    frame = cfg.get("frame_id")
    objects = cfg.get("objects") or []
    if not frame:
        raise ValueError(f"{path}: frame_id is required")
    seen = set()
    for i, o in enumerate(objects):
        where = f"{path}: objects[{i}]"
        if o.get("type", "box") != "box":
            raise ValueError(f"{where}: only type 'box' is supported, got {o.get('type')!r}")
        if not o.get("id") or o["id"] in seen:
            raise ValueError(f"{where}: needs a unique id")
        seen.add(o["id"])
        size, center = o.get("size"), o.get("center")
        if not (isinstance(size, list) and len(size) == 3 and all(float(v) > 0 for v in size)):
            raise ValueError(f"{where} ({o['id']}): size must be 3 positive numbers")
        if not (isinstance(center, list) and len(center) == 3):
            raise ValueError(f"{where} ({o['id']}): center must be 3 numbers")
    cfg.setdefault("allow_collision_with", [])
    return cfg


# ------------------------------------------------------------------------ ACM
def acm_to_table(acm: AllowedCollisionMatrix) -> Tuple[List[str], List[List[bool]]]:
    names = list(acm.entry_names)
    return names, [list(e.enabled) for e in acm.entry_values]


def allow_pairs(names: List[str], table: List[List[bool]],
                pairs: Iterable[Tuple[str, str]]) -> Tuple[List[str], List[List[bool]]]:
    """Return a copy of an ACM table with each (a, b) pair allowed, both ways.

    Names not yet in the matrix are appended (their other entries default to
    not allowed, i.e. collision-checked).
    """
    names = list(names)
    table = [list(row) for row in table]
    for a, b in pairs:
        for n in (a, b):
            if n not in names:
                names.append(n)
                for row in table:
                    row.append(False)
                table.append([False] * len(names))
        i, j = names.index(a), names.index(b)
        table[i][j] = table[j][i] = True
    return names, table


def table_to_acm(names: List[str], table: List[List[bool]], template: AllowedCollisionMatrix) -> AllowedCollisionMatrix:
    acm = AllowedCollisionMatrix()
    acm.entry_names = names
    acm.entry_values = [AllowedCollisionEntry(enabled=row) for row in table]
    acm.default_entry_names = list(template.default_entry_names)
    acm.default_entry_values = list(template.default_entry_values)
    return acm


def box_object(frame_id: str, obj: Dict) -> CollisionObject:
    co = CollisionObject()
    co.header.frame_id = frame_id
    co.id = obj["id"]
    co.operation = CollisionObject.ADD
    prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in obj["size"]])
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in obj["center"])
    pose.orientation.w = 1.0
    co.primitives = [prim]
    co.primitive_poses = [pose]
    co.pose.orientation.w = 1.0
    return co


# ----------------------------------------------------------------------- node
class SceneObjects(Node):
    def __init__(self) -> None:
        super().__init__("scene_objects")
        self.declare_parameter("config_file", "")
        self.declare_parameter("move_group_timeout_s", 60.0)
        self._path = Path(self.get_parameter("config_file").value)
        self._timeout = float(self.get_parameter("move_group_timeout_s").value)
        self._apply = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._get = self.create_client(GetPlanningScene, "/get_planning_scene")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._info_pub = self.create_publisher(String, "/scene_objects/info", latched)

    def _call(self, client, req):
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.done() or fut.result() is None:
            raise RuntimeError(f"{client.srv_name} did not answer within 10 s")
        return fut.result()

    def _scene(self, components: int) -> PlanningScene:
        req = GetPlanningScene.Request()
        req.components.components = components
        return self._call(self._get, req).scene

    def run(self) -> int:
        log = self.get_logger()
        try:
            cfg = load_config(self._path)
        except (OSError, ValueError) as e:
            log.fatal(f"cannot load scene objects: {e}")
            return 1
        for client in (self._apply, self._get):
            if not client.wait_for_service(timeout_sec=self._timeout):
                log.fatal(f"{client.srv_name} not available after {self._timeout:.0f}s: is move_group running?")
                return 1

        frame, objs = cfg["frame_id"], cfg["objects"]
        try:
            current = self._scene(PlanningSceneComponents.ALLOWED_COLLISION_MATRIX).allowed_collision_matrix
            pairs = [(link, o["id"]) for link in cfg["allow_collision_with"] for o in objs]
            names, table = allow_pairs(*acm_to_table(current), pairs)

            diff = PlanningScene()
            diff.is_diff = True
            diff.world.collision_objects = [box_object(frame, o) for o in objs]
            # A diff's ACM replaces the scene's whole ACM, so send the full,
            # extended matrix (keeps the SRDF's disabled self-collision pairs).
            diff.allowed_collision_matrix = table_to_acm(names, table, current)
            if not self._call(self._apply, ApplyPlanningScene.Request(scene=diff)).success:
                log.fatal("move_group rejected the planning-scene diff (unknown frame_id?)")
                return 1

            world = self._scene(PlanningSceneComponents.WORLD_OBJECT_NAMES).world
            present = {c.id for c in world.collision_objects}
        except RuntimeError as e:
            log.fatal(str(e))
            return 1
        missing = [o["id"] for o in objs if o["id"] not in present]
        if missing:
            log.fatal(f"objects not in the planning scene after applying: {missing}")
            return 1

        for o in objs:
            log.info(f"added {o['id']}: box {o['size']} at {o['center']} in {frame}")
        if cfg["allow_collision_with"]:
            log.info(f"collisions allowed between {cfg['allow_collision_with']} and these objects")
        self._info_pub.publish(String(data=json.dumps({"config_file": str(self._path), **cfg})))
        return 0


def main(args=None) -> int:
    rclpy.init(args=args)
    node = SceneObjects()
    code = node.run()
    if code == 0:
        try:
            rclpy.spin(node)  # keep /scene_objects/info latched
        except KeyboardInterrupt:
            pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
