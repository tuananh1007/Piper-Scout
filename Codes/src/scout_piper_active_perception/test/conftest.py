import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ("..", "../../scout_piper_whole_body_mpc", "../../scout_piper_scene_repr/python"):
    p = os.path.normpath(os.path.join(HERE, rel))
    if p not in sys.path:
        sys.path.insert(0, p)
