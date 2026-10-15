from pathlib import Path

import pytest

from piper_quickstart.scene_objects import allow_pairs, load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "scene_objects.yaml"


def test_shipped_config_loads():
    cfg = load_config(CONFIG)
    assert cfg["frame_id"] == "base_link"
    ids = {o["id"]: o for o in cfg["objects"]}
    assert ids["base_slab"]["size"] == [0.32, 0.20, 0.02] and ids["base_slab"]["center"] == [0.0, 0.0, 0.06]
    assert ids["pc_monitor"]["center"] == [0.20, -0.725, 0.65]
    assert ids["base_z_below_floor_no_go"]["size"] == [3.0, 3.0, 1.0]


def test_config_rejects_bad_boxes(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("frame_id: base_link\nobjects:\n  - {id: a, size: [1, 0, 1], center: [0, 0, 0]}\n")
    with pytest.raises(ValueError, match="positive"):
        load_config(p)
    p.write_text("frame_id: base_link\nobjects:\n  - {id: a, size: [1, 1, 1], center: [0, 0, 0]}\n"
                 "  - {id: a, size: [1, 1, 1], center: [0, 0, 0]}\n")
    with pytest.raises(ValueError, match="unique"):
        load_config(p)


def test_allow_pairs_extends_matrix_and_keeps_existing_entries():
    names = ["base_link", "link1"]
    table = [[False, True], [True, False]]  # SRDF: base_link-link1 adjacent
    n2, t2 = allow_pairs(names, table, [("base_link", "slab"), ("base_link", "floor")])
    assert n2 == ["base_link", "link1", "slab", "floor"]
    assert all(len(r) == 4 for r in t2) and len(t2) == 4
    i = {n: k for k, n in enumerate(n2)}
    assert t2[i["base_link"]][i["link1"]] and t2[i["link1"]][i["base_link"]]       # kept
    assert t2[i["base_link"]][i["slab"]] and t2[i["slab"]][i["base_link"]]         # added, symmetric
    assert not t2[i["link1"]][i["slab"]] and not t2[i["link1"]][i["floor"]]        # still checked
    assert table == [[False, True], [True, False]]                                 # input untouched
