"""nvblox_semantic.launch.py: the classes:= subset (one nvblox process per class)."""

import importlib.util
import os

import pytest

LAUNCH = os.path.join(os.path.dirname(__file__), "..", "launch", "nvblox_semantic.launch.py")


@pytest.fixture(scope="module")
def launch_file():
    # the launch file imports ROS's launch packages; skip per test without them (a
    # module-level skip would stop ROS's pytest plugin collecting the other files)
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    spec = importlib.util.spec_from_file_location("nvblox_semantic_launch", LAUNCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_subset_keeps_the_canonical_order(launch_file):
    assert launch_file.selected_classes("target, stem") == ["stem", "target"]
    assert launch_file.selected_classes(",".join(launch_file.CLASS_NAMES)) == launch_file.CLASS_NAMES


@pytest.mark.parametrize("spec", ["stem,trunk", "", " , "])
def test_unknown_or_empty_class_lists_are_rejected(spec, launch_file):
    with pytest.raises(ValueError):
        launch_file.selected_classes(spec)
