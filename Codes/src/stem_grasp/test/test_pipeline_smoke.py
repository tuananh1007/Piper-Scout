"""Smoke test — verifies the pipeline node imports cleanly without ROS running."""

import importlib


def test_pipeline_imports():
    mod = importlib.import_module("stem_grasp.pipeline_node")
    assert hasattr(mod, "StemGraspPipeline")
    assert hasattr(mod, "PipelineParams")
    assert hasattr(mod, "main")


def test_segmentation_imports():
    mod = importlib.import_module("stem_grasp.segmentation_node")
    assert hasattr(mod, "SegmentationNode")
    assert hasattr(mod, "main")


def test_core_imports():
    mod = importlib.import_module("stem_grasp.core")
    assert hasattr(mod, "FullAdaptiveServoController")
