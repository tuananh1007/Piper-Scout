"""Smoke tests — verify the ported modules import and basic math is correct.

Run without ROS:
    cd Piper_Scout_ws/Codes
    python -m pytest src/stem_grasp/test -q
"""

import importlib

import numpy as np


def test_pipeline_node_imports():
    mod = importlib.import_module("stem_grasp.pipeline_node")
    assert hasattr(mod, "StemGraspPipeline")
    assert hasattr(mod, "PipelineParams")
    assert hasattr(mod, "main")


def test_core_servo_classes_present():
    from stem_grasp import core
    assert hasattr(core, "StemVelocityObserver")
    assert hasattr(core, "OnlineJacobianEstimator")
    assert hasattr(core, "FullAdaptiveServoController")
    assert hasattr(core, "skeletonize_plant_points")
    assert hasattr(core, "extract_main_stem")
    assert hasattr(core, "select_grasp_candidates")


def test_servo_step_produces_clipped_velocity():
    from stem_grasp.core import FullAdaptiveServoController
    s = FullAdaptiveServoController(fx=600.0, fy=600.0)
    vel, diag = s.step(
        raw_uv=(330.0, 250.0),
        desired_uv=(320.0, 240.0),
        depth_z=0.30,
        force_n=0.0,
    )
    assert vel.shape == (3,)
    assert np.all(np.abs(vel) <= 0.05 + 1e-9)
    assert "error_norm" in diag
    assert "lambda" in diag


def test_select_candidates_returns_quaternions():
    from stem_grasp.core import select_grasp_candidates
    # Synthetic vertical stem with 30 points
    stem = np.column_stack(
        [np.zeros(30), np.linspace(0, 0.3, 30), np.zeros(30) + 0.4]
    )
    cands = select_grasp_candidates(stem, num_candidates=5, target_offset=0.12)
    assert len(cands) == 5
    for c in cands:
        assert c["pos"].shape == (3,)
        assert c["pre_pos"].shape == (3,)
        assert c["quat"].shape == (4,)


def test_skeleton_on_small_synthetic_cloud():
    from stem_grasp.core import (
        extract_main_stem,
        skeletonize_plant_points,
    )
    # A thin vertical "stem" (Y is vertical here)
    n = 200
    pts = np.column_stack(
        [
            np.random.normal(0.0, 0.001, n),
            np.linspace(0.0, 0.2, n),
            np.random.normal(0.0, 0.001, n) + 0.4,
        ]
    )
    G, sk = skeletonize_plant_points(pts, voxel_size=0.005)
    assert sk.shape[1] == 3
    stem = extract_main_stem(G, sk)
    assert stem.shape[1] == 3
