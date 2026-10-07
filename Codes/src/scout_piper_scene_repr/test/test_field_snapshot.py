"""Distance-field snapshot (P1.7.7): export, numpy reference sampler, C++ sampler.

    cd Codes/src/scout_piper_scene_repr && python -m pytest test -q

The C++ checks compile test/field_query_driver.cpp with the system compiler
and are skipped when none is available.
"""

import os
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

from scout_piper_scene_repr_py.distance_query import SemanticDistanceQuery  # noqa: E402
from scout_piper_scene_repr_py.field import (  # noqa: E402
    AGE_UNKNOWN, FieldSampler, export_field, fill_msg)
from scout_piper_scene_repr_py.policy import DEFAULT_POLICIES  # noqa: E402
from scout_piper_scene_repr_py.voxel_map import SemanticVoxelMap, grid_around  # noqa: E402
from test_distance_query import LEAF_Z, STEM_R, STEM_X, WALL_Z, build  # noqa: E402

VOXEL = 0.005


@pytest.fixture(scope="module")
def scene():
    vmap = build(voxel=VOXEL)                       # stamps 0, 1, 2
    q = SemanticDistanceQuery(vmap, DEFAULT_POLICIES, max_age_s=100.0)
    snap = export_field(q)
    return vmap, q, snap, FieldSampler(snap)


def probe_points(vmap, n=400, seed=0):
    spec = vmap.spec
    rng = np.random.default_rng(seed)
    lo, hi = spec.origin, spec.origin + np.array(spec.shape) * spec.voxel_size
    pts = rng.uniform(lo - 0.02, hi + 0.02, size=(n, 3))       # some outside the grid
    near_stem = np.column_stack([STEM_X + rng.uniform(-0.03, 0.03, n // 2),
                                 rng.uniform(-0.1, 0.1, n // 2),
                                 0.40 + rng.uniform(-0.03, 0.03, n // 2)])
    return np.vstack([pts, near_stem])


# ------------------------------------------------------------------ export
def test_export_layout(scene):
    vmap, _, snap, _ = scene
    assert snap.shape == vmap.spec.shape and snap.stamp == 2.0
    assert snap.hard_classes == ["stem", "branch", "other"]
    assert snap.hard_distance.dtype == np.float32 and snap.hard_class.dtype == np.uint8
    assert snap.soft_class == "leaf" and snap.soft_max_penetration == pytest.approx(0.02)
    assert np.all((snap.age_ds == AGE_UNKNOWN) == ~vmap.observed)
    assert snap.age_ds[vmap.observed].max() <= 20                 # seen within the 3 frames


def test_export_before_first_frame_is_none():
    vmap = SemanticVoxelMap(grid_around([0, 0, 0.4], 0.1, 0.01))
    assert export_field(SemanticDistanceQuery(vmap)) is None


def test_fill_msg_round_trip(scene):
    _, _, snap, _ = scene
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace()),
                          origin=SimpleNamespace())
    fill_msg(msg, snap, "odom")
    n = int(np.prod(snap.shape))
    assert msg.header.frame_id == "odom" and (msg.header.stamp.sec, msg.header.stamp.nanosec) == (2, 0)
    assert list(msg.size) == list(snap.shape)
    assert len(msg.hard_distance) == len(msg.hard_class) == len(msg.age_ds) == len(msg.soft_distance) == n
    back = np.frombuffer(msg.hard_distance.tobytes(), np.float32).reshape(snap.shape)
    assert np.array_equal(back, snap.hard_distance, equal_nan=True)
    assert np.array_equal(np.frombuffer(msg.hard_class, np.uint8).reshape(snap.shape), snap.hard_class)
    assert np.array_equal(np.frombuffer(msg.age_ds.tobytes(), np.uint16).reshape(snap.shape), snap.age_ds)


# ------------------------------------------------------- sampler vs query
def test_sampler_is_conservative_and_matches_query(scene):
    vmap, q, snap, fs = scene
    pts = probe_points(vmap)
    ref = q.query(pts, now=snap.stamp, gradient=False)
    got = fs.query(pts, now=snap.stamp, max_voxel_age_s=100.0)
    assert np.array_equal(got["fresh"], ref.valid)
    ok = ref.valid
    # merged-min interpolation is never less conservative than the per-class query ...
    assert np.all(got["hard"][ok] <= ref.hard_distance[ok] + 1e-6)
    # ... and equal where one hard class dominates the whole interpolation cell
    near = ok & (np.abs(pts[:, 2] - 0.40) < 0.03) & (np.abs(pts[:, 0] - STEM_X) < 0.03)
    assert near.sum() > 50
    assert np.allclose(got["hard"][near], ref.hard_distance[near], atol=1e-5)
    assert np.allclose(got["soft"], ref.class_distance["leaf"], atol=1e-5)


def test_sphere_statuses(scene):
    vmap, _, snap, fs = scene
    leaf_x, behind_wall = -0.06, WALL_Z + 0.03
    centers = np.array([
        [STEM_X, 0.0, 0.40 - STEM_R - 0.015],     # 1.5 cm in front of the stem, r = 1 cm → free
        [STEM_X, 0.0, 0.40 - STEM_R - 0.004],     # r = 1 cm reaches the padded stem → hard
        [leaf_x, 0.0, LEAF_Z - 0.005],            # r = 1 cm: ≤ 2 cm into the leaf → allowed
        [leaf_x, 0.0, LEAF_Z + 0.030],            # sphere behind the leaf: leaf occludes → unknown
        [0.0, 0.10, behind_wall],                 # behind the wall: never observed → unknown
        [1.0, 1.0, 1.0],                          # outside the grid
    ])
    radii = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    st = [s for s, _, _ in fs.check_spheres(centers, radii, now=snap.stamp)]
    assert st == ["free", "hard", "free", "unknown", "unknown", "outside"]
    # a 2.5 cm sphere centred on the leaf overlaps it by more than the 2 cm cap
    deep = fs.check_spheres([[leaf_x, 0.0, LEAF_Z + 0.001]], [0.025], now=snap.stamp)
    assert deep[0][0] == "soft" and deep[0][2] and deep[0][1] < 0
    # observed voxels go stale once older than max_voxel_age_s
    stale = fs.check_spheres(centers[:1], radii[:1], now=snap.stamp + 31.0, max_voxel_age_s=30.0)
    assert stale[0][0] == "stale" and stale[0][2]
    # unknown space is free when the caller opts out
    st2 = [s for s, _, _ in fs.check_spheres(centers, radii, now=snap.stamp, unknown_is_occupied=False)]
    assert st2[4] == "free"


# ----------------------------------------------------------- C++ sampler
@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    cxx = shutil.which(os.environ.get("CXX", "c++")) or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("drv") / "field_query_driver"
    inc = os.path.join(HERE, "..", "include")
    subprocess.run([cxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-I", inc,
                    os.path.join(HERE, "field_query_driver.cpp"), "-o", str(out)], check=True)
    return str(out)


def write_field(path, snap):
    with open(path, "wb") as f:
        f.write(struct.pack("<3dd3Iddb", *snap.origin, snap.voxel_size, *snap.shape, snap.stamp,
                            snap.soft_max_penetration, snap.soft_distance is not None))
        f.write(snap.hard_distance.astype("<f4").tobytes())
        f.write(snap.hard_class.astype(np.uint8).tobytes())
        if snap.soft_distance is not None:
            f.write(snap.soft_distance.astype("<f4").tobytes())
        f.write(snap.age_ds.astype("<u2").tobytes())


def run_driver(driver, tmp_path, snap, pts, radii, now, max_age, unknown_occ, soft):
    fpath, qpath = tmp_path / "field.bin", tmp_path / "query.bin"
    write_field(fpath, snap)
    with open(qpath, "wb") as f:
        f.write(struct.pack("<Iddbb", len(pts), now, max_age, unknown_occ, soft))
        f.write(np.column_stack([pts, radii]).astype("<f8").tobytes())
    out = subprocess.run([driver, str(fpath), str(qpath)], check=True, capture_output=True, text=True)
    rows = [line.split() for line in out.stdout.strip().splitlines()]
    return rows


@pytest.mark.parametrize("now,max_age,unknown_occ,soft", [
    (2.0, 100.0, 1, 1), (2.0, 100.0, 0, 1), (2.5, 1.0, 1, 0), (40.0, 30.0, 1, 1)])
def test_cpp_matches_numpy_reference(scene, driver, tmp_path, now, max_age, unknown_occ, soft):
    vmap, _, snap, fs = scene
    pts = probe_points(vmap, seed=1)
    radii = np.random.default_rng(2).uniform(0.0, 0.03, len(pts))
    rows = run_driver(driver, tmp_path, snap, pts, radii, now, max_age, unknown_occ, soft)
    ref = fs.query(pts, now=now, max_voxel_age_s=max_age)
    sph = fs.check_spheres(pts, radii, now=now, max_voxel_age_s=max_age,
                           unknown_is_occupied=bool(unknown_occ), check_soft=bool(soft))
    assert len(rows) == len(pts)
    for j, r in enumerate(rows):
        assert [int(r[0]), int(r[1]), int(r[2])] == [ref["in_bounds"][j], ref["known"][j], ref["fresh"][j]]
        assert float(r[3]) == pytest.approx(ref["hard"][j], abs=1e-9)
        assert int(r[4]) == ref["hard_class"][j]
        assert float(r[5]) == pytest.approx(ref["soft"][j], abs=1e-9)
        status, clearance, collision = sph[j]
        assert r[6] == status
        assert float(r[7]) == pytest.approx(clearance, abs=1e-9) or (np.isinf(clearance) and r[7] == "inf")
        assert int(r[8]) == int(collision)


def test_cpp_sphere_cover(driver):
    out = subprocess.run([driver, "--cover"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout
    assert out.stdout.strip() == "ok"
