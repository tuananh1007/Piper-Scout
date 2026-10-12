"""Per-instance descriptions for the operator layer (detections.py)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from stem_grasp.detections import colour_name, describe_instances  # noqa: E402


def test_describe_two_flowers_with_colour_and_3d_point():
    H, W = 120, 160
    bgr = np.zeros((H, W, 3), np.uint8)
    bgr[:] = (40, 120, 40)                                      # green background
    m1 = np.zeros((H, W), np.uint8)
    m1[20:40, 20:40] = 1
    bgr[20:40, 20:40] = (0, 0, 220)                             # red (BGR)
    m2 = np.zeros((H, W), np.uint8)
    m2[60:90, 100:130] = 1
    bgr[60:90, 100:130] = (0, 220, 230)                         # yellow
    tiny = np.zeros((H, W), np.uint8)
    tiny[0:2, 0:2] = 1
    depth = np.full((H, W), 0.5, np.float32)
    K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
    d = describe_instances([m1, m2, tiny], [3, 3, 3], [0.9, 0.8, 0.7], {3: "flower"}, bgr, depth, K)
    assert len(d) == 2                                          # the 4-pixel instance is dropped
    assert [x["colour"] for x in d] == ["red", "yellow"]
    assert d[0]["class"] == "flower" and d[0]["bbox"] == [20, 20, 39, 39] and d[0]["area_px"] == 400
    poly = np.array(d[0]["polygon"])
    assert len(poly) >= 4 and poly[:, 0].min() == 20 and poly[:, 0].max() == 39   # the square's outline
    x, y, z = d[1]["p_camera"]
    assert z == pytest.approx(0.5) and x == pytest.approx((114.5 - 80) * 0.5 / 100, abs=1e-3)


def test_colour_names():
    assert colour_name(0, 200, 200) == "red" and colour_name(176, 200, 200) == "red"
    assert colour_name(60, 200, 200) == "green" and colour_name(60, 10, 230) == "white"
    assert colour_name(60, 200, 20) == "black"
