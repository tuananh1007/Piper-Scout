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


def test_caption_instances_become_detections():
    """Grounded-SAM's per-caption instances (stand-in boxes for the models)
    become classes named after the captions."""
    from stem_grasp.detections import collect_caption_instances
    H, W = 100, 120

    def box(x0, y0, x1, y1):
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        return m
    found = {"flower": [(box(20, 20, 44, 40), 0.6), (box(80, 20, 104, 40), 0.55)], "leaf": [(box(40, 70, 76, 90), 0.5)]}
    masks, classes, confs, names = collect_caption_instances(
        "stem", [(box(58, 40, 62, 100), 0.9)], ["flower", "leaf", "Flower", ""], lambda c: found.get(c, []))
    assert names == {0: "stem", 1: "flower", 2: "leaf"}                 # repeated and empty captions skipped
    assert classes == [0, 1, 1, 2] and confs == [0.9, 0.6, 0.55, 0.5]
    bgr = np.zeros((H, W, 3), np.uint8)
    bgr[20:40, 20:44] = (0, 0, 220)
    dets = describe_instances(masks, classes, confs, names, bgr)
    assert [d["class"] for d in dets] == ["stem", "flower", "flower", "leaf"] and dets[1]["colour"] == "red"
