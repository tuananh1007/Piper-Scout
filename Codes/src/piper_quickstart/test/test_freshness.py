import math

from piper_quickstart.freshness import StampWatch


def test_new_data_only_when_stamp_advances():
    w = StampWatch()
    assert w.update(100, now_mono=0.0) is True
    assert w.update(100, now_mono=0.1) is False   # cached repeat
    assert w.update(90, now_mono=0.2) is False    # older stamp
    assert w.update(110, now_mono=0.3) is True
    assert w.new_count == 2


def test_frozen_stream_ages_while_still_arriving():
    w = StampWatch()
    w.update(5, now_mono=1.0)
    for i in range(1, 11):                         # 10 more repeats over 1 s
        w.update(5, now_mono=1.0 + 0.1 * i)
    assert math.isclose(w.arrival_age(2.0), 0.0)
    assert math.isclose(w.data_age(2.0), 1.0)


def test_never_seen_is_infinitely_old():
    w = StampWatch()
    assert w.data_age(10.0) == float("inf") and w.arrival_age(10.0) == float("inf")
