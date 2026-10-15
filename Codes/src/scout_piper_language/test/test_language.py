"""Parser, grounding, clarification, reachability and the instruction session (no ROS)."""

import numpy as np
import pytest

from scout_piper_language.grounding import Detection, ground, resolve_followup, summarize, DEFAULT_ALIASES
from scout_piper_language.parser import parse
from scout_piper_language.session import InstructionSession
from scout_piper_language.vlm import HttpProposer, RuleProposer


def dets():
    return [Detection(0, "flower", 0.9, [10, 10, 30, 30], [20, 20], 400, "red", np.array([0.8, 0.2, 0.50])),
            Detection(1, "flower", 0.8, [60, 10, 80, 30], [70, 20], 380, "yellow", np.array([0.8, 0.0, 0.45])),
            Detection(2, "flower", 0.85, [110, 10, 130, 30], [120, 20], 420, "red", np.array([0.9, -0.2, 0.60])),
            Detection(3, "potted plant", 0.7, [50, 40, 90, 120], [70, 80], 3000, "green", np.array([0.85, 0.0, 0.3]))]


@pytest.mark.parametrize("text, action, cls, attrs, ordinal, order, sup, rel", [
    ("Grasp the red flower", "grasp", "flower", ["red"], None, "left", None, None),
    ("pick the second flower from the left", "grasp", "flower", [], 2, "left", None, None),
    ("take the 3rd one from the right", "grasp", None, [], 3, "right", None, None),
    ("please grab the tallest stem", "grasp", "stem", [], None, "left", "top", None),
    ("look at the leaf behind the white flower", "inspect", "leaf", [], None, "left", None, "behind"),
    ("grab the flower to the left of the yellow one", "grasp", "flower", [], None, "left", None, "left_of"),
    ("grasp number 2", "grasp", None, [], 2, "left", None, None),
    ("grasp the one on the right", "grasp", None, [], None, "left", "right", None),
    ("pick the big ripe tomato", "grasp", "fruit", ["big", "ripe"], None, "left", None, None),
])
def test_parse(text, action, cls, attrs, ordinal, order, sup, rel):
    q = parse(text)
    assert (q.action, q.target.cls, q.target.attributes, q.target.ordinal, q.target.order, q.target.superlative,
            q.relation) == (action, cls, attrs, ordinal, order, sup, rel)
    assert not q.unknown_words


def test_parse_refusals_and_commands():
    assert parse("cut the stem").action == "unsupported" and parse("cut the stem").unsupported == "cut"
    assert parse("stop!").action == "stop" and parse("let go").action == "release"
    assert parse("scan").action == "scan" and parse("hello there").action == "unknown"
    assert parse("grasp the zebra").unknown_words == ["zebra"]
    q = parse("grab the flower to the left of the yellow one")
    assert q.reference.cls == "flower" and q.reference.attributes == ["yellow"]


def test_ground_unique_ambiguous_none_and_relations():
    D = dets()
    g = ground(parse("grasp the yellow flower"), D)
    assert g.status == "ok" and g.chosen.index == 1
    g = ground(parse("grasp the red flower"), D)
    assert g.status == "ambiguous" and [d.index for d in g.candidates] == [0, 2] and len(g.options) == 2
    assert resolve_followup("the right one", g).chosen.index == 2
    assert resolve_followup("the first", g).chosen.index == 0
    assert resolve_followup("hmm", g).status == "ambiguous"
    assert ground(parse("pick the second flower from the left"), D).chosen.index == 1
    assert ground(parse("pick the second flower from the right"), D).chosen.index == 1
    assert ground(parse("grab the flower to the left of the yellow one"), D).chosen.index == 0
    assert ground(parse("grab the flower right of the yellow one"), D).chosen.index == 2
    assert ground(parse("grasp the tallest flower"), D).chosen.index == 2
    assert ground(parse("grasp the nearest flower"), D).chosen.index == 1
    assert ground(parse("grasp the stem"), D).chosen.index == 3              # "potted plant" alias
    g = ground(parse("grasp the white flower"), D)
    assert g.status == "none" and "3 flowers" in g.message
    assert ground(parse("grasp the fifth flower"), D).status == "none"
    assert summarize([], DEFAULT_ALIASES).startswith("I don't see")


class FakeVerdict:
    def __init__(self, ok, code="ok", message="I can reach it.", advisories=()):
        self.ok, self.code, self.message, self.advisories = ok, code, message, list(advisories)

    def as_dict(self):
        return {"ok": self.ok, "code": self.code}


class FakeChecker:
    def __init__(self):
        self.calls = []

    def check(self, p, x, **kw):
        self.calls.append((p, kw))
        if p[2] > 0.55:
            return FakeVerdict(False, "too_high", "It is too high for me to reach.")
        return FakeVerdict(True, advisories=["It is behind a leaf; I will look from another side first."])


def test_session_countdown_clarify_reject_and_stop():
    t = [0.0]
    chk = FakeChecker()
    s = InstructionSession(RuleProposer(), chk, countdown_s=2.0, clock=lambda: t[0], width_fn=lambda d: 0.004)
    x = np.zeros(9)
    ev = s.instruction("grasp the yellow flower", dets(), x)
    prop = [e for e in ev if e["type"] == "proposal"][0]
    assert prop["target"]["index"] == 1 and prop["advisories"] and s.state == "countdown"
    assert chk.calls[0][1]["width_m"] == 0.004 and chk.calls[0][1]["target_class"] == "flower"
    t[0] = 1.0
    assert s.tick() == []
    t[0] = 2.1
    ev = s.tick()
    assert ev[0]["type"] == "commit" and ev[0]["target"]["index"] == 1 and s.state == "idle"
    # clarification, then the answer
    ev = s.instruction("grasp the red flower", dets(), x)
    assert ev[0]["type"] == "question" and s.state == "clarifying"
    ev = s.instruction("the left one", dets(), x)
    assert ev[0]["type"] == "proposal" and ev[0]["target"]["index"] == 0
    assert s.cancel()[0]["type"] == "cancelled" and s.state == "idle"
    # rejected by reachability: nothing pending
    ev = s.instruction("grasp the tallest flower", dets(), x)
    assert any(e["type"] == "rejected" and e["code"] == "too_high" for e in ev) and s.state == "idle"
    # stop during a countdown: immediate, cancels
    s.instruction("grasp the yellow flower", dets(), x)
    ev = s.instruction("stop", dets(), x)
    assert ev[0] == {"type": "command", "name": "stop", "reason": "stop"} and ev[1]["type"] == "cancelled"
    # refusals and unknown words never propose
    for text in ("cut the stem", "grasp the zebra", "hello"):
        ev = s.instruction(text, dets(), x)
        assert all(e["type"] == "say" for e in ev) and s.state == "idle"
    # confirm commits at once; release has no target
    ev = s.instruction("release", dets(), x)
    assert ev[0]["type"] == "proposal" and ev[0]["target"] is None
    assert s.confirm()[0]["type"] == "commit"


def test_http_proposer_validates_and_falls_back():
    class Stub(HttpProposer):
        def __init__(self, answer):
            super().__init__("http://unused")
            self.answer = answer

        def _request(self, text, detections, image_jpeg):
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer
    D = dets()
    p = Stub({"index": 2, "class": "flower", "rationale": "the red one on the right"}).propose("grasp the red flower on the right", D)
    assert p.source == "vlm" and p.grounding.chosen.index == 2
    p = Stub({"index": 7, "class": "flower"}).propose("grasp the yellow flower", D)
    assert p.source.startswith("rules (vlm answer rejected") and p.grounding.chosen.index == 1
    p = Stub({"index": 1, "class": "leaf"}).propose("grasp the yellow flower", D)
    assert "class leaf" in p.source
    p = Stub(TimeoutError()).propose("grasp the yellow flower", D)
    assert p.source == "rules (vlm failed: TimeoutError)" and p.grounding.chosen.index == 1


def test_reachability_reasons():
    from scout_piper_language.reachability import ReachabilityChecker
    from scout_piper_whole_body_mpc.dynamics.whole_body import WholeBodyModel
    rc = ReachabilityChecker(WholeBodyModel())
    x = np.r_[0.0, 0.0, 0.0, 0.0, 1.2, -1.0, 0.0, 0.5, 0.0]
    assert rc.check(np.array([1.0, 0.1, 0.45]), x).ok
    assert rc.check(np.array([1.0, 0.0, 1.4]), x).code == "too_high"
    assert rc.check(np.array([1.0, 0.0, 0.0]), x).code == "too_low"
    assert rc.check(np.array([3.5, 0.0, 0.4]), x).code == "too_far"
    assert rc.check(None, x).code == "no_depth"
    assert rc.check(np.array([1.0, 0.1, 0.45]), x, width_m=0.08, target_class="stem").code == "too_thick"
    # a neighbouring stem 2 cm from the target: crowded; an obstacle wall: no pose
    field = pytest.importorskip("scout_piper_scene_repr_py.field")
    n, vs = 60, 0.01
    origin = np.array([0.7, -0.3, 0.15])
    g = origin + (np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1) + 0.5) * vs
    tgt = np.array([1.0, 0.0, 0.45])

    def stem_d(x0, y0):
        return np.linalg.norm(g[..., :2] - [x0, y0], axis=-1) - 0.004

    hard = np.minimum(stem_d(1.0, 0.0), stem_d(1.0, 0.02)).astype(np.float32)
    snap = field.DistanceFieldSnapshot(origin=origin, voxel_size=vs, shape=(n,) * 3, stamp=0.0, hard_classes=["stem"],
                                       hard_distance=hard, hard_class=np.zeros((n,) * 3, np.uint8),
                                       age_ds=np.zeros((n,) * 3, np.uint16))
    v = rc.check(tgt, x, snapshot=snap)
    assert v.code == "crowded" and "too close to another stem" in v.message
