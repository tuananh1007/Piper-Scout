"""Ground a parsed query in the current detections; ask back when ambiguous.

Detections come from ``/stem_grasp/detections`` (stem_grasp ``detections.py``):
class name, confidence, bounding box, centroid, area, colour name and, when
the node has depth and TF, a 3-D point in the robot's base frame.

Directions are the operator's view of the camera image: left / right by the
image column, top / bottom by the 3-D height when known (else the image
row), near / far by the distance from the robot base (else the depth or the
image row). Ordinals count along ``order`` (default left to right).

``ground`` returns a ``Grounding``:
  ok          exactly one detection fits;
  ambiguous   several fit and nothing in the query separates them: the
              question lists them by position ("the left, middle or right
              one?"), and ``resolve_followup`` takes the answer ("the
              left one", "the second", "number 3");
  none        nothing fits; the message says what is seen instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .parser import Phrase, Query, parse

DEFAULT_ALIASES = {
    "flower": ["flower", "blossom", "bloom", "rose", "daisy", "tulip", "sunflower"],
    "stem": ["stem", "stalk", "peduncle", "potted plant"],
    "branch": ["branch", "twig"],
    "leaf": ["leaf", "leaves"],
    "fruit": ["fruit", "tomato", "strawberry", "pepper", "berry", "apple", "orange", "banana"],
    "bud": ["bud"],
}


@dataclass
class Detection:
    index: int
    cls: str
    conf: float
    bbox: Sequence[int]
    centroid: Sequence[float]
    area_px: int
    colour: str = ""
    p_base: Optional[np.ndarray] = None
    polygon: Optional[list] = None
    depth_m: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict, p_base=None) -> "Detection":
        pc = d.get("p_camera")
        return cls(index=int(d["index"]), cls=str(d.get("class", "")), conf=float(d.get("conf", 0.0)),
                   bbox=list(d.get("bbox", [0, 0, 0, 0])), centroid=list(d.get("centroid", [0.0, 0.0])),
                   area_px=int(d.get("area_px", 0)), colour=str(d.get("colour", "")),
                   p_base=None if p_base is None else np.asarray(p_base, float), polygon=d.get("polygon"),
                   depth_m=None if pc is None else float(pc[2]))

    def width_m(self, fx: Optional[float]) -> Optional[float]:
        """Width across the object from its outline (short side of the
        minimum-area rectangle) at its depth; None without depth or outline."""
        if not fx or self.depth_m is None or not self.polygon or len(self.polygon) < 3:
            return None
        import cv2  # noqa: PLC0415
        (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(self.polygon, np.float32))
        return float(min(w, h) * self.depth_m / fx)

    def label(self) -> str:
        return f"#{self.index + 1} {self.colour + ' ' if self.colour else ''}{self.cls}"


@dataclass
class Grounding:
    status: str                              # ok | ambiguous | none
    chosen: Optional[Detection] = None
    candidates: List[Detection] = field(default_factory=list)
    message: str = ""
    question: Optional[str] = None
    options: List[str] = field(default_factory=list)


def class_of(det_cls: str, aliases: Dict[str, List[str]]) -> Optional[str]:
    n = det_cls.lower()
    for c, names in aliases.items():
        if any(a == n or a in n for a in names):
            return c
    return None


def _key(d: Detection, axis: str, robot_xy) -> float:
    if axis in ("left", "right"):
        return d.centroid[0]
    if axis in ("top", "bottom"):
        return -d.p_base[2] if d.p_base is not None else d.centroid[1]
    if axis in ("near", "far"):
        if d.p_base is not None:
            return float(np.linalg.norm(d.p_base[:2] - np.asarray(robot_xy)))
        return -d.centroid[1]
    if axis in ("big", "small"):
        return -d.area_px
    raise ValueError(axis)


def _sorted(dets: List[Detection], axis: str, robot_xy) -> List[Detection]:
    """Sorted so that index 0 is the extreme named by ``axis``."""
    rev = axis in ("right", "bottom", "far", "small")
    return sorted(dets, key=lambda d: _key(d, axis, robot_xy), reverse=rev)


def _filter(phrase: Phrase, dets: List[Detection], aliases) -> List[Detection]:
    out = [d for d in dets if phrase.cls is None or class_of(d.cls, aliases) == phrase.cls]
    for a in phrase.attributes:
        if a in ("big", "small"):
            if len(out) >= 2:
                areas = np.array([d.area_px for d in out])
                med = float(np.median(areas))
                out = [d for d in out if (d.area_px >= med if a == "big" else d.area_px <= med)]
        elif a in ("ripe", "unripe"):
            out = [d for d in out if (d.colour in ("red", "orange")) == (a == "ripe")]
        elif a in ("open", "closed"):
            continue                                   # not observable from a detection; ignored
        else:
            out = [d for d in out if d.colour == a]
    return out


def _select(phrase: Phrase, dets: List[Detection], robot_xy) -> List[Detection]:
    if not dets:
        return dets
    if phrase.superlative == "middle":
        s = _sorted(dets, "left", robot_xy)
        return [s[len(s) // 2]] if len(s) % 2 == 1 else s[len(s) // 2 - 1:len(s) // 2 + 1]
    if phrase.superlative is not None:
        s = _sorted(dets, phrase.superlative, robot_xy)
        return [s[0]]
    if phrase.ordinal is not None:
        s = _sorted(dets, phrase.order, robot_xy)
        k = phrase.ordinal
        if k == -1:
            return [s[-1]]
        return [s[k - 1]] if 1 <= k <= len(s) else []
    return dets


def _relate(relation: str, ref: Detection, dets: List[Detection], robot_xy) -> List[Detection]:
    margin = 0.25 * (ref.bbox[2] - ref.bbox[0] + 1)
    out = []
    for d in dets:
        if d.index == ref.index:
            continue
        du = d.centroid[0] - ref.centroid[0]
        if relation == "left_of" and du < -margin:
            out.append(d)
        elif relation == "right_of" and du > margin:
            out.append(d)
        elif relation in ("above", "below"):
            if d.p_base is not None and ref.p_base is not None:
                dz = d.p_base[2] - ref.p_base[2]
                ok = dz > 0.02 if relation == "above" else dz < -0.02
            else:
                dv = d.centroid[1] - ref.centroid[1]
                ok = dv < -margin if relation == "above" else dv > margin
            if ok:
                out.append(d)
        elif relation in ("behind", "in_front_of"):
            if d.p_base is not None and ref.p_base is not None:
                dd = _key(d, "near", robot_xy) - _key(ref, "near", robot_xy)
                if (dd > 0.02) if relation == "behind" else (dd < -0.02):
                    out.append(d)
    if relation == "next_to":
        def dist(d):
            if d.p_base is not None and ref.p_base is not None:
                return float(np.linalg.norm(d.p_base - ref.p_base))
            return float(np.hypot(d.centroid[0] - ref.centroid[0], d.centroid[1] - ref.centroid[1]))
        others = sorted([d for d in dets if d.index != ref.index], key=dist)
        out = others[:1]
    return out


def _positions(dets: List[Detection]) -> List[str]:
    """Position words for listing candidates left to right."""
    n = len(dets)
    if n == 2:
        return ["left", "right"]
    if n == 3:
        return ["left", "middle", "right"]
    return [f"{_ord(k + 1)} from the left" for k in range(n)]


def _ord(k: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(k, f"#{k}")


def summarize(dets: List[Detection], aliases) -> str:
    if not dets:
        return "I don't see anything I recognise."
    counts: Dict[str, List[str]] = {}
    for d in dets:
        c = class_of(d.cls, aliases) or d.cls
        counts.setdefault(c, []).append(d.colour)
    parts = []
    for c, cols in counts.items():
        cs = sorted({x for x in cols if x})
        parts.append(f"{len(cols)} {c}{'s' if len(cols) > 1 else ''}" + (f" ({', '.join(cs)})" if cs else ""))
    return "I see " + ", ".join(parts) + "."


def ground(q: Query, detections: List[Detection], robot_xy=(0.0, 0.0),
           aliases: Optional[Dict[str, List[str]]] = None) -> Grounding:
    aliases = aliases or DEFAULT_ALIASES
    pool = _filter(q.target, detections, aliases)
    if q.relation and q.reference is not None:
        refs = _select(q.reference, _filter(q.reference, detections, aliases), robot_xy)
        if not refs:
            return Grounding("none", message=f"I can't find {q.reference.describe()}. " + summarize(detections, aliases))
        if len(refs) > 1:
            return _ambiguous(refs, q.reference.describe(), "Which one do you mean as the reference")
        pool = _relate(q.relation, refs[0], pool, robot_xy)
        if not pool:
            return Grounding("none", message=f"I don't see {q.target.describe()} "
                                             f"{q.relation.replace('_', ' ')} {q.reference.describe()}.")
    sel = _select(q.target, pool, robot_xy)
    if not sel:
        what = q.target.describe()
        return Grounding("none", message=f"I can't find {what}. " + summarize(detections, aliases))
    if len(sel) == 1:
        d = sel[0]
        return Grounding("ok", chosen=d, candidates=sel, message=f"{q.describe().capitalize()}: {d.label()}.")
    return _ambiguous(sel, q.target.describe(), "Which one")


def _ambiguous(dets: List[Detection], what: str, ask: str) -> Grounding:
    s = sorted(dets, key=lambda d: d.centroid[0])
    pos = _positions(s)
    opts = [f"the {p} one" for p in pos]
    q = f"I see {len(s)} that match {what}. {ask}: " + (", ".join(opts[:-1]) + " or " + opts[-1]) + "?"
    return Grounding("ambiguous", candidates=s, message=q, question=q, options=opts)


def resolve_followup(answer: str, previous: Grounding, robot_xy=(0.0, 0.0)) -> Grounding:
    """Pick among ``previous.candidates`` with a short answer ("the left
    one", "the second", "number 3", "#2", or a detection label number)."""
    q = parse("grasp " + answer)
    cands = previous.candidates
    if not cands:
        return Grounding("none", message="There is nothing to choose from.")
    ph = q.target
    if ph.superlative is None and ph.ordinal is None and not ph.attributes:
        return Grounding("ambiguous", candidates=cands, message=previous.question or "", question=previous.question,
                         options=previous.options)
    sel = _select(ph, _filter(Phrase(attributes=ph.attributes), cands, DEFAULT_ALIASES) if ph.attributes else cands,
                  robot_xy)
    if len(sel) == 1:
        return Grounding("ok", chosen=sel[0], candidates=sel, message=f"OK: {sel[0].label()}.")
    return Grounding("ambiguous", candidates=cands, message=previous.question or "", question=previous.question,
                     options=previous.options)
