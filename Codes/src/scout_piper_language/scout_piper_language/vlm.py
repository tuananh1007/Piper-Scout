"""Sub-goal proposers: instruction + detections (+ image) -> one target (P5.1.3).

The VLM is not the motor controller (ROADMAP.md Phase 5): a proposer only
names one of the current detections, with a class and a rationale. The
instruction node then runs the reachability filter, shows the proposal with
a countdown, and only a confirmed proposal becomes a target for the existing
pipeline (stem_grasp + whole-body MPC + servo).

  RuleProposer   the parser + grounding of this package (default; offline)
  HttpProposer   a vision-language model behind an HTTP endpoint (a local
                 server on the Orin or the workstation). Request: JSON
                 {"instruction", "detections": [{index, class, colour, bbox,
                 centroid}], "image_jpeg_b64" (optional)}. Expected answer:
                 {"index": int | null, "class": str, "rationale": str,
                 "question": str | null}. The answer is validated: the index
                 must be one of the detections and the class must match it,
                 else the rule-based proposer answers instead and the reason
                 is reported. Model choice and quantisation (π₀ / Octo-Small /
                 a VLM, TensorRT-LLM INT8) are P5.1.1 / P5.1.2 on the Orin.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .grounding import DEFAULT_ALIASES, Detection, Grounding, class_of, ground
from .parser import Query, parse


@dataclass
class Proposal:
    query: Query
    grounding: Grounding
    source: str                                # rules | vlm | rules (vlm failed: ...)
    rationale: str = ""
    notes: List[str] = field(default_factory=list)


class RuleProposer:
    def __init__(self, aliases: Optional[Dict[str, List[str]]] = None):
        self.aliases = aliases or DEFAULT_ALIASES

    def propose(self, text: str, detections: List[Detection], robot_xy=(0.0, 0.0), image_jpeg: bytes = b"") -> Proposal:
        q = parse(text)
        g = ground(q, detections, robot_xy, self.aliases) if q.needs_target else Grounding("ok")
        return Proposal(q, g, "rules", rationale=g.message)


class HttpProposer:
    def __init__(self, url: str, timeout_s: float = 3.0, fallback: Optional[RuleProposer] = None,
                 aliases: Optional[Dict[str, List[str]]] = None):
        self.url, self.timeout = url, timeout_s
        self.aliases = aliases or DEFAULT_ALIASES
        self.fallback = fallback or RuleProposer(self.aliases)

    def _request(self, text: str, detections: List[Detection], image_jpeg: bytes) -> dict:
        body = {"instruction": text,
                "detections": [{"index": d.index, "class": d.cls, "colour": d.colour, "bbox": list(d.bbox),
                                "centroid": list(d.centroid)} for d in detections]}
        if image_jpeg:
            body["image_jpeg_b64"] = base64.b64encode(image_jpeg).decode("ascii")
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:   # noqa: S310 — operator-configured URL
            return json.loads(r.read().decode())

    def propose(self, text: str, detections: List[Detection], robot_xy=(0.0, 0.0), image_jpeg: bytes = b"") -> Proposal:
        q = parse(text)
        if not q.needs_target:
            return Proposal(q, Grounding("ok"), "rules")
        try:
            ans = self._request(text, detections, image_jpeg)
        except Exception as exc:  # noqa: BLE001 — network, timeout, bad JSON: fall back
            p = self.fallback.propose(text, detections, robot_xy)
            p.source = f"rules (vlm failed: {type(exc).__name__})"
            return p
        by_index = {d.index: d for d in detections}
        idx = ans.get("index")
        if idx is None:
            msg = ans.get("question") or ans.get("rationale") or "The model found no matching object."
            return Proposal(q, Grounding("none", message=str(msg), question=ans.get("question")), "vlm",
                            rationale=str(ans.get("rationale", "")))
        d = by_index.get(int(idx)) if isinstance(idx, (int, float, str)) and str(idx).lstrip("-").isdigit() else None
        cls = str(ans.get("class", "")).lower()
        if d is None or (cls and class_of(d.cls, self.aliases) not in (cls, None)
                         and class_of(cls, self.aliases) != class_of(d.cls, self.aliases)):
            p = self.fallback.propose(text, detections, robot_xy)
            p.source = "rules (vlm answer rejected: " + ("unknown index" if d is None else f"class {cls} != {d.cls}") + ")"
            return p
        g = Grounding("ok", chosen=d, candidates=[d], message=f"{q.describe().capitalize()}: {d.label()}.")
        return Proposal(q, g, "vlm", rationale=str(ans.get("rationale", "")))
