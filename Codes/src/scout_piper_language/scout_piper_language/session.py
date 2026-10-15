"""Instruction session: clarify, propose, count down, commit or abort (P5.3.4).

ROS-free state machine behind the instruction node, so it is testable with a
fake clock. Events it emits (dicts with ``type``):

  say        plain-language text for the operator (GUI log, TTS)
  question   a clarification with ``options`` (the next instruction is read
             as the answer)
  proposal   what the robot is about to do: action, target (detection, 3-D
             point), the reachability verdict and advisories, and a
             ``countdown_s`` (2 s) after which it commits unless cancelled;
             ``confirm`` commits at once
  commit     the action goes ahead (the node turns it into pipeline / JEPA /
             active-perception inputs)
  command    stop (immediate, never counted down), cancelling any proposal
  cancelled  the proposal was cancelled or aborted

States: idle, clarifying, countdown. A new instruction during a countdown
replaces the proposal. Unsupported actions (cut, pull, push) and words the
parser does not know are answered, never acted on.
"""

from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from .grounding import Detection, Grounding, resolve_followup, summarize
from .parser import parse

ACTIONS_HELP = "I can grasp, inspect, release, scan or stop."


class InstructionSession:
    def __init__(self, proposer, checker=None, countdown_s: float = 2.0,
                 clock: Callable[[], float] = time.monotonic, width_fn: Optional[Callable] = None):
        self.proposer = proposer
        self.checker = checker
        self.countdown_s = float(countdown_s)
        self.clock = clock
        self.width_fn = width_fn
        self.state = "idle"
        self.pending: Optional[Dict] = None
        self.deadline = 0.0
        self.question: Optional[Grounding] = None
        self.question_action = "grasp"
        self._ids = itertools.count(1)

    # ------------------------------------------------------------ inputs
    def instruction(self, text: str, detections: List[Detection], x_now: Optional[np.ndarray] = None,
                    reach_kwargs: Optional[dict] = None) -> List[Dict]:
        q = parse(text)
        if q.action == "stop":
            return self.abort(reason="stop")
        if self.state == "clarifying" and self.question is not None and q.action == "unknown":
            g = resolve_followup(text, self.question, self._robot_xy(x_now))
            if g.status == "ok":
                return self._propose(self.question_action, g.chosen, g.message, x_now, reach_kwargs, "rules")
            return [{"type": "question", "text": g.question or "Which one?", "options": g.options}]
        if q.action == "unsupported":
            return [self._say(f"I can't {q.unsupported} plants: I have no cutter and I don't pull. {ACTIONS_HELP}")]
        if q.action == "unknown":
            return [self._say(f"I didn't understand what to do. {ACTIONS_HELP}")]
        if q.unknown_words and q.needs_target:
            return [self._say(f"I don't know the word{'s' if len(q.unknown_words) > 1 else ''} "
                              f"'{', '.join(q.unknown_words)}'. " + summarize(detections, getattr(self.proposer, "aliases", {})
                                                                              or {}))]
        if not q.needs_target:
            return self._propose(q.action, None, {"release": "Releasing the gripper.",
                                                  "scan": "Scanning for the target."}[q.action], x_now, None, "rules")
        p = self.proposer.propose(text, detections, self._robot_xy(x_now))
        g = p.grounding
        if g.status == "ambiguous":
            self.state, self.question, self.question_action = "clarifying", g, q.action
            return [{"type": "question", "text": g.question, "options": g.options}]
        if g.status != "ok" or g.chosen is None:
            self.state = "idle"
            return [self._say(g.message or "I can't find it.")]
        return self._propose(q.action, g.chosen, g.message, x_now, reach_kwargs, p.source, p.rationale)

    def confirm(self) -> List[Dict]:
        if self.state != "countdown" or self.pending is None:
            return [self._say("There is nothing to confirm.")]
        return self._commit()

    def cancel(self) -> List[Dict]:
        if self.state == "idle":
            return []
        self.state, self.pending, self.question = "idle", None, None
        return [{"type": "cancelled", "reason": "cancelled"}, self._say("Cancelled.")]

    def abort(self, reason: str = "abort") -> List[Dict]:
        ev = [{"type": "command", "name": "stop", "reason": reason}]
        if self.pending is not None:
            ev.append({"type": "cancelled", "reason": reason})
        self.state, self.pending, self.question = "idle", None, None
        ev.append(self._say("Stopped."))
        return ev

    def tick(self) -> List[Dict]:
        if self.state == "countdown" and self.clock() >= self.deadline:
            return self._commit()
        return []

    def status(self) -> Dict:
        rem = max(0.0, self.deadline - self.clock()) if self.state == "countdown" else 0.0
        return {"state": self.state, "pending": self.pending, "countdown_remaining_s": round(rem, 2),
                "question": None if self.question is None else {"text": self.question.question,
                                                                 "options": self.question.options}}

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _robot_xy(x):
        return (0.0, 0.0) if x is None else (float(x[0]), float(x[1]))

    @staticmethod
    def _say(text: str) -> Dict:
        return {"type": "say", "text": text}

    def _propose(self, action, det: Optional[Detection], message, x_now, reach_kwargs, source, rationale=""):
        verdict = None
        if det is not None and action == "grasp" and self.checker is not None and x_now is not None:
            from .grounding import DEFAULT_ALIASES, class_of  # noqa: PLC0415
            width = self.width_fn(det) if self.width_fn is not None else None
            verdict = self.checker.check(det.p_base, x_now, target_class=class_of(det.cls, DEFAULT_ALIASES) or det.cls,
                                         width_m=width, **(reach_kwargs or {}))
            if not verdict.ok:
                self.state, self.pending = "idle", None
                return [self._say(f"{message} {verdict.message}"),
                        {"type": "rejected", "code": verdict.code, "message": verdict.message,
                         "target": self._target(det)}]
        prop = {"id": next(self._ids), "action": action, "target": self._target(det) if det else None,
                "message": message, "source": source, "rationale": rationale,
                "reach": None if verdict is None else verdict.as_dict(),
                "advisories": [] if verdict is None else verdict.advisories,
                "countdown_s": self.countdown_s}
        self.state, self.pending, self.question = "countdown", prop, None
        self.deadline = self.clock() + self.countdown_s
        text = f"{message} {' '.join(prop['advisories'])} {self._verb(action)} in {self.countdown_s:.0f} s unless you cancel."
        return [{"type": "proposal", **prop}, self._say(" ".join(text.split()))]

    @staticmethod
    def _verb(action):
        return {"grasp": "Grasping", "inspect": "Looking at it", "release": "Releasing", "scan": "Scanning"}[action]

    @staticmethod
    def _target(det: Detection) -> Dict:
        return {"index": det.index, "label": det.label(), "class": det.cls, "colour": det.colour,
                "bbox": list(det.bbox), "centroid": list(det.centroid),
                "p_world": None if det.p_base is None else np.round(det.p_base, 4).tolist(),
                "polygon": det.polygon}

    def _commit(self) -> List[Dict]:
        prop = self.pending
        self.state, self.pending = "idle", None
        return [{"type": "commit", **prop}, self._say(f"{self._verb(prop['action'])}.")]
