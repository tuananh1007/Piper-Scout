"""Instruction -> structured target query (P5.1.3, rule-based).

ROADMAP.md Phase 5 maps an instruction ℓ to
q_ℓ = (class, ordinal, spatial relation, attribute, action). ``parse`` does
that with a small grammar over a fixed vocabulary, so that every field it
fills can be explained back to the operator and nothing it does not
understand is guessed:

  action      grasp ("grasp", "grab", "pick", "take", "hold", "get"), inspect
              ("look at", "inspect", "check", "show", "examine", "find"),
              release ("release", "let go", "drop", "open"), stop ("stop",
              "halt", "abort", "freeze", "wait"), scan ("scan", "look
              around", "search"); cutting, pruning and harvesting are
              recognised and refused (no cutter on the gripper)
  class       flower, stem, branch, leaf, fruit, bud (with synonyms; "one",
              "it" keep the class open)
  ordinal     first .. tenth, 1st .. 10th, "number 2", "#2", last, with the
              counting direction "from the left/right/top/bottom/front/back"
              (default left to right, as the operator sees the camera image)
  superlative leftmost, rightmost, highest/tallest/top, lowest/bottom,
              nearest/closest, farthest, biggest/largest, smallest; "the left
              one" reads as leftmost
  attribute   colours, sizes (big, small), open / closed, ripe / unripe
  relation    left of, right of, above, below, behind, in front of, next
              to / beside / near, with a reference phrase parsed the same way
              ("the flower to the left of the yellow one")

Words outside the vocabulary are kept in ``unknown_words``; the instruction
node asks back instead of acting on a query it only half understood.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

ACTIONS = {
    "grasp": ["pick up", "grasp", "grab", "pick", "take", "hold", "get", "seize", "catch"],
    "inspect": ["look at", "inspect", "check", "show me", "show", "examine", "find", "view", "observe"],
    "release": ["let go", "release", "drop", "open the gripper", "open"],
    "stop": ["stop", "halt", "abort", "freeze", "wait", "emergency"],
    "scan": ["look around", "scan", "search"],
}
UNSUPPORTED = {"cut": ["cut", "prune", "snip", "trim", "harvest", "chop", "slice"],
               "pull": ["pull", "yank", "tear", "rip"], "move": ["push", "bend", "shake"]}
CLASSES = {
    "flower": ["flowers", "flower", "blossoms", "blossom", "blooms", "bloom", "petals"],
    "stem": ["stems", "stem", "stalks", "stalk", "peduncles", "peduncle", "pedicel"],
    "branch": ["branches", "branch", "twigs", "twig", "shoots", "shoot"],
    "leaf": ["leaves", "leaf", "foliage"],
    "fruit": ["fruits", "fruit", "tomatoes", "tomato", "strawberries", "strawberry", "peppers", "pepper",
              "berries", "berry", "apples", "apple"],
    "bud": ["buds", "bud"],
}
GENERIC = ["one", "ones", "it", "that", "this", "thing", "object", "plant"]
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "white", "black", "grey", "gray"]
ATTRS = {"big": "big", "large": "big", "huge": "big", "small": "small", "little": "small", "tiny": "small",
         "open": "open", "opened": "open", "closed": "closed", "ripe": "ripe", "unripe": "unripe",
         "green-ish": "green", "violet": "purple", "gray": "grey"}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
            "eighth": 8, "ninth": 9, "tenth": 10, "last": -1}
SUPERLATIVES = {
    "leftmost": "left", "rightmost": "right", "highest": "top", "tallest": "top", "topmost": "top",
    "uppermost": "top", "lowest": "bottom", "bottommost": "bottom", "nearest": "near", "closest": "near",
    "farthest": "far", "furthest": "far", "biggest": "big", "largest": "big", "smallest": "small",
}
SIDE_WORDS = {"left": "left", "right": "right", "top": "top", "upper": "top", "bottom": "bottom", "lower": "bottom",
              "middle": "middle", "center": "middle", "centre": "middle", "front": "near", "back": "far"}
RELATIONS = [  # longest first
    ("to the left of", "left_of"), ("to the right of", "right_of"), ("on the left of", "left_of"),
    ("on the right of", "right_of"), ("left of", "left_of"), ("right of", "right_of"),
    ("in front of", "in_front_of"), ("behind", "behind"), ("above", "above"), ("over", "above"),
    ("below", "below"), ("under", "below"), ("underneath", "below"), ("beneath", "below"),
    ("next to", "next_to"), ("beside", "next_to"), ("near", "next_to"), ("close to", "next_to"),
    ("by", "next_to"), ("between", "between"),
]
FILLER = {"the", "a", "an", "please", "can", "you", "could", "would", "robot", "now", "and", "then", "to",
          "of", "from", "on", "in", "at", "me", "i", "want", "need", "with", "that's", "which", "is", "its",
          "for", "up", "go", "just", "only", "one's", "number", "#", "us", "let's", "lets", "try", "kindly",
          "side", "row", "image", "picture", "screen", "camera", "view", "there", "here", "carefully",
          "gently", "slowly", "very", "most", "more"}


@dataclass
class Phrase:
    cls: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    ordinal: Optional[int] = None
    order: str = "left"                     # direction the ordinal counts from
    superlative: Optional[str] = None
    plural: bool = False

    def describe(self) -> str:
        words = []
        if self.ordinal is not None:
            words.append("last" if self.ordinal == -1 else _ordinal_word(self.ordinal))
        if self.superlative:
            words.append({"left": "leftmost", "right": "rightmost", "top": "highest", "bottom": "lowest",
                          "near": "nearest", "far": "farthest", "big": "biggest", "small": "smallest",
                          "middle": "middle"}[self.superlative])
        words += self.attributes
        words.append(self.cls or "one")
        out = "the " + " ".join(words)
        if self.ordinal is not None and self.order != "left":
            out += f" from the {_order_word(self.order)}"
        return out


@dataclass
class Query:
    text: str
    action: str = "unknown"                 # grasp | inspect | release | stop | scan | unsupported | unknown
    unsupported: Optional[str] = None       # cut | pull | move
    target: Phrase = field(default_factory=Phrase)
    relation: Optional[str] = None
    reference: Optional[Phrase] = None
    unknown_words: List[str] = field(default_factory=list)

    @property
    def needs_target(self) -> bool:
        return self.action in ("grasp", "inspect")

    def describe(self) -> str:
        s = self.target.describe()
        if self.relation and self.reference is not None:
            s += " " + self.relation.replace("_", " ") + " " + self.reference.describe()
        return s


def _ordinal_word(k: int) -> str:
    inv = {v: w for w, v in ORDINALS.items()}
    return inv.get(k, f"#{k}")


def _order_word(order: str) -> str:
    return {"near": "front", "far": "back"}.get(order, order)


def _tokens(text: str) -> List[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"#\1", t)
    t = re.sub(r"\bnumber\s+(\d+)", r"#\1", t)
    t = re.sub(r"#\s+(\d+)", r"#\1", t)
    t = re.sub(r"[^a-z0-9#'\- ]+", " ", t)
    return t.split()


def _match_phrase(tokens: List[str], i: int, phrase: str) -> bool:
    p = phrase.split()
    return tokens[i:i + len(p)] == p


def _find_action(tokens: List[str]):
    """(action, unsupported, index after the verb)."""
    for i in range(len(tokens)):
        for name, verbs in UNSUPPORTED.items():
            for v in verbs:
                if _match_phrase(tokens, i, v):
                    return "unsupported", name, i + len(v.split())
        for name, verbs in ACTIONS.items():
            for v in sorted(verbs, key=lambda x: -len(x.split())):
                if _match_phrase(tokens, i, v):
                    return name, None, i + len(v.split())
    return "unknown", None, 0


def _parse_phrase(tokens: List[str], unknown: List[str]) -> Phrase:
    ph = Phrase()
    i = 0
    while i < len(tokens):
        w = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if w in ("from", "counting") and nxt in ("the",) and i + 2 < len(tokens) and tokens[i + 2] in SIDE_WORDS:
            ph.order = SIDE_WORDS[tokens[i + 2]]
            i += 3
            continue
        if w == "from" and nxt in SIDE_WORDS:
            ph.order = SIDE_WORDS[nxt]
            i += 2
            continue
        if w in ORDINALS:
            ph.ordinal = ORDINALS[w]
        elif w.startswith("#") and w[1:].isdigit():
            ph.ordinal = int(w[1:])
        elif w.isdigit() and ph.cls is None and not ph.attributes:
            ph.ordinal = int(w)
        elif w in SUPERLATIVES:
            ph.superlative = SUPERLATIVES[w]
        elif w in COLOURS:
            ph.attributes.append(ATTRS.get(w, w))
        elif w in ATTRS:
            ph.attributes.append(ATTRS[w])
        elif w in SIDE_WORDS and nxt not in ("of",):
            # "the left flower", "the one on the right", "the middle one"
            if ph.superlative is None:
                ph.superlative = SIDE_WORDS[w]
        elif any(w in syn for syn in CLASSES.values()):
            ph.cls = next(c for c, syn in CLASSES.items() if w in syn)
            ph.plural = w.endswith("s") and w not in ("ones",) and not w.endswith("ss")
        elif w in GENERIC:
            pass
        elif w in FILLER or w.endswith("ish"):
            pass
        else:
            unknown.append(w)
        i += 1
    # "the last" with a counting direction makes the ordinal a superlative-like order
    return ph


def parse(text: str) -> Query:
    tokens = _tokens(text)
    q = Query(text=text)
    q.action, q.unsupported, start = _find_action(tokens)
    rest = tokens[start:] if q.action != "unknown" else tokens
    # split at the first relation phrase
    rel_at, rel_len = None, 0
    for i in range(len(rest)):
        for phrase, name in RELATIONS:
            if _match_phrase(rest, i, phrase):
                # "on the left" / "the left one" are side words, not relations, unless followed by "of"
                q.relation = name
                rel_at, rel_len = i, len(phrase.split())
                break
        if rel_at is not None:
            break
    unknown: List[str] = []
    if rel_at is None:
        q.target = _parse_phrase(rest, unknown)
    else:
        q.target = _parse_phrase(rest[:rel_at], unknown)
        q.reference = _parse_phrase(rest[rel_at + rel_len:], unknown)
        if q.reference.cls is None and q.target.cls is not None and not q.reference.attributes \
                and q.reference.superlative is None and q.reference.ordinal is None:
            q.reference = None                                   # "near" used as a plain word
            q.relation = None
        elif q.reference.cls is None:
            q.reference.cls = q.target.cls                       # "the flower left of the yellow one"
    q.unknown_words = unknown
    return q
