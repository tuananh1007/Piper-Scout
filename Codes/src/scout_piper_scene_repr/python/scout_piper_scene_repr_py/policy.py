"""Class policies from ``config/semantic_classes.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import yaml


@dataclass
class ClassPolicy:
    name: str
    label_id: int
    behavior: str                    # 'hard' | 'soft' | 'attractor'
    padding_m: float = 0.0
    cost_weight: float = 0.0
    max_penetration_m: float = 0.0
    attract_radius_m: float = 0.0
    attract_weight: float = 0.0


def load_policies(path: str) -> Dict[str, ClassPolicy]:
    """Load the class YAML. A missing ``other`` class (non-plant depth) is
    added as hard with 1 cm padding, so it is never silently ignored."""
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    pol = policies_from_dict(doc["classes"])
    pol.setdefault("other", DEFAULT_POLICIES["other"])
    return pol


def policies_from_dict(classes: dict) -> Dict[str, ClassPolicy]:
    out = {}
    for name, c in classes.items():
        if c["behavior"] not in ("hard", "soft", "attractor"):
            raise ValueError(f"{name}: unknown behavior {c['behavior']!r}")
        out[name] = ClassPolicy(name=name, **c)
    return out


DEFAULT_POLICIES = policies_from_dict({
    "stem": {"label_id": 1, "behavior": "hard", "padding_m": 0.005},
    "branch": {"label_id": 2, "behavior": "hard", "padding_m": 0.008},
    "leaf": {"label_id": 3, "behavior": "soft", "padding_m": 0.0,
             "cost_weight": 50.0, "max_penetration_m": 0.02},
    "target": {"label_id": 4, "behavior": "attractor",
               "attract_radius_m": 0.10, "attract_weight": -20.0},
    "other": {"label_id": 0, "behavior": "hard", "padding_m": 0.01},
})
