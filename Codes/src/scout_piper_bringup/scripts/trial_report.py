#!/usr/bin/env python3
"""Per-trial summary of a robot session recorded by trial_recorder.py.

    ros2 run scout_piper_bringup trial_report.py session.jsonl [more.jsonl ...] [--json out.json] [--md out.md]

A trial starts when the stem_grasp pipeline enters SCANNING from IDLE,
ABORTED or the start of the log, and ends at the next IDLE or ABORTED (or the
end of the log); a re-grounding (SERVOING -> SCANNING) stays in the trial.
Per trial: outcome (grasped, released, aborted with the safety trip or the
last servo / MPC status, incomplete), the time in each state, the views asked
for and the stem-joint uncertainty before / after them, the reach (time,
final TCP and orientation error, MPPI solve times), the servo (time, final
pixel error and axis miss), the approach, the peak contact force, the
gripper opening when GRASPED, the base travel, safety trips, the operator's
instructions and notes (``/trial/note``, e.g. "slipped"). Prints a Markdown
table for RESULTS.md and the success rate and median times over all trials.

ROS-free (reads the JSONL only).
"""

import argparse
import json
import math
import statistics
import sys
from typing import Dict, Iterable, List, Optional

ACTIVE_ORDER = ("SCANNING", "VIEWING", "REACHING", "SERVOING", "APPROACHING", "GRASPING",
                "GRASPED", "AT_GRASP", "RELEASING", "RETREATING")


def load(paths: Iterable[str]) -> List[Dict]:
    events = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue                      # a line cut off by a crash
    events.sort(key=lambda e: e.get("t", 0.0))
    return events


def segment(events: List[Dict]) -> List[List[Dict]]:
    """Split the events into trials (lists of events, state changes included)."""
    trials: List[List[Dict]] = []
    cur: Optional[List[Dict]] = None
    for e in events:
        if e.get("kind") == "state":
            s = e.get("data")
            if s == "SCANNING" and cur is None:
                cur = []
                trials.append(cur)
            if cur is not None:
                cur.append(e)
                if s in ("IDLE", "ABORTED"):
                    cur = None
            continue
        if cur is not None:
            cur.append(e)
    return trials


def _last(events, kind, before=None, after=None):
    out = None
    for e in events:
        if e.get("kind") != kind:
            continue
        if before is not None and e["t"] > before:
            break
        if after is not None and e["t"] < after:
            continue
        out = e
    return out


def _intervals(states: List[Dict], end_t: float) -> Dict[str, List[List[float]]]:
    iv: Dict[str, List[List[float]]] = {}
    for a, b in zip(states, states[1:] + [{"t": end_t, "data": None}]):
        iv.setdefault(a["data"], []).append([a["t"], b["t"]])
    return iv


def _r(v, nd=3):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def summarize(trial: List[Dict]) -> Dict:
    states = [e for e in trial if e.get("kind") == "state"]
    t0 = states[0]["t"] if states else trial[0]["t"]
    t1 = trial[-1]["t"]
    seq = [s["data"] for s in states]
    iv = _intervals(states, t1)
    time_in = {k: _r(sum(b - a for a, b in v), 2) for k, v in iv.items() if k not in ("IDLE", "ABORTED")}
    trips = [e["data"].get("tripped") for e in trial if e.get("kind") == "safety"
             and isinstance(e.get("data"), dict) and e["data"].get("ok") is False]
    trips = list(dict.fromkeys(t for t in trips if t))

    if "RELEASING" in seq or ("RETREATING" in seq and "GRASPED" in seq):
        outcome = "released" if seq[-1] == "IDLE" else "grasped"
    elif "GRASPED" in seq:
        outcome = "grasped"
    elif seq and seq[-1] == "ABORTED":
        last_active = next((s for s in reversed(seq[:-1]) if s != "ABORTED"), None)
        why = f"safety: {trips[-1]}" if trips else f"in {last_active}"
        outcome = f"aborted ({why})"
    elif "AT_GRASP" in seq:
        outcome = "at grasp"
    else:
        outcome = "incomplete"

    out: Dict = {"start_t": _r(t0, 3), "duration_s": _r(t1 - t0, 2), "outcome": outcome,
                 "states": " -> ".join(seq), "time_in_s": time_in, "views": seq.count("VIEWING"),
                 "safety_trips": trips}

    # active perception: stem-joint uncertainty before the first view and at the reach
    ap = [e for e in trial if e.get("kind") == "ap" and isinstance(e.get("data"), dict)]
    if ap:
        first_v = iv.get("VIEWING", [[None]])[0][0]
        reach_t = iv.get("REACHING", [[None]])[0][0]
        before = _last(ap, "ap", before=first_v) if first_v else ap[0]
        at = _last(ap, "ap", before=reach_t) if reach_t else ap[-1]
        out["u_joint"] = [_r((before or ap[0])["data"].get("u_joint")), _r((at or ap[-1])["data"].get("u_joint"))]

    # reach: the last MPC status inside the last REACHING interval
    if "REACHING" in iv:
        a, b = iv["REACHING"][-1]
        m = _last(trial, "mpc", before=b, after=a)
        solves = [e["data"].get("solve_ms") for e in trial if e.get("kind") == "mpc"
                  and a <= e["t"] <= b and isinstance(e.get("data"), dict) and e["data"].get("solve_ms") is not None]
        out["reach"] = {"time_s": _r(b - a, 2),
                        "tcp_error_m": _r(m["data"].get("tcp_error_m")) if m else None,
                        "orientation_error_deg": _r(m["data"].get("approach_error_deg"), 1) if m else None,
                        "solve_ms_median": _r(statistics.median(solves), 1) if solves else None,
                        "solve_ms_max": _r(max(solves), 1) if solves else None}
    # servo: the last status inside SERVOING
    if "SERVOING" in iv:
        a, b = iv["SERVOING"][-1]
        sv = _last(trial, "servo", before=b, after=a)
        out["servo"] = {"time_s": _r(sum(y - x for x, y in iv["SERVOING"]), 2),
                        "error_px": _r(sv["data"].get("error_px"), 1) if sv else None,
                        "axis_miss_m": _r(sv["data"].get("axis_miss_m")) if sv else None,
                        "controller": (sv["data"].get("controller") or "ibvs") if sv else None}
    if "APPROACHING" in iv:
        a, b = iv["APPROACHING"][-1]
        sv = _last(trial, "servo", before=b, after=a)
        out["approach"] = {"time_s": _r(b - a, 2), "distance_m": _r(sv["data"].get("distance_m")) if sv else None}

    forces = [e["data"] for e in trial if e.get("kind") == "force" and isinstance(e.get("data"), (int, float))]
    out["force_max_n"] = _r(max(forces), 2) if forces else None
    if "GRASPED" in iv:
        g = _last(trial, "gripper", before=iv["GRASPED"][0][0] + 0.5)
        out["gripper_at_grasp_mm"] = _r(1000 * g["data"], 1) if g else None
    odo = [e["data"] for e in trial if e.get("kind") == "odom" and isinstance(e.get("data"), list)]
    out["base_travel_m"] = _r(sum(math.dist(p[:2], q[:2]) for p, q in zip(odo, odo[1:])), 3) if len(odo) > 1 else 0.0
    out["instructions"] = [e["data"] for e in trial if e.get("kind") == "instruction"]
    out["notes"] = [e["data"] for e in trial if e.get("kind") == "note"]
    return out


def aggregate(rows: List[Dict]) -> Dict:
    n = len(rows)
    ok = [r for r in rows if r["outcome"] in ("grasped", "released")]
    reach = [r["reach"]["time_s"] for r in rows if r.get("reach") and r["reach"]["time_s"] is not None]
    dur = [r["duration_s"] for r in ok]
    return {"trials": n, "grasped": len(ok), "success_rate": _r(len(ok) / n, 3) if n else None,
            "aborted": sum(r["outcome"].startswith("aborted") for r in rows),
            "median_reach_s": _r(statistics.median(reach), 2) if reach else None,
            "median_success_duration_s": _r(statistics.median(dur), 2) if dur else None,
            "safety_trips": sum(len(r["safety_trips"]) for r in rows)}


def markdown(rows: List[Dict], agg: Dict) -> str:
    head = ("| # | outcome | s | views | u_joint | reach s | TCP err cm | servo s | err px | "
            "approach s | F max N | grip mm | base m | notes |")
    lines = [head, "|" + "---|" * 14]
    for i, r in enumerate(rows, 1):
        re, sv, ap = r.get("reach") or {}, r.get("servo") or {}, r.get("approach") or {}
        u = r.get("u_joint")
        cells = [str(i), r["outcome"], f"{r['duration_s']}", str(r["views"]),
                 f"{u[0]} → {u[1]}" if u else "",
                 str(re.get("time_s", "")), "" if re.get("tcp_error_m") is None else f"{100 * re['tcp_error_m']:.1f}",
                 str(sv.get("time_s", "")), "" if sv.get("error_px") is None else str(sv["error_px"]),
                 str(ap.get("time_s", "")), "" if r.get("force_max_n") is None else str(r["force_max_n"]),
                 "" if r.get("gripper_at_grasp_mm") is None else str(r["gripper_at_grasp_mm"]),
                 str(r.get("base_travel_m", "")),
                 "; ".join(r["instructions"] + r["notes"] + r["safety_trips"]).replace("|", "/")]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"{agg['grasped']} of {agg['trials']} grasped"
                 + (f" ({100 * agg['success_rate']:.0f} %)" if agg["success_rate"] is not None else "")
                 + f", {agg['aborted']} aborted, {agg['safety_trips']} safety trips; median reach "
                 f"{agg['median_reach_s']} s, median successful trial {agg['median_success_duration_s']} s.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", help="trial_recorder.py JSONL files")
    ap.add_argument("--json", help="write the per-trial summaries and the aggregate here")
    ap.add_argument("--md", help="write the Markdown table here")
    a = ap.parse_args(argv)
    rows = [summarize(t) for t in segment(load(a.logs)) if t]
    agg = aggregate(rows)
    md = markdown(rows, agg)
    print(md)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"trials": rows, "aggregate": agg}, f, indent=1)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
