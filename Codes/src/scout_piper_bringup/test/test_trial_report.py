"""trial_report.py segmentation and per-trial numbers, trial_recorder.py throttle, without ROS."""

import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = _load(os.path.join(PKG, "scripts", "trial_report.py"), "trial_report")
rec = _load(os.path.join(PKG, "scripts", "trial_recorder.py"), "trial_recorder")


def _session():
    E = []

    def ev(t, kind, data):
        E.append({"t": t, "kind": kind, "data": data})
    ev(0.0, "state", "IDLE")
    ev(0.5, "odom", [0.0, 0.0, 0.0, 0.0, 0.0])
    # trial 1: two views, reach, servo, approach, grasp, release
    ev(1.0, "state", "SCANNING")
    ev(1.1, "instruction", "grasp the red flower")
    ev(1.5, "ap", {"u_joint": 0.99, "needs_view": True})
    ev(2.0, "state", "VIEWING")
    ev(6.0, "state", "SCANNING")
    ev(7.0, "state", "VIEWING")
    ev(10.0, "state", "SCANNING")
    ev(11.5, "ap", {"u_joint": 0.45, "needs_view": False})
    ev(12.0, "state", "REACHING")
    ev(13.0, "mpc", {"mode": "whole_body", "tcp_error_m": 0.3, "solve_ms": 40.0})
    ev(14.0, "odom", [0.3, 0.4, 0.0, 0.2, 0.0])
    ev(17.0, "mpc", {"mode": "handoff_ready", "tcp_error_m": 0.012, "approach_error_deg": 4.0, "solve_ms": 60.0})
    ev(18.0, "state", "SERVOING")
    ev(20.0, "servo", {"error_px": 30.0, "axis_miss_m": 0.01})
    ev(23.0, "servo", {"error_px": 4.2, "axis_miss_m": 0.002})
    ev(24.0, "state", "APPROACHING")
    ev(26.0, "servo", {"error_px": 3.0, "distance_m": 0.01})
    ev(27.0, "force", 1.5)
    ev(28.0, "state", "GRASPING")
    ev(28.5, "force", 4.0)
    ev(29.0, "gripper", 0.0082)
    ev(30.0, "state", "GRASPED")
    ev(31.0, "note", "firm grip")
    ev(35.0, "state", "RELEASING")
    ev(37.0, "state", "RETREATING")
    ev(40.0, "state", "IDLE")
    ev(41.0, "force", 9.0)                       # between trials: belongs to none
    # trial 2: aborted by the safety supervisor during the reach
    ev(50.0, "state", "SCANNING")
    ev(52.0, "state", "REACHING")
    ev(53.0, "safety", {"ok": False, "tripped": "operator heartbeat lost (last 1.0 s ago) while moving"})
    ev(53.1, "state", "ABORTED")
    # trial 3: rescan, the log ends while servoing
    ev(60.0, "state", "SCANNING")
    ev(61.0, "state", "REACHING")
    ev(64.0, "state", "SERVOING")
    ev(65.0, "state", "SCANNING")               # re-grounding stays in the trial
    ev(66.0, "state", "SERVOING")
    return E


def test_segments_trials_and_summarises(tmp_path):
    path = tmp_path / "s.jsonl"
    with open(path, "w") as f:
        for e in _session():
            f.write(json.dumps(e) + "\n")
        f.write('{"t": 70.0, "kind": "sta')          # cut off by a crash: skipped
    trials = rep.segment(rep.load([str(path)]))
    assert len(trials) == 3
    r1, r2, r3 = (rep.summarize(t) for t in trials)

    assert r1["outcome"] == "released" and r1["views"] == 2 and r1["duration_s"] == 39.0
    assert r1["u_joint"] == [0.99, 0.45]
    assert r1["time_in_s"]["VIEWING"] == 7.0 and r1["time_in_s"]["SCANNING"] == 4.0
    assert r1["reach"] == {"time_s": 6.0, "tcp_error_m": 0.012, "orientation_error_deg": 4.0,
                           "solve_ms_median": 50.0, "solve_ms_max": 60.0}
    assert r1["servo"]["error_px"] == 4.2 and r1["servo"]["time_s"] == 6.0
    assert r1["approach"]["time_s"] == 4.0 and r1["force_max_n"] == 4.0
    assert r1["gripper_at_grasp_mm"] == 8.2 and r1["base_travel_m"] == 0.0     # one odom sample in the trial
    assert r1["instructions"] == ["grasp the red flower"] and r1["notes"] == ["firm grip"]

    assert r2["outcome"].startswith("aborted (safety: operator heartbeat lost")
    assert r2["safety_trips"] and r2["views"] == 0
    assert r3["outcome"] == "incomplete" and r3["states"].count("SERVOING") == 2
    assert r3["servo"]["time_s"] == 1.0           # 64-65 s; the second SERVOING starts at the log's end (66 s)

    agg = rep.aggregate([r1, r2, r3])
    assert agg == {"trials": 3, "grasped": 1, "success_rate": 0.333, "aborted": 1, "median_reach_s": 3.0,
                   "median_success_duration_s": 39.0, "safety_trips": 1}
    md = rep.markdown([r1, r2, r3], agg)
    assert "| 1 | released | 39.0 | 2 | 0.99 → 0.45 | 6.0 | 1.2 |" in md and "1 of 3 grasped (33 %)" in md


def test_cli_writes_json_and_markdown(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in _session()) + "\n")
    out_j, out_m = tmp_path / "r.json", tmp_path / "r.md"
    assert rep.main([str(path), "--json", str(out_j), "--md", str(out_m)]) == 0
    assert json.loads(out_j.read_text())["aggregate"]["trials"] == 3
    assert out_m.read_text().startswith("| # | outcome")


def test_approach_force_and_phase_2b_gate(tmp_path):
    r1, r2, r3 = (rep.summarize(t) for t in rep.segment(_session()))
    assert r1["force_max_approach_n"] == 1.5      # SERVOING to GRASPING: the 4.0 N while closing is left out
    assert "force_max_approach_n" not in r2 and r3["force_max_approach_n"] is None   # no SERVOING / no force

    def row(controller, outcome, force):
        return {"outcome": outcome, "servo": {"controller": controller}, "force_max_approach_n": force}
    rows = ([row("ibvs", "released", 2.0)] * 5 + [row("ibvs", "aborted (in APPROACHING)", 3.0)] * 5
            + [row("mppi", "released", 1.0)] * 8 + [row("mppi", "aborted (in SERVOING)", None)] * 2)
    per = rep.by_controller(rows)
    assert per["ibvs"] == {"trials": 10, "grasped": 5, "success_rate": 0.5, "median_force_approach_n": 2.5,
                           "force_samples": 10}
    assert per["mppi"]["success_rate"] == 0.8 and per["mppi"]["median_force_approach_n"] == 1.0
    assert rep.gate(per) == {"success_gain_pp": 30.0, "force_reduction": 0.6, "success_ok": True, "force_ok": True,
                             "passed": True}
    per["mppi"]["success_rate"] = 0.6                 # +10 points only
    assert rep.gate(per)["passed"] is False
    assert rep.gate({"ibvs": per["ibvs"]})["passed"] is None
    judged = rep.summarize(rep.segment(_session())[0] + [{"t": 32.0, "kind": "note", "data": "FAIL stem pushed over"}])
    assert judged["outcome"] == "failed (FAIL stem pushed over)"

    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in _session()) + "\n")
    out_j = tmp_path / "r.json"
    assert rep.main([str(path), "--gate", "--json", str(out_j)]) == 0
    agg = json.loads(out_j.read_text())["aggregate"]
    assert agg["controllers"]["ibvs"]["trials"] == 1 and agg["controllers"]["unknown"]["trials"] == 1
    assert agg["gate"]["passed"] is None          # no MPPI trials here


def test_recorder_throttle():
    th = rec.Throttle(0.2)
    assert th(0.0, "a") and not th(0.1, "a")
    assert th(0.15, "b")                          # a change passes at once
    assert th(0.4, "b") and not th(0.5, "b")
