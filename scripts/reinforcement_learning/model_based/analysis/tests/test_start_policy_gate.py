"""CPU tests for start_policy_gate.py."""
import json
from pathlib import Path
import sys

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
import start_policy_gate as S  # noqa: E402


def good(v):
    return {"falls": 0, "plan_fall_events": 0, "low_body_fraction_lt_0p25m": 0.0, "speed_ratio": 0.97,
            "foot_clearance_cm_mean": 4.0, "pitch_std_deg": 0.4, "trot_index": 0.86}


def write(tmp, gait_summary, falls):
    g = tmp / "gait.json"
    g.write_text(json.dumps({"label": "x", "checkpoint": "c", "summary": gait_summary}))
    for tag, per_cmd in falls.items():
        for cmd, f in per_cmd.items():
            for seed in (42, 43):
                d = tmp / f"{tag}_cmd{cmd}_s{seed}"
                d.mkdir(parents=True)
                (d / "eval.log").write_text(f"noise\n===== EPISODE STATS =====\n  falls: {f}\n  episodes_completed: 300\n")
    return g


def run(tmp, gait_summary, falls):
    g = write(tmp, gait_summary, falls)
    out = tmp / "gate.txt"
    S.main(["--gait", str(g), "--eval-dir", str(tmp), "--tag", "vel", "--out", str(out)])
    return out.read_text()


def test_pass(tmp_path):
    summ = {str(v): good(v) for v in (0.1, 0.2, 0.45, 1.0)}
    summ["0.1"]["speed_ratio"] = 0.2           # below 0.2 m/s: speed ratio not scored
    txt = run(tmp_path, summ, {"vel": {10: 5, 30: 2}, "june": {10: 50, 30: 10}})
    assert "GATE PASS: True" in txt


def test_fail_low_body_and_june(tmp_path):
    summ = {str(v): good(v) for v in (0.2, 0.45)}
    summ["0.2"]["low_body_fraction_lt_0p25m"] = 0.3
    txt = run(tmp_path, summ, {"vel": {10: 60, 30: 2}, "june": {10: 50, 30: 10}})
    assert "FAIL  G1" in txt and "FAIL  G6" in txt and "GATE PASS: False" in txt


def test_missing_june_eval_fails_g6(tmp_path):
    summ = {str(v): good(v) for v in (0.2, 0.45)}
    txt = run(tmp_path, summ, {"vel": {10: 1, 30: 1}})
    assert "FAIL  G6" in txt
