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


def ref_json(tmp, heights=0.33, bounce=0.3, clearance=1.2):
    r = tmp / "ref.json"
    r.write_text(json.dumps({"label": "unitree", "summary": {
        str(v): {"base_height_mean_m": heights, "base_height_std_cm": bounce, "foot_clearance_cm_mean": clearance}
        for v in (0.3, 0.45, 0.9)}}))
    return r


def run_ref(tmp, gait_summary, falls, ref):
    g = write(tmp, gait_summary, falls)
    out = tmp / "gate.txt"
    S.main(["--gait", str(g), "--eval-dir", str(tmp), "--tag", "vel", "--out", str(out), "--reference-gait", str(ref)])
    return out.read_text()


def posture(v, height=0.34, bounce=0.5, clearance=1.0):
    d = good(v)
    d.update(base_height_mean_m=height, base_height_std_cm=bounce, foot_clearance_cm_mean=clearance)
    return d


def test_reference_gate_pass_and_bars(tmp_path):
    summ = {str(v): posture(v) for v in (0.2, 0.3, 0.45, 0.9, 1.0)}
    txt = run_ref(tmp_path, summ, {"vel": {10: 5, 30: 2}, "june": {10: 50, 30: 10}}, ref_json(tmp_path))
    assert "decision 15 Nov 2026" in txt and "(0.96 cm)" in txt and "GATE PASS: True" in txt
    assert "PASS  G7" in txt and "PASS  G8" in txt


def test_reference_gate_flags_tall_bouncy_and_low_feet(tmp_path):
    summ = {str(v): posture(v) for v in (0.2, 0.3, 0.45, 0.9)}
    summ["0.45"]["base_height_mean_m"] = 0.42      # straight legs: 9 cm above the reference
    summ["0.9"]["base_height_std_cm"] = 0.7        # > 2 x 0.3
    summ["0.2"]["foot_clearance_cm_mean"] = 0.9    # < 0.8 x 1.2
    txt = run_ref(tmp_path, summ, {"vel": {10: 5, 30: 2}, "june": {10: 50, 30: 10}}, ref_json(tmp_path))
    assert "FAIL  G4" in txt and "FAIL  G7" in txt and "FAIL  G8" in txt and "GATE PASS: False" in txt


def test_reference_gate_without_common_speeds_fails(tmp_path):
    summ = {str(v): posture(v) for v in (0.2, 1.0)}
    txt = run_ref(tmp_path, summ, {"vel": {10: 5, 30: 2}, "june": {10: 50, 30: 10}}, ref_json(tmp_path))
    assert "FAIL  G7" in txt and "FAIL  G8" in txt
