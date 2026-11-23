"""E1-R2 readout: cells written as eval_episode_stats.py writes them (EpisodeStats summary inside the same wrapper),
known answers for the pooled counts, the per-seed comparison and every reading branch."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import e1r2_score as sc  # noqa: E402
from go2_g0_gate import ProtocolMismatch  # noqa: E402


def real_cell(folder, falls, seed, speed=1.0, std=0.5, push=True):
    """One evaluation cell with `falls` falls; summary dict from EpisodeStats, wrapper keys as the evaluator writes."""
    from episode_stats import EpisodeStats
    n, steps = 2, 500
    st = EpisodeStats(n, 0.02, 20.0, std=std)
    g = np.random.default_rng(seed)
    fall_rows = {20 + 10 * i for i in range(falls)}
    for t in range(steps):
        cmd = np.array([[1.0, 0.0], [0.8, 0.0]])
        act = speed * cmd + g.normal(0, 0.02, (n, 2))
        done = np.array([t in fall_rows, t == steps - 1])
        st.step(cmd, act, done, np.array([False, t == steps - 1]))
    st.finish()
    folder.mkdir(parents=True)
    with open(folder / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(st.episodes[0].keys()))
        w.writeheader()
        for e in st.episodes:
            w.writerow(e)
    with open(folder / "episodes_summary.json", "w") as f:
        json.dump({"checkpoint": "x.pt", "task": "Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0", "num_envs": n,
                   "num_steps": steps, "no_push": not push, "cmd_resample_s": 10.0, "friction_shift": None, "seed": seed,
                   "std": std, "step_dt": 0.02, "episode_length_s": 20.0, "command_resampling_time_range_s": [10.0, 10.0],
                   "push_robot": {"enabled": push}, "summary": st.summary(), "old_logs": {}}, f, indent=1)


def arm(root, tag, falls10, falls30, **kw):
    for cmd, n in ((10, falls10), (30, falls30)):
        for s in (42, 43):
            real_cell(root / f"{tag}_cmd{cmd}_s{s}", n, s, **kw)
    return f"{tag}={root}:{tag}"


def run(tmp_path, specs, expect=None):
    args = [x for s in specs for x in ("--arm", s)] + [x for e in (expect or []) for x in ("--expect", e)]
    sc.main(args + ["--out", str(tmp_path / "out")])
    return json.loads((tmp_path / "out" / "e1r2_score.json").read_text())["result"], \
        (tmp_path / "out" / "e1r2_score.txt").read_text()


def test_same_direction_and_known_counts(tmp_path):
    e = tmp_path / "eval"
    specs = [arm(e, "B1", 5, 2), arm(e, "D1", 1, 1), arm(e, "B2", 4, 3), arm(e, "D2", 2, 1), arm(e, "B3", 6, 2),
             arm(e, "D3", 3, 2)]
    r, t = run(tmp_path, specs, ["B1:10=10", "D3:30=4"])
    c = r["by_cmd"]["10"]
    assert [p["b_falls"] for p in c["per_seed"]] == [10, 8, 12] and [p["d_falls"] for p in c["per_seed"]] == [2, 4, 6]
    assert c["d_below"] == 3 and r["reading"] == "SAME DIRECTION AS E1"
    assert r["by_cmd"]["30"]["d_below"] == 2                       # seed 3 at 30 s: 4 vs 4, not below
    assert "CROSSCHECK: PASS" in t
    rate = 10 / 12.8                                               # 10 falls over 2 x 32 x 10000 x 0.02 s
    assert abs(c["per_seed"][0]["b_rate"] - rate) < 1e-9


def test_opposite_mixed_and_not_readable(tmp_path):
    e = tmp_path / "a"
    r, _ = run(tmp_path / "1", [arm(e, "B1", 1, 1), arm(e, "D1", 3, 1), arm(e, "B2", 1, 1), arm(e, "D2", 4, 1),
                                arm(e, "B3", 5, 1), arm(e, "D3", 1, 1)])
    assert r["reading"] == "OPPOSITE"
    e = tmp_path / "b"
    r, _ = run(tmp_path / "2", [arm(e, "B1", 3, 1), arm(e, "D1", 1, 1), arm(e, "B2", 3, 1), arm(e, "D2", 3, 1),
                                arm(e, "B3", 3, 1), arm(e, "D3", 4, 1)])
    assert r["reading"] == "MIXED"
    e = tmp_path / "c"
    r, _ = run(tmp_path / "3", [arm(e, "B1", 3, 1), arm(e, "D1", 1, 1, speed=0.3)])
    assert r["reading"].startswith("NOT READABLE")


def test_tracking_margin_blocks_same_direction(tmp_path):
    e = tmp_path / "e"
    r, _ = run(tmp_path, [arm(e, "B1", 3, 1), arm(e, "D1", 1, 1, speed=0.85)])
    p = r["by_cmd"]["10"]["per_seed"][0]
    assert p["d_below"] and not p["track_ok"] and r["reading"] == "MIXED"


def test_crosscheck_fail_and_protocol_mismatch(tmp_path):
    e = tmp_path / "e"
    specs = [arm(e, "B2", 3, 1), arm(e, "D2", 1, 1)]
    _, t = run(tmp_path / "x", specs, ["B2:10=7"])
    assert "CROSSCHECK: FAIL" in t and "DIFFERENT" in t
    f = tmp_path / "f"
    specs = [arm(f, "B1", 3, 1), arm(f, "D1", 1, 1, push=False)]
    with pytest.raises(ProtocolMismatch):
        run(tmp_path / "y", specs)


def test_bad_arm_spec(tmp_path):
    with pytest.raises(SystemExit):
        sc.parse_arms(["X1=a:b"])
