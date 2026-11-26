"""A4 D8 scorer: cell validation (protocol, completeness, exposure), nondefault budgets, readout rules, and the
historical scorers left byte-identical."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import go2_d8_score as sc  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

FIELDS = ["env", "start", "length", "counted_steps", "fall", "censored", "mean_speed_act", "steps_since_cmd_change",
          "cmd_speed_end"]


def episodes_for(envs, steps, n_falls, startup_falls=0, seed=0):
    """Per env: episodes that tile `steps` exactly; the first n_falls episodes (round robin) end in falls,
    `startup_falls` of them within 30 rows; the last episode of each env is censored."""
    rng = np.random.default_rng(seed)
    eps, made, early = [], 0, 0
    for e in range(envs):
        t = 0
        while t < steps:
            if early < startup_falls and made < n_falls:
                L, fall = 30, True
                early += 1; made += 1
            elif made < n_falls and rng.random() < 0.5:
                L, fall = int(rng.integers(100, 400)), True
                made += 1
            else:
                L, fall = 1000, False
            L = min(L, steps - t)
            censored = t + L >= steps and not (fall and t + L == steps)
            eps.append((e, t, L, fall and not censored, censored))
            t += L
    return eps


def write_cell(folder, envs=4, steps=2000, n_falls=5, startup_falls=0, seed=42, cmd=10, friction=0.3, speed=0.9,
               track=0.85, exposure="auto", task=sc.TASK, push=False, drop_env_rows=0, seed_data=0):
    folder.mkdir(parents=True)
    eps = episodes_for(envs, steps, n_falls, startup_falls, seed_data)
    if drop_env_rows:
        eps = eps[:-drop_env_rows]
    counted = 0
    with open(folder / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for env, start, L, fall, cens in eps:
            w.writerow(dict(env=env, start=start, length=L, counted_steps=L - 1, fall=fall, censored=cens,
                            mean_speed_act=1.0, steps_since_cmd_change=L, cmd_speed_end=1.0))
            counted += L - 1
    falls = sum(1 for e in eps if e[3])
    inner = dict(falls=falls, counted_steps=counted, speed_act_per_step=1.0, speed_ratio=speed, track_per_step=track)
    if exposure == "auto":
        inner["exposure_s"] = envs * steps * 0.02
    elif exposure is not None:
        inner["exposure_s"] = exposure
    (folder / "episodes_summary.json").write_text(json.dumps(dict(
        task=task, num_envs=envs, num_steps=steps, seed=seed, std=0.5, cmd_resample_s=float(cmd), step_dt=0.02,
        episode_length_s=20.0, push_robot=dict(enabled=push), friction_shift=dict(factor=friction, verified=True),
        summary=inner)))
    return falls


def make_all(tmp_path, falls=None, envs=4, steps=2000, cmd=10, **kw):
    falls = falls or dict(F8=6, U8=3, D7=6, S2=4)
    new, ref = tmp_path / "new", tmp_path / "ref"
    got = {}
    for t, n in falls.items():
        for s in sc.SEEDS:
            got[(t, s)] = write_cell((new if t in ("F8", "U8") else ref) / f"{t}_cmd{cmd}_s{s}", envs=envs, steps=steps,
                                     n_falls=n, seed=s, cmd=cmd, seed_data=hash((t, s)) % 1000, **kw.get(t, {}))
    return new, ref, got


def test_valid_cells_pool_and_count(tmp_path):
    new, ref, got = make_all(tmp_path, envs=4, steps=2000)
    p = sc.score(new, ref, 0.3, envs=4, steps=2000)
    for t in sc.POLICIES:
        assert p[t]["falls"] == got[(t, 42)] + got[(t, 43)]
        assert p[t]["exposure_s"] == pytest.approx(2 * 4 * 2000 * 0.02)
        assert all(c["exposure_source"].startswith("summary") for c in p[t]["per_seed"])
        assert p[t]["startup"]["first_second"]["rows"] == 50


def test_nondefault_budget_is_refused_unless_declared(tmp_path):
    new, ref, _ = make_all(tmp_path, envs=32, steps=5000)          # half the G6 budget
    with pytest.raises(sc.CellInvalid, match="num_steps"):
        sc.score(new, ref, 0.3)                                     # default expectation 32 x 10,000
    p = sc.score(new, ref, 0.3, envs=32, steps=5000)
    assert p["F8"]["exposure_s"] == pytest.approx(2 * 32 * 5000 * 0.02)   # 6,400 s per policy, not 12,800
    r, lo, hi = sc.falls_per_1000s(p["F8"]["falls"], 6400.0)
    assert p["F8"]["rate_per_1000s"] == pytest.approx(r)


def test_incomplete_or_mismatched_cells_are_rejected(tmp_path):
    cases = {
        "incomplete": dict(U8=dict(drop_env_rows=1)),
        "exposure": dict(U8=dict(exposure=999.0)),
        "friction": dict(D7=dict(friction=0.45)),
        "task": dict(S2=dict(task="Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0")),
        "push": dict(F8=dict(push=True)),
    }
    for name, kw in cases.items():
        new, ref, _ = make_all(tmp_path / name, envs=4, steps=2000, **kw)
        with pytest.raises(sc.CellInvalid):
            sc.score(new, ref, 0.3, envs=4, steps=2000)
    # a missing summary exposure is accepted from validated metadata, and flagged
    new, ref, _ = make_all(tmp_path / "noexp", envs=4, steps=2000, D7=dict(exposure=None))
    p = sc.score(new, ref, 0.3, envs=4, steps=2000)
    assert p["D7"]["per_seed"][0]["exposure_source"].startswith("validated metadata")
    # wrong command schedule in the folder vs the file
    new, ref, _ = make_all(tmp_path / "cmd", envs=4, steps=2000)
    s = json.loads((ref / "S2_cmd10_s42" / "episodes_summary.json").read_text())
    s["cmd_resample_s"] = 30.0
    (ref / "S2_cmd10_s42" / "episodes_summary.json").write_text(json.dumps(s))
    with pytest.raises(sc.CellInvalid, match="cmd_resample_s"):
        sc.score(new, ref, 0.3, envs=4, steps=2000)


def _c(F8, U8, D7=306, S2=219, move=("F8", "U8", "D7")):
    p = {t: dict(falls=v, speed_ratio=0.9 if t in move or t == "S2" else 0.5, track_per_step=0.80)
         for t, v in dict(F8=F8, U8=U8, D7=D7, S2=S2).items()}
    return sc.components(p)


def test_readout_rules_differ_where_the_draft_is_ambiguous():
    d, pr = "draft-2026-11-24", "proposed-2026-11-25"
    c = _c(300, 200)
    assert sc.readout(c, d)["A"] == sc.readout(c, pr)["A"] == "UPDATED BETTER"
    assert sc.readout(c, pr)["B1"] and sc.readout(c, pr)["B2"]             # 200 <= 244.8 and <= 276.75
    assert sc.readout(c, pr)["follows"].startswith("propose a scored")
    # zero falls in both arms: the draft text makes both conditions true; the proposal reads no difference
    z = _c(0, 0)
    assert sc.readout(z, d)["A"] == "AMBIGUOUS (both conditions hold)"
    assert sc.readout(z, pr)["A"] == "NO CLEAR DIFFERENCE"
    # only U8 keeps movement: the draft lets U8 win; the proposal reports movement not kept
    m = _c(300, 200, move=("U8", "D7"))
    assert sc.readout(m, d)["A"] == "UPDATED BETTER"
    assert sc.readout(m, pr)["A"] == "MOVEMENT NOT KEPT (F8)"
    # B1 met, B2 missed: the draft's progression reads only B2; the proposal needs both
    b = _c(300, 230, S2=150)                         # 230 <= 244.8 (B1), 230 > 190.5 (B2 falls)
    assert sc.readout(b, pr)["B1"] and not sc.readout(b, pr)["B2"]
    assert sc.readout(b, pr)["follows"].startswith("development result")
    assert sc.readout(b, d)["follows"].startswith("development result")
    with pytest.raises(ValueError):
        sc.readout(c, "other")


def test_cli_components_without_rules_and_missing_30s(tmp_path):
    new, ref, _ = make_all(tmp_path, envs=4, steps=2000)
    out = tmp_path / "d8.json"
    assert sc.main(["--new", str(new), "--ref", str(ref), "--friction", "0.3", "--envs", "4", "--steps", "2000",
                    "--also-cmd", "30", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["readout"] is None and r["also"]["30"]["reported"] is False and len(r["also"]["30"]["missing"]) == 8
    assert "READOUT: not made" in out.with_suffix(".txt").read_text()
    with pytest.raises(SystemExit):
        sc.main(["--new", str(new), "--ref", str(ref), "--friction", "0.3", "--rules", "proposed-2026-11-25",
                 "--out", str(tmp_path / "x.json")])
    (ref / "D7_cmd10_s43" / "episodes.csv").unlink()
    assert sc.main(["--new", str(new), "--ref", str(ref), "--friction", "0.3", "--envs", "4", "--steps", "2000",
                    "--out", str(tmp_path / "y.json")]) == 3


# the scorers that produced registered or reported results stay byte-identical (sha256 at base 2d0d2fc)
PINNED = {
    "go2_g0_gate.py": None, "go2_d3_gate.py": None, "go2_d3_fall_timing.py": None, "go2_startup_hazard.py": None,
    "e1r2_score.py": None, "episode_stats.py": None, "fall_rate.py": None, "c0_battery.py": None,
    "c0_supplement.py": None, "go2_c0_check.py": None,
}
PINNED_FILE = Path(__file__).with_name("pinned_scorers.json")


def test_historical_scorers_unchanged():
    pins = json.loads(PINNED_FILE.read_text())
    assert set(pins) == set(PINNED)
    for name, digest in pins.items():
        assert sha256(HERE / name) == digest, f"{name} changed; historical results depend on it"
