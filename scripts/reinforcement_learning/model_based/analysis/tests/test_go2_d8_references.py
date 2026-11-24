"""A4 D8 reference check: saved D7 / s2 cells are validated, hashed and totalled; missing 30 s cells are listed."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import go2_d8_references as rf  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402
from test_go2_d8_score import write_cell  # noqa: E402


def refs(tmp_path, readback=True):
    ref = tmp_path / "eval"
    ck = {"D7": tmp_path / "policy_4298.pt", "S2": tmp_path / "s2.pt"}
    for t, p in ck.items():
        p.write_bytes(t.encode())
    falls = {}
    for t in ("D7", "S2"):
        for s in (42, 43):
            d = ref / f"{t}_cmd10_s{s}"
            falls[(t, s)] = write_cell(d, envs=32, steps=200, n_falls=3 if t == "D7" else 2, seed=s, seed_data=s)
            js = json.loads((d / "episodes_summary.json").read_text())
            js["checkpoint"] = str(ck[t])
            (d / "episodes_summary.json").write_text(json.dumps(js))
            (d / "eval.log").write_text("[friction] verified in the simulator: {...}\n" if readback else "no line\n")
    return ref, ck, falls


def test_ok_and_missing_30s(tmp_path):
    ref, ck, falls = refs(tmp_path)
    exp = {t: sha256(p)[:8] for t, p in ck.items()}
    tot = {(t, 10): falls[(t, 42)] + falls[(t, 43)] for t in ("D7", "S2")}
    r = rf.verify(ref, 0.3, [10, 30], exp, tot, envs=32, steps=200)
    assert r["ok"], r["problems"]
    assert r["schedules"]["10"]["policies"]["D7"]["falls"] == tot[("D7", 10)]
    assert r["schedules"]["30"]["status"] == "MISSING" and len(r["schedules"]["30"]["missing"]) == 4
    row = r["schedules"]["10"]["policies"]["S2"]["per_seed"][0]
    assert row["friction_readback"] and row["checkpoint_sha256"] == sha256(ck["S2"]) and row["eval_log_sha256"]


def test_problems_are_named(tmp_path):
    ref, ck, falls = refs(tmp_path, readback=False)
    r = rf.verify(ref, 0.3, [10], {"D7": "deadbeef"}, {("S2", 10): 999}, envs=32, steps=200)
    assert not r["ok"]
    text = " ".join(r["problems"])
    assert "read-back" in text and "deadbeef" in text and "documented 999" in text
    assert rf.main(["--ref", str(ref), "--cmd", "10", "--out", str(tmp_path / "r.json")]) == 3
