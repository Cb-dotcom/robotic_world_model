"""A3 §1 supplementary C0 analyses: they reproduce the registered d25 numbers from the saved windows exactly,
and the decomposition is an identity."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("rsl_rl.modules")

import c0_battery as c0  # noqa: E402
import c0_supplement as sup  # noqa: E402
from test_c0_battery import identity_contract, random_trace, real_model  # noqa: E402


def c0_run(tmp_path, monkeypatch, n_boot=300):
    import analyse_transition_fidelity as fidelity
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    paths = []
    for i, falls in enumerate((((69, 0), (120, 2)), ((80, 1), (130, 0)))):
        p = tmp_path / f"t{i}.npz"
        np.savez(p, **random_trace(falls=falls, seed=i))
        paths.append((f"src{i}", p))
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    c0.run(wm, paths, tmp_path / "c0", n_boot=n_boot)
    return tmp_path / "c0"


def test_supplement_reproduces_the_registered_numbers(tmp_path, monkeypatch):
    out = c0_run(tmp_path, monkeypatch)
    res = sup.analyse(sup.load_sources(out / "c0_windows.npz"), n_boot=300, seed=0)
    chk = sup.check_registered(res, out / "c0_report.json")
    assert chk["reproduced"], chk
    for d in ("d25", "d16"):
        r = res[d]
        assert abs(r["difference"]["identity_error"]) < 1e-12
        assert r["difference"]["dH"] == pytest.approx(r["difference"]["context_part"] + r["difference"]["imagination_part"])
        assert r["fall"]["geomean_imag_over_ctx"] == pytest.approx(np.exp(-r["fall"]["H"]))
    assert res["d16"]["ms_before_terminal_target"] == 180
    rep = json.loads((out / "c0_report.json").read_text())["report"]["pooled"]
    assert res["d16"]["fall"]["H"] == pytest.approx(rep["fall"]["rows"]["d16/sum"]["H"], rel=1e-6)


def test_supplement_cli_stops_when_the_report_does_not_match(tmp_path, monkeypatch):
    out = c0_run(tmp_path, monkeypatch, n_boot=10_000)
    monkeypatch.setattr(sys, "argv", ["c0_supplement.py", "--windows", str(out / "c0_windows.npz"),
                                      "--report", str(out / "c0_report.json"), "--out", str(tmp_path / "s")])
    sup.main()
    assert json.loads((tmp_path / "s" / "c0_supplement.json").read_text())["reproduction"]["reproduced"]
    rep = json.loads((out / "c0_report.json").read_text())
    rep["report"]["pooled"]["fall"]["rows"]["d25/sum"]["H"] *= 1.01
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rep))
    monkeypatch.setattr(sys, "argv", ["c0_supplement.py", "--windows", str(out / "c0_windows.npz"),
                                      "--report", str(bad), "--out", str(tmp_path / "s2")])
    with pytest.raises(SystemExit, match="NOT reproduced"):
        sup.main()
