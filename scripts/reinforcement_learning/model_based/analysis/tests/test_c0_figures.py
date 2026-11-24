"""C0 paper figures: drawn only after the registered numbers are reproduced from c0_windows.npz written by the real
c0_battery.run; the curves equal the supplement's d25 and d16 values; a changed report stops the script."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("rsl_rl.modules")
pytest.importorskip("matplotlib")

import c0_figures as fig  # noqa: E402
from test_c0_supplement import c0_run  # noqa: E402


def test_figures_after_the_gate(tmp_path, monkeypatch):
    src = c0_run(tmp_path, monkeypatch, n_boot=300)
    res = fig.run(src / "c0_windows.npz", src / "c0_report.json", tmp_path / "figs", n_boot=300)
    assert res["gate"]["check"]["reproduced"]
    for name in ("fig_c0_disagreement.pdf", "fig_c0_disagreement.png", "fig_c0_healing.pdf", "fig_c0_healing.png",
                 "c0_curves.csv", "c0_figures.json"):
        assert (tmp_path / "figs" / name).stat().st_size > 0
    rows = res["curves"]
    assert [r["d"] for r in rows] == list(range(26)) and rows[25]["ms_before_terminal"] == 0
    for r in rows:                                          # H = log u_ctx - log u_imag (geometric means)
        for kind in ("fall", "nonfall"):
            assert r[kind]["H"] == pytest.approx(np.log(r[kind]["u_ctx"]) - np.log(r[kind]["u_imag"]))
    assert abs(rows[0]["fall"]["H"]) < 1e-6                 # d = 0: both arms see the same inputs
    with open(tmp_path / "figs" / "c0_curves.csv") as f:
        table = list(csv.DictReader(f))
    assert len(table) == 52 and float(table[-2]["H"]) == pytest.approx(rows[25]["fall"]["H"])


def test_gate_stops_on_a_changed_report(tmp_path, monkeypatch):
    src = c0_run(tmp_path, monkeypatch, n_boot=300)
    rep = json.loads((src / "c0_report.json").read_text())
    rep["report"]["pooled"]["fall"]["rows"]["d25/sum"]["H"] *= 1.01
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rep))
    rc = fig.main(["--windows", str(src / "c0_windows.npz"), "--report", str(bad), "--out", str(tmp_path / "x"),
                   "--n-boot", "300"])
    assert rc == 3 and not (tmp_path / "x").exists()


def test_redraw_reuses_the_gate_passed_curves(tmp_path, monkeypatch):
    src = c0_run(tmp_path, monkeypatch, n_boot=300)
    res = fig.run(src / "c0_windows.npz", src / "c0_report.json", tmp_path / "figs", n_boot=300)
    rc = fig.main(["--redraw-from", str(tmp_path / "figs"), "--out", str(tmp_path / "re")])
    assert rc == 0
    for name in ("fig_c0_disagreement.pdf", "fig_c0_disagreement.png", "fig_c0_healing.pdf", "fig_c0_healing.png"):
        assert (tmp_path / "re" / name).stat().st_size > 0
    meta = json.loads((tmp_path / "re" / "redraw.json").read_text())
    assert meta["source_sha256"] == fig.sup.sha256(tmp_path / "figs" / "c0_figures.json")
    assert json.loads((tmp_path / "figs" / "c0_figures.json").read_text())["curves"] == res["curves"]
    bad = json.loads((tmp_path / "figs" / "c0_figures.json").read_text())
    bad["gate"]["check"]["reproduced"] = False
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "c0_figures.json").write_text(json.dumps(bad))
    assert fig.main(["--redraw-from", str(tmp_path / "bad"), "--out", str(tmp_path / "x")]) == 3
    assert not (tmp_path / "x").exists()
