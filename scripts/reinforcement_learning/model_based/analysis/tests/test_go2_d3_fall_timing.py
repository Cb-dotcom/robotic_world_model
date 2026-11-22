import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import go2_d3_fall_timing as ft  # noqa: E402

FIELDS = ["env", "start", "length", "counted_steps", "fall", "censored", "mean_speed_act", "steps_since_cmd_change",
          "cmd_speed_end"]


def cell(folder, episodes, exposure=640.0):
    folder.mkdir(parents=True)
    with open(folder / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (length, fall, censored, since, spd) in enumerate(episodes):
            w.writerow(dict(env=i, start=0, length=length, counted_steps=length - 1, fall=fall, censored=censored,
                            mean_speed_act=1.0, steps_since_cmd_change=since, cmd_speed_end=spd))
    falls = sum(1 for e in episodes if e[1] and not e[2])
    (folder / "episodes_summary.json").write_text(json.dumps(dict(step_dt=0.02, exposure_s=exposure, falls=falls,
                                                                  friction_shift=dict(factor=0.3, verified=True))))


def make(tmp_path, d3, s2):
    for s in (42, 43):
        cell(tmp_path / f"D3_cmd10_s{s}", d3)
        cell(tmp_path / f"S2_cmd10_s{s}", s2)
    return tmp_path


def test_bins_cap_and_excess(tmp_path):
    d3 = [(256, True, False, 256, 1.2), (257, True, False, 257, 0.7), (600, True, False, 50, 1.6), (1000, False, False, 500, 0.2),
          (40, False, True, 40, 0.1)]
    s2 = [(600, True, False, 100, 0.4), (1000, False, False, 500, 0.2)]
    r = ft.analyse(make(tmp_path, d3, s2))
    p = r["D3"]["pooled"]
    assert p["falls"] == 6 and r["S2"]["pooled"]["falls"] == 2
    assert p["within_cap"] == 2                       # 256 steps = 5.12 s counts as within the cap; 257 does not
    assert p["before_any_change"] == 4                # counter == length
    assert p["reset_counts"] == [0, 0, 2, 2, 2]       # 256 steps -> (2.56, 5.12]; 257 -> (5.12, 10]; 600 (12 s) -> (10, end]
    assert r["excess_D3_minus_S2"]["falls"] == 4
    assert p["speed_counts"] == [0, 2, 2, 2]
    assert p["distance_m"] == pytest.approx(2 * 0.02 * (255 + 256 + 599 + 999 + 39))
    assert p["falls_per_km"] == pytest.approx(1000 * 6 / p["distance_m"])
    assert "excess falls D3 - S2: total 4" in ft.text(r)


def test_summary_mismatch_and_exposure_mismatch(tmp_path):
    d = make(tmp_path / "a", [(100, True, False, 100, 1.0)], [(100, False, False, 100, 1.0)])
    s = json.loads((d / "D3_cmd10_s42" / "episodes_summary.json").read_text())
    s["falls"] = 5
    (d / "D3_cmd10_s42" / "episodes_summary.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="differ from the summary"):
        ft.analyse(d)
    e = tmp_path / "b"
    for t in (42, 43):
        cell(e / f"D3_cmd10_s{t}", [(100, True, False, 100, 1.0)], exposure=640.0)
        cell(e / f"S2_cmd10_s{t}", [(100, True, False, 100, 1.0)], exposure=320.0)
    with pytest.raises(ValueError, match="exposures differ"):
        ft.analyse(e)


def test_right_closed_bins():
    assert ft.hist([0.0, 1.0, 1.0000001, 5.12], ft.RESET_BINS, right=True) == [2, 1, 1, 0, 0]
    assert ft.labels(ft.RESET_BINS, True)[2] == "(2.56, 5.12]"


def test_main_writes_once(tmp_path):
    d = make(tmp_path / "e", [(100, True, False, 100, 1.0)], [(100, False, False, 100, 1.0)])
    ft.main(["--eval", str(d), "--out", str(tmp_path / "o")])
    assert json.loads((tmp_path / "o" / "d3_fall_timing.json").read_text())["D3"]["pooled"]["falls"] == 2
    with pytest.raises(FileExistsError):
        ft.main(["--eval", str(d), "--out", str(tmp_path / "o")])
