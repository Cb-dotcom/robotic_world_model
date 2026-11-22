import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import go2_startup_hazard as hz  # noqa: E402
from test_go2_d3_fall_timing import real_cell  # noqa: E402


def test_hazard_bins_by_hand():
    # lengths 15 (fall), 35 (fall), 100 (time-out), 60 (censored); dt 0.02
    L = np.array([15, 35, 100, 60])
    F = np.array([True, True, False, False])
    r = hz.hazards(L, F, 0.02)
    assert r["resets"] == 4 and r["falls"] == 2 and r["falls_per_reset"] == 0.5
    assert r["fall_within_32_rows_per_reset"] == 0.25 and r["fall_within_40_rows_per_reset"] == 0.5
    b = {x["bin"]: x for x in r["bins"]}
    assert b["(10, 20]"]["at_risk"] == 4 and b["(10, 20]"]["falls"] == 1
    assert b["(10, 20]"]["exposure_s"] == pytest.approx((5 + 10 + 10 + 10) * 0.02)
    assert b["(32, 40]"]["at_risk"] == 3 and b["(32, 40]"]["falls"] == 1
    assert b["(32, 40]"]["exposure_s"] == pytest.approx((3 + 8 + 8) * 0.02)
    assert b["(10, 20]"]["hazard_per_s"][0] == pytest.approx(1 / (35 * 0.02))
    assert b["(10, 20]"]["conditional"] == 0.25
    assert r["exposure_s"] == pytest.approx(L.sum() * 0.02)


def test_real_writer_end_to_end(tmp_path):
    for s in (42, 43):
        real_cell(tmp_path / f"D3_cmd10_s{s}", {15, 399}, s)
        real_cell(tmp_path / f"S2_cmd10_s{s}", {399}, s)
    res = hz.analyse(tmp_path)
    assert res["D3"]["falls"] == 4 and res["S2"]["falls"] == 2
    assert res["D3"]["fall_within_32_rows_per_reset"] > 0 and res["S2"]["fall_within_32_rows_per_reset"] == 0
    assert len(res["hazard_ratio"]) == len(hz.EDGES) - 1
    hz.main(["--eval", str(tmp_path), "--out", str(tmp_path / "o")])
    assert json.loads((tmp_path / "o" / "startup_hazard.json").read_text())["tags"] == ["D3", "S2"]
    assert (tmp_path / "o" / "startup_hazard.txt").read_text().startswith("Startup hazard")
