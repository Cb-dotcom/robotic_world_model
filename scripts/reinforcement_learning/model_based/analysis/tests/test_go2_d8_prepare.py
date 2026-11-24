"""A4 D8 data: the probe and every withheld episode are absent from the fitting segments, the start bank and the reset
states (checked on the written files, by row identity); the window census equals the fitter's own count; probe counts."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MB = Path(__file__).resolve().parents[2]
for p in (str(MB), str(MB / "analysis"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rsl_rl.modules")

import go2_d8_prepare as d8  # noqa: E402
import go2_round0_prepare as prep  # noqa: E402
import go2_startup_prepare as sp  # noqa: E402
from test_go2_g0 import save  # noqa: E402
from test_go2_startup_prepare import startup_trace  # noqa: E402


def setup(tmp_path, frac=0.2):
    r0 = save(tmp_path / "r0.npz", startup_trace(seed=0))
    sh = save(tmp_path / "sh.npz", startup_trace(seed=7))
    sp.prepare_startup(r0, sh, tmp_path / "d6b", frac=frac, seed=101, reset_torque="zero")
    batch = save(tmp_path / "batch.npz", startup_trace(seed=21))
    probe = save(tmp_path / "probe.npz", startup_trace(seed=33))
    return r0, sh, batch, probe


def test_pooled_data_and_split_membership(tmp_path):
    r0, sh, batch, probe = setup(tmp_path)
    r = d8.prepare_d8(tmp_path / "d6b", batch, tmp_path / "d8", probe, "recompute", frac=0.2)
    out = tmp_path / "d8"
    chk = json.loads((out / "split_check.json").read_text())
    assert chk["ok"] and r["split_check_ok"]
    g = chk["groups"]
    assert g["probe_916_all_rows"]["rows_checked"] > 1000 and g["d7_batch_withheld"]["episodes"] > 0
    assert all(v["found_in_outputs"] == 0 for v in g.values())
    # the D6b part is unchanged; the batch's training episodes are present, its withheld ones are not
    names = [l.split(",")[0] for l in (out / "segments" / "manifest.csv").read_text().splitlines()[1:]]
    n6 = len((tmp_path / "d6b" / "segments" / "manifest.csv").read_text().splitlines()) - 1
    assert sum(n.startswith("d7_") for n in names) == r["batch"]["counts"]["segments"] and len(names) == n6 + r["batch"]["counts"]["segments"]
    tr = d8.load(batch)
    present, absent = d8.output_keys(out)[0], 0
    held = d8.withheld_episodes(out / "d7" / "withheld.json")
    for n, s, e in held:
        absent += not (d8._keys(prep.rows_of(tr, n, s, e)) & present)
    assert absent == len(held)
    train = [x for x in d8.all_episodes(tr) if x not in held]
    assert all(d8._keys(prep.rows_of(tr, n, s, e)) <= present for n, s, e in train)
    # reset states: D6b's two sources then the batch (source 2), torque block 0 N m everywhere
    with np.load(out / "reset_states.npz") as z:
        assert set(np.unique(z["source"])) == {0, 1, 2} and not z["state"][:, 33:45].any()
    with np.load(tmp_path / "d6b" / "reset_states.npz") as z6, np.load(out / "reset_states.npz") as z:
        assert np.array_equal(z["state"][:len(z6["state"])], z6["state"])
    # q: both values recorded; the chosen rule decides
    m6 = json.loads((tmp_path / "d6b" / "manifest.json").read_text())
    assert r["q_d6b"] == m6["startup_reset_share"] and r["startup_reset_share"] == r["q_recompute"]
    r2 = d8.prepare_d8(tmp_path / "d6b", batch, tmp_path / "d8b", probe, "d6b", frac=0.2)
    assert r2["startup_reset_share"] == m6["startup_reset_share"]


def test_a_leaked_probe_row_is_caught(tmp_path):
    r0, sh, batch, probe = setup(tmp_path)
    d8.prepare_d8(tmp_path / "d6b", batch, tmp_path / "d8", probe, "d6b", frac=0.2)
    out = tmp_path / "d8"
    tp = d8.load(probe)
    leak = prep.rows_of(tp, 1, 0, 3)                               # four probe rows (0..3), written into the start bank
    f = sorted((out / "start_data").glob("state_action_data_*.csv"))[0]
    with f.open("a") as fh:
        np.savetxt(fh, leak, delimiter=",", fmt="%.9g")
    chk = d8.verify_split(out, [("probe", tp, d8.all_episodes(tp))])
    assert not chk["ok"] and chk["groups"]["probe"]["found_in_outputs"] == 4
    # a withheld D6b episode copied into the segments is caught too
    t0 = d8.load(r0)
    wh = d8.withheld_episodes(tmp_path / "d6b" / "r0" / "withheld.json")
    n, s, e = wh[0]
    np.savetxt(out / "segments" / "leak.csv", prep.rows_of(t0, n, s, e), delimiter=",", fmt="%.9g")
    chk = d8.verify_split(out, [("r0", t0, wh)])
    assert not chk["ok"] and chk["groups"]["r0"]["found_in_outputs"] >= e - s + 1


def test_refusals(tmp_path):
    r0, sh, batch, probe = setup(tmp_path)
    with pytest.raises(ValueError, match="probe is one of the fitting sources"):
        d8.prepare_d8(tmp_path / "d6b", batch, tmp_path / "x", batch, "d6b")
    with pytest.raises(ValueError, match="q rule"):
        d8.prepare_d8(tmp_path / "d6b", batch, tmp_path / "y", probe, "other")
    sp.prepare_startup(r0, sh, tmp_path / "d6", frac=0.2, seed=101, reset_torque="recorded")
    with pytest.raises(ValueError, match="0 N m"):
        d8.prepare_d8(tmp_path / "d6", batch, tmp_path / "z", probe, "d6b")


def test_window_census_equals_the_fitter(tmp_path):
    import fit_world_model_go2_pilot_segments as F
    r0, sh, batch, probe = setup(tmp_path)
    r = d8.prepare_d8(tmp_path / "d6b", batch, tmp_path / "d8", probe, "d6b", frac=0.2)
    out = tmp_path / "d8"
    segs, _, _ = F.read_segments(out / "segments", 45, 12, 0, 8, 1)
    z = torch.zeros(45), torch.ones(45), torch.zeros(12), torch.ones(12)
    H, seq = d8._horizons()
    win = F.build_segment_windows(segs, 45, 12, 8, 1, H, seq - H, "cpu", *z, skip_short=[])
    startup = json.loads((out / "startup.json").read_text())["segments"]
    groups = F.build_startup_groups(segs, startup, 45, 12, H, seq - H, "cpu", *z)
    c = r["window_census"]
    assert c["d6b"]["standard"] + c["d7"]["standard"] == win.shape[0]
    assert c["d6b"]["startup"] + c["d7"]["startup"] == sum(g.shape[0] for g in groups.values())
    assert 0 < c["d7_share_uniform"] < 1


def test_probe_counts_and_minimum():
    tr = startup_trace(seed=0)            # falls at 150 (env 0), 60 (env 3), a 10-row fall in env 5; env 2 mid-episode
    tr["done"][50, 2] = tr["time_out"][50, 2] = True     # env 2's first episode ends: finished, but no reset in view
    c = d8.probe_counts(tr)
    assert c["startup_falls"] == 1 and c["startup_falls_le25"] == 1        # the 10-row fall
    assert c["later_falls"] == 2 and c["censored_no_reset"] == 1
    assert c["censored_unfinished"] == sum(1 for *_, f in prep.episodes(tr["done"]) if not f)
    assert c["from_reset"] + c["censored_no_reset"] + c["censored_unfinished"] == c["episodes"]
    m = d8.probe_minimum(c)
    assert not m["ok"] and d8.probe_minimum(c, 1, 1)["ok"]
