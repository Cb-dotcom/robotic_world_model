"""A4 D6 (22 Nov 2026): D2's data with real startup prefixes (the recorded pre-action reset state)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

MB = Path(__file__).resolve().parents[2]
for p in (str(MB), str(MB / "analysis"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rsl_rl.modules")

import go2_mixed_prepare as mix  # noqa: E402
import go2_startup_prepare as sp  # noqa: E402
from test_c0_battery import random_trace  # noqa: E402
from test_go2_g0 import save  # noqa: E402


def startup_trace(seed=0, T=300, N=6, mid_start_env=2):
    """Falls at 150 (env 0), 60 (env 3) and a SHORT fall 10 rows after the start of env 5; time-outs; env
    `mid_start_env` starts mid-episode (episode_step 40 at row 0: no physical reset in the recording)."""
    tr = random_trace(T=T, N=N, falls=((150, 0), (60, 3), (9, 5)), timeouts=((99, 1), (199, 1), (249, 4)), seed=seed)
    g = np.random.default_rng(seed + 1)
    tr["contact"] = (g.random((T, N, 8)) < .5).astype(np.float32)
    tr["termination_observation"] = tr["terminated"][..., None].astype(np.float32)
    tr["previous_action"] = g.normal(0, .5, (T, N, 12)).astype(np.float32)
    step = np.zeros((T, N), np.int64)
    for n in range(N):
        k = 40 if n == mid_start_env else 1
        for t in range(T):
            step[t, n] = k
            k = 1 if tr["done"][t, n] else k + 1
    tr["episode_step"] = step
    return tr


def test_reset_row_first_transition_and_short_fall(tmp_path):
    tr = startup_trace()
    src = save(tmp_path / "r0.npz", tr)
    sh = save(tmp_path / "sh.npz", startup_trace(seed=7))
    r = sp.prepare_startup(src, sh, tmp_path / "d6", frac=0.0, seed=101)
    out = tmp_path / "d6"
    startup = set(json.loads((out / "startup.json").read_text())["segments"])
    names = [l.split(",")[0] for l in (out / "segments" / "manifest.csv").read_text().splitlines()[1:]]
    checked_first = checked_short = checked_mid = 0
    for name in names:
        if not name.startswith("r0_"):
            continue
        n, s = int(name[6:9]), int(name[11:])
        seg = np.loadtxt(out / "segments" / f"{name}.csv", delimiter=",", ndmin=2)
        if name in startup:
            # row 0 = the recorder's pre-action reset state and previous action; contact and termination 0
            np.testing.assert_allclose(seg[0, :45], tr["state"][s, n], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(seg[0, 45:57], tr["previous_action"][s, n], rtol=1e-6, atol=1e-6)
            assert not seg[0, 57:].any()
            # the fitter pairs state row j with action row j+1: (reset state, first action) -> first successor
            np.testing.assert_allclose(seg[1, 45:57], tr["action"][s, n], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(seg[1, :45], tr["next_state"][s, n], rtol=1e-6, atol=1e-6)
            checked_first += 1
            if n == 5 and s == 0:
                # the short fall: 10 recorded rows + the reset row, terminal target 1 on its last row only
                assert len(seg) == 11 and seg[-1, -1] == 1 and not seg[:-1, -1].any()
                checked_short += 1
        else:
            assert n == 2 and s == 0                  # only the mid-episode start has no reset row
            np.testing.assert_allclose(seg[0, :45], tr["next_state"][s, n], rtol=1e-6, atol=1e-6)
            checked_mid += 1
    assert checked_first > 0 and checked_short == 1 and checked_mid == 1
    c = r["sources"]["r0"]["counts"]
    assert c["without_reset_row"] == 1 and c["with_reset_row"] == c["training"] - 1
    assert c["falls_in_startup_windows"] == 3 and c["falls_in_standard_windows"] == 2   # the short fall: startup only
    with np.load(out / "reset_states.npz") as z:
        k = int(np.flatnonzero((z["source"] == 0) & (z["env"] == 5) & (z["row"] == 0))[0])
        np.testing.assert_allclose(z["state"][k], tr["state"][0, 5])
        np.testing.assert_allclose(z["previous_action"][k], tr["previous_action"][0, 5])
        assert len(z["env"]) == r["reset_states"]
    eps = r["sources"]["r0"]["counts"]["training"] + r["sources"]["sh"]["counts"]["training"]
    rows = r["sources"]["r0"]["counts"]["training_episode_rows"] + r["sources"]["sh"]["counts"]["training_episode_rows"]
    assert r["startup_reset_share"] == pytest.approx(min(0.5, 256 / (rows / eps)))
    # start blocks: one end flag per training episode, reset rows included
    blocks = [np.loadtxt(f, delimiter=",", ndmin=2) for f in sorted((out / "start_data").glob("*.csv"))]
    assert sum(int(b[:, -1].sum()) for b in blocks) == eps
    with pytest.raises(FileExistsError):
        sp.prepare_startup(src, sh, out)


def test_withheld_split_equals_d2(tmp_path):
    r0 = save(tmp_path / "r0.npz", startup_trace())
    sh = save(tmp_path / "sh.npz", startup_trace(seed=7))
    mix.prepare_mixed(r0, sh, tmp_path / "d2")
    r = sp.prepare_startup(r0, sh, tmp_path / "d6", d2_data=tmp_path / "d2")
    held = json.loads((tmp_path / "d6" / "r0" / "withheld.json").read_text())
    for h in held:
        assert f"r0_env{h['env']:03d}_r{h['first']:05d}" not in json.loads((tmp_path / "d6" / "startup.json").read_text())["segments"]
    assert r["sources"]["r0"]["counts"]["withheld"] == len(held)
    (tmp_path / "d2" / "shift" / "withheld.json").write_text("[]")
    with pytest.raises(ValueError, match="differ from D2"):
        sp.prepare_startup(r0, sh, tmp_path / "d6b", d2_data=tmp_path / "d2")


def test_recording_without_episode_step_is_refused(tmp_path):
    tr = startup_trace()
    del tr["episode_step"]
    src = save(tmp_path / "r0.npz", tr)
    with pytest.raises(ValueError, match="episode_step"):
        sp.prepare_startup(src, src, tmp_path / "d6")
