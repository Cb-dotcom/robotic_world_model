"""CPU tests for e1_prepare_data.py on a synthetic pre-reset trace (no simulator, no torch)."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
sys.path.insert(0, str(MB / "analysis"))
import e1_prepare_data as prep  # noqa: E402
from pre_reset_transitions import SCHEMA, validate_trace  # noqa: E402


def make_trace(path, T=300, N=8, seed=0):
    rng = np.random.default_rng(seed)
    done = np.zeros((T, N), bool)
    term = np.zeros((T, N), bool)
    tout = np.zeros((T, N), bool)
    # env 0: fall at 99, time-out at 249; env 1: short fall at 20 (skipped); env 2: fall at the last step
    term[99, 0] = done[99, 0] = True
    tout[249, 0] = done[249, 0] = True
    term[20, 1] = done[20, 1] = True
    term[T - 1, 2] = done[T - 1, 2] = True
    state = np.zeros((T, N, 45), np.float32)
    nxt = np.zeros((T, N, 45), np.float32)
    for n in range(N):
        s = rng.normal(size=45).astype(np.float32)
        for t in range(T):
            state[t, n] = s
            nxt[t, n] = rng.normal(size=45).astype(np.float32)
            s = rng.normal(size=45).astype(np.float32) if done[t, n] else nxt[t, n]
    arrays = dict(state=state, next_state=nxt, action=rng.normal(size=(T, N, 12)).astype(np.float32),
                  contact=(rng.random((T, N, 8)) > .5).astype(np.float32),
                  termination_observation=term[..., None].astype(np.float32),
                  terminated=term, time_out=tout, done=done, command=np.zeros((T, N, 3), np.float32))
    meta = dict(schema=SCHEMA, verified_terminal_rows=int(done.sum()), seed=101, num_envs=N)
    with open(path, "xb") as f:
        np.savez_compressed(f, **arrays, metadata=json.dumps(meta))
    return arrays


def test_convert(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "seq_len_from_config", lambda: 40)
    a = make_trace(tmp_path / "b.npz")
    make_trace(tmp_path / "r.npz", seed=1)
    out = tmp_path / "out"
    assert prep.main(["--baseline", str(tmp_path / "b.npz"), "--reference", str(tmp_path / "r.npz"),
                      "--out", str(out), "--test-fraction", "0.25"]) == 0
    s = json.loads((out / "summary.json").read_text())
    b = s["inputs"]["baseline1601"]
    assert b["train_envs"] == [0, 5] and b["test_envs"] == [6, 7]
    assert b["train"]["skipped_short"] == 1 and b["train"]["falls_skipped_short"] == 1
    assert b["train"]["falls_kept"] == 2 and b["train"]["timeouts_kept"] == 1
    m = pd.read_csv(out / "segments" / "manifest.csv")
    assert (m["num_envs"] == 1).all() and (m["rows"] == m["steps"]).all()
    # env 0, episode 0: rows 0..99; the last row is the true terminal next_state with termination 1
    ep = np.loadtxt(out / "segments" / "baseline1601_env000_ep000.csv", delimiter=",", dtype=np.float32)
    assert ep.shape == (100, 66)
    np.testing.assert_array_equal(ep[:, :45], a["next_state"][:100, 0])
    np.testing.assert_array_equal(ep[:, 45:57], a["action"][:100, 0])
    assert ep[-1, 65] == 1 and ep[:-1, 65].sum() == 0
    # fitter pairing: state row k (next_state[k] = state[k+1]) with the action of row k+1 (applied at state[k+1])
    np.testing.assert_array_equal(ep[:-1, :45], a["state"][1:100, 0])
    ep2 = np.loadtxt(out / "segments" / "baseline1601_env000_ep001.csv", delimiter=",", dtype=np.float32)
    assert ep2.shape == (150, 66) and ep2[:, 65].sum() == 0  # a time-out ends with label 0
    with np.load(out / "test_baseline1601.npz") as t:
        meta = validate_trace(t)
        assert t["done"].shape == (300, 2) and meta["e1_test_envs"] == [6, 7]
    with np.load(out / "eth_norm.npz") as z:
        assert z["state_mean"].shape == (45,) and z["state_std"][8] == np.float32(0.04)
    # fitter loader accepts the output (format check without torch)
    first = m.iloc[0]
    data = pd.read_csv(out / "segments" / f"{first['name']}.csv", header=None).values
    assert data.shape == (first["rows"], 66)


def test_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "seq_len_from_config", lambda: 40)
    make_trace(tmp_path / "b.npz")
    (tmp_path / "out").mkdir()
    try:
        prep.main(["--baseline", str(tmp_path / "b.npz"), "--reference", str(tmp_path / "b.npz"),
                   "--out", str(tmp_path / "out")])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("existing --out accepted")


def test_baseline_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "seq_len_from_config", lambda: 40)
    make_trace(tmp_path / "b.npz")
    make_trace(tmp_path / "r.npz", seed=1)
    out = tmp_path / "out"
    assert prep.main(["--baseline", str(tmp_path / "b.npz"), "--reference", str(tmp_path / "r.npz"),
                      "--out", str(out), "--test-fraction", "0.25", "--baseline-tag", "baseline1701"]) == 0
    assert (out / "test_baseline1701.npz").is_file() and not (out / "test_baseline1601.npz").exists()
    assert "baseline1701" in json.loads((out / "summary.json").read_text())["inputs"]
