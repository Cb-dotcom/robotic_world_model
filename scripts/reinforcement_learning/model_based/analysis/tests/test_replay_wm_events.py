"""CPU tests for analysis/replay_wm_events.py (world-model replay around command changes)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # model_based/
_RSL = os.environ.get("RSL_RL_PATH", "/home/claude/cb-dotcom/rsl_rl_rwm")
if os.path.isdir(_RSL) and _RSL not in sys.path:
    sys.path.insert(0, _RSL)
sys.path.insert(0, HERE)                                    # tests/ (make_wm_ckpt)

import replay_wm_events as rp  # noqa: E402

H, W = 32, 25


def _traj(T=400, N=3, seed=0):
    rng = np.random.default_rng(seed)
    state = np.cumsum(rng.normal(0, 0.05, (T, N, 45)), axis=0).astype(np.float32)
    state[..., 8] = -1.0  # gravity z
    action = rng.normal(0, 0.5, (T, N, 12)).astype(np.float32)
    cmd = np.zeros((T, N, 3), np.float32)
    done = np.zeros((T, N), bool)
    to = np.zeros((T, N), bool)
    # env 0: command change at 100, fall at 160 (60 steps later) -> "fall"
    cmd[:100, 0, 0] = 0.5; cmd[100:, 0, 0] = -1.0
    done[160, 0] = True; cmd[160:, 0, 0] = 0.3
    # env 1: change at 100, time-out at 300 -> "survived"; change at 20 -> short history
    cmd[:20, 1, 0] = 0.1; cmd[20:100, 1, 0] = 0.2; cmd[100:, 1, 0] = 0.9
    done[300, 1] = True; to[300, 1] = True; cmd[300:, 1, 0] = 0.0
    # env 2: change at 100, fall at 115 (episode ends < W+1 after c) -> excluded too_close_to_end;
    #        new episode starts at 115; change at 200 runs to the end (T-1-200 >= 100) -> "survived";
    #        change at 350 -> censored too soon -> "other"
    cmd[:100, 2, 1] = 0.4; cmd[100:115, 2, 1] = -0.4
    done[115, 2] = True; cmd[115:200, 2, 1] = 0.0; cmd[200:350, 2, 1] = 0.7; cmd[350:, 2, 1] = -0.7
    return {"state": state, "action": action, "cmd": cmd, "done": done, "time_out": to}


def test_episode_ids():
    d = np.array([[0], [0], [1], [0], [1]], bool)
    assert rp.episode_ids(d)[:, 0].tolist() == [0, 0, 1, 1, 2]


def test_find_events_labels_and_exclusions():
    tr = _traj()
    ev, ex = rp.find_events(tr["cmd"], tr["done"], tr["time_out"], H, W, 100)
    got = sorted((e["env"], e["c"], e["label"]) for e in ev)
    assert got == [(0, 100, "fall"), (1, 100, "survived"), (2, 200, "survived")]
    assert ex == {"short_history": 1, "too_close_to_end": 1, "other": 1}
    e0 = [e for e in ev if e["env"] == 0][0]
    assert e0["end"] == 160 and e0["last_row"] == 159 and e0["end_is_fall"]


def test_fall_later_than_window_is_survived():
    tr = _traj()
    ev, _ = rp.find_events(tr["cmd"], tr["done"], tr["time_out"], H, W, fall_within=50)
    lab = {(e["env"], e["c"]): e["label"] for e in ev}
    assert lab[(0, 100)] == "survived"  # fall 60 steps after the change > 50


def test_auc_basic():
    assert rp.auc([2, 3], [0, 1]) == 1.0
    assert rp.auc([0, 1], [2, 3]) == 0.0
    assert rp.auc([1, 1], [1, 1]) == 0.5
    assert np.isnan(rp.auc([], [1]))


torch = pytest.importorskip("torch")
try:
    from rsl_rl.modules import SystemDynamicsEnsemble  # noqa: F401
    HAVE_RSL = True
except Exception:  # pragma: no cover
    HAVE_RSL = False


@pytest.mark.skipif(not HAVE_RSL, reason="rsl_rl not importable")
def test_end_to_end_cpu(tmp_path):
    from test_eval_signals import make_wm_ckpt
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck), seed=1, normalized=True)
    tr = _traj()
    npz = tmp_path / "t.npz"
    np.savez_compressed(npz, **tr, meta=json.dumps({}))
    out = tmp_path / "out"
    rp.main(["--traj", str(npz), "--wm", str(ck), "--config", "go2_flat", "--device", "cpu",
             "--out", str(out), "--rollout_k", "30"])
    s = json.load(open(out / "summary.json"))
    ev = json.load(open(out / "events.json"))
    assert s["n_fall"] == 1 and s["n_survived"] == 2
    for e in ev:
        assert len(e["pred_term_max_heads"]) == 5
        assert 0.0 <= e["pred_term_max_mean"] <= 1.0
        assert e["rollout_k"] == min(30, e["last_row"] - e["c"])
        assert e["epi_mean"] >= 0 and e["mse_mean"] >= 0
    # jump baseline: env 0 jump 1.5, env 1 jump 0.7, env 2 (c=200) jump 0.7
    j = {(e["env"], e["c"]): e["jump"] for e in ev}
    assert j[(0, 100)] == pytest.approx(1.5) and j[(1, 100)] == pytest.approx(0.7)


@pytest.mark.skipif(not HAVE_RSL, reason="rsl_rl not importable")
def test_teacher_forced_window_matches_scorer_pairing(tmp_path):
    """One teacher-forced window must equal the scorer's per-sample path (wm_score_indices)."""
    from test_eval_signals import make_wm_ckpt
    from eval_signals import load_wm, wm_score_indices
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck), seed=2, normalized=True)
    sd, ckpt, Hs = load_wm(str(ck), "cpu")
    sd2, H2, norm, _ = rp.load_wm_cfg(str(ck), "go2_flat", "cpu")
    tr = _traj()
    ev = [{"env": 1, "c": 100, "label": "survived", "end": 300, "end_is_fall": False, "last_row": 299}]
    out = rp.run_replay(sd2, H2, norm, tr, ev, W=1, rollout_k=2, device="cpu")
    # scorer on env-1 rows (normalized identically), index f = 101
    S = ((tr["state"][:, 1] - norm["state_mean"]) / norm["state_std"]).astype(np.float32)
    A = ((tr["action"][:, 1] - norm["action_mean"]) / norm["action_std"]).astype(np.float32)
    ref = wm_score_indices(sd, S, A, [101], Hs, "cpu")
    assert out[0]["epi_mean"] == pytest.approx(ref[0, 0], rel=1e-5)
    assert out[0]["term_max"] == pytest.approx(ref[0, 2], rel=1e-5)
