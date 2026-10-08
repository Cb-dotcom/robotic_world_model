# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only synthetic tests for analysis/eval_signals.py.

Run from anywhere:  python -m pytest scripts/reinforcement_learning/model_based/analysis/tests -q
The WM parity tests need torch and rsl_rl (path from $RSL_RL_PATH, default the local clone).
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(HERE)
MODEL_BASED_DIR = os.path.dirname(ANALYSIS_DIR)
RSL_RL_PATH = os.environ.get("RSL_RL_PATH", "/home/claude/cb-dotcom/rsl_rl_rwm")
for p in (ANALYSIS_DIR, MODEL_BASED_DIR, RSL_RL_PATH):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import eval_signals as es  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from rsl_rl.modules import SystemDynamicsEnsemble  # noqa: F401
    HAVE_RSL = torch is not None
except Exception:  # pragma: no cover
    HAVE_RSL = False

needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")
needs_rsl = pytest.mark.skipif(not HAVE_RSL, reason="torch/rsl_rl not importable")

H = 32


# ======================================================================================
# Verbatim references copied from score_go2_exploit_trace_uncertainty.py
# ======================================================================================
def ref_scorer_loop(term_all, num_envs, steps_per_env, H):
    raw_terms = int(term_all.sum())
    seam_terms = 0
    interior_terms = 0
    fall_idx = []
    pre1_idx = []
    pre5_idx = []
    neg_cand = []

    for e in range(num_envs):
        base = e * steps_per_env
        seam = base + steps_per_env - 1
        if term_all[seam]:
            seam_terms += 1

        for local_t in range(H, steps_per_env):
            f = base + local_t
            is_seam = f == seam
            hist_clean = term_all[f - H:f].sum() == 0

            if term_all[f] and (not is_seam):
                interior_terms += 1

            if term_all[f] and (not is_seam) and hist_clean:
                fall_idx.append(f)
                if local_t - 1 >= H and term_all[f - H - 1:f].sum() == 0:
                    pre1_idx.append(f - 1)
                if local_t - 5 >= H and term_all[f - H - 5:f - 4].sum() == 0:
                    pre5_idx.append(f - 5)

            if (not term_all[f]) and hist_clean:
                neg_cand.append(f)

    rng = np.random.default_rng(0)
    return fall_idx, pre1_idx, pre5_idx, neg_cand, raw_terms, seam_terms, interior_terms, rng


def ref_auc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ======================================================================================
# Synthetic termination arrays
# ======================================================================================
def synth_terms(seed, num_envs, steps, p=0.004):
    rng = np.random.default_rng(seed)
    t = rng.random(num_envs * steps) < p
    for e in range(num_envs):
        base = e * steps
        r = rng.random()
        if r < 0.3:
            t[base + steps - 1] = True  # seam termination
        if rng.random() < 0.3:
            t[base + rng.integers(0, H + 6)] = True  # fall near the start (local_t < H or ~H)
        if rng.random() < 0.4:  # two falls < H apart
            a = rng.integers(H + 10, steps - 60)
            t[base + a] = True
            t[base + a + rng.integers(1, H)] = True
        if rng.random() < 0.3:  # falls at exactly H, H+1, ..., H+5
            t[base + H + rng.integers(0, 6)] = True
        if rng.random() < 0.2:  # fall right before seam
            t[base + steps - 2] = True
    return t


CASES = [(s, ne, st, p) for s in range(6) for (ne, st) in ((16, 1000), (8, 2000)) for p in (0.001, 0.004, 0.02)]


# ======================================================================================
# Index selection
# ======================================================================================
@pytest.mark.parametrize("seed,num_envs,steps,p", CASES)
def test_select_indices_matches_scorer(seed, num_envs, steps, p):
    term = synth_terms(seed, num_envs, steps, p)
    ref = ref_scorer_loop(term, num_envs, steps, H)
    got = es.select_indices(term, num_envs, steps, H)
    assert got.fall_idx == ref[0]
    assert got.pre1_idx == ref[1]
    assert got.pre5_idx == ref[2]
    assert got.neg_cand == ref[3]
    assert (got.raw_terms, got.seam_terms, got.interior_terms) == ref[4:7]
    for num_neg in (5000, 100, len(ref[3]) + 7):
        ref_neg = np.random.default_rng(0).choice(ref[3], size=min(num_neg, len(ref[3])), replace=False)
        got_neg = es.sample_scorer_negatives(got.neg_cand, num_neg)
        assert got_neg.dtype == ref_neg.dtype
        assert np.array_equal(got_neg, ref_neg)
    # the scorer's rng object is created after the loop; first draw must be the sample
    ref_neg = ref[7].choice(ref[3], size=min(5000, len(ref[3])), replace=False)
    assert np.array_equal(es.sample_scorer_negatives(got.neg_cand, 5000), ref_neg)


def test_select_indices_has_edge_cases():
    """Make sure the synthetic generator actually exercises the edge cases."""
    n_seam = n_dropped_hist = n_pre_drop = 0
    for seed, ne, st, p in CASES:
        term = synth_terms(seed, ne, st, p)
        s = es.select_indices(term, ne, st, H)
        n_seam += s.seam_terms
        n_dropped_hist += s.interior_terms - len(s.fall_idx)
        n_pre_drop += len(s.fall_idx) - len(s.pre5_idx)
    assert n_seam > 0 and n_dropped_hist > 0 and n_pre_drop > 0


def test_select_indices_64_envs():
    term = synth_terms(123, 64, 1000, 0.003)
    ref = ref_scorer_loop(term, 64, 1000, H)
    got = es.select_indices(term, 64, 1000, H)
    assert (got.fall_idx, got.pre1_idx, got.pre5_idx, got.neg_cand) == ref[:4]


@pytest.mark.parametrize("seed,num_envs,steps,p", CASES)
def test_lead_sets(seed, num_envs, steps, p):
    term = synth_terms(seed, num_envs, steps, p)
    s = es.select_indices(term, num_envs, steps, H)
    ks = [0, 1, 2, 3, 5, 10, 15, 20, 25, 32, 33, 34, 50, 80]
    lead = es.lead_time_sets(term, s.fall_idx, steps, H, ks)
    assert np.array_equal(lead[0], np.asarray(s.fall_idx))
    assert np.array_equal(lead[1], np.asarray(s.pre1_idx))
    assert np.array_equal(lead[5], np.asarray(s.pre5_idx))
    for k in ks[1:]:
        # spec formula without the same-episode guard
        spec = [f - k for f in s.fall_idx
                if (f % steps) - k >= H and term[f - H - k:f - k + 1].sum() == 0]
        if k <= H + 1:
            assert np.array_equal(lead[k], np.asarray(spec, dtype=np.int64))
        else:
            assert set(lead[k].tolist()) <= set(spec)
        for r in lead[k]:
            f = r + k
            assert r // steps == f // steps  # same env block
            assert not term[r] and term[f]
            assert term[r - H:f].sum() == 0  # window clean and same episode as the fall
            assert (r % steps) >= H


def test_lead_guard_excludes_previous_episode():
    steps = 300
    term = np.zeros(steps, dtype=bool)
    term[200] = True  # earlier episode ends here
    term[240] = True  # fall: hist window 208..239 clean, so it is a valid fall
    s = es.select_indices(term, 1, steps, H)
    assert 240 in s.fall_idx
    lead = es.lead_time_sets(term, s.fall_idx, steps, H, [50])
    # spec formula alone would accept row 190 (rows 158..190 clean) although it belongs to the
    # episode that ended at row 200
    assert term[240 - H - 50:240 - 50 + 1].sum() == 0
    assert 190 not in lead[50].tolist()


@pytest.mark.parametrize("seed,num_envs,steps,p", CASES[::3])
@pytest.mark.parametrize("censor", [True, False])
def test_clean_negatives(seed, num_envs, steps, p, censor):
    term = synth_terms(seed, num_envs, steps, p)
    s = es.select_indices(term, num_envs, steps, H)
    gap = 25
    cand = es.clean_negative_candidates(term, s.neg_cand, steps, gap, censor)
    assert set(cand.tolist()) <= set(s.neg_cand)
    for f in cand:
        end = (f // steps + 1) * steps
        assert not term[f + 1:min(f + gap + 1, end)].any()
        if censor:
            assert f + gap < end
    # every excluded candidate has a reason
    excl = set(s.neg_cand) - set(cand.tolist())
    for f in list(excl)[:500]:
        end = (f // steps + 1) * steps
        assert term[f + 1:min(f + gap + 1, end)].any() or (censor and f + gap >= end)
    samp = es.sample_clean_negatives(cand, 300, seed=0)
    assert np.array_equal(samp, np.random.default_rng(1).choice(cand, size=min(300, len(cand)), replace=False))
    assert len(set(samp.tolist())) == len(samp)


# ======================================================================================
# AUC
# ======================================================================================
def test_auc_scorer_equals_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        pos = rng.normal(0.5, 1, rng.integers(1, 200))
        neg = rng.normal(0, 1, rng.integers(1, 500))
        if rng.random() < 0.5:
            pos, neg = np.round(pos, 1), np.round(neg, 1)  # ties
        assert es.auc_scorer(pos, neg) == ref_auc(pos, neg)


def test_auc_tie_equals_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    for i in range(100):
        pos = rng.normal(0.3, 1, rng.integers(1, 300))
        neg = rng.normal(0, 1, rng.integers(1, 600))
        if i % 2:
            pos, neg = np.round(pos), np.round(neg)
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        ref = roc_auc_score(y, np.r_[pos, neg])
        assert es.auc_tie(pos, neg) == pytest.approx(ref, abs=1e-12)
    # all-tied -> 0.5 (scorer's version is order-dependent here)
    assert es.auc_tie(np.zeros(5), np.zeros(7)) == 0.5
    assert np.isnan(es.auc_tie([], [1.0]))
    assert np.isnan(es.auc_tie([np.nan], [1.0]))


# ======================================================================================
# Bootstrap
# ======================================================================================
def _sep_data(seed=0, num_envs=16, shift=2.0):
    rng = np.random.default_rng(seed)
    pos_env = rng.integers(0, num_envs, 120)
    neg_env = rng.integers(0, num_envs, 2000)
    pos = rng.normal(shift, 1, len(pos_env))
    neg = rng.normal(0, 1, len(neg_env))
    return pos, pos_env, neg, neg_env, num_envs


def test_bootstrap_deterministic_and_brackets():
    pos, pe, neg, ne, E = _sep_data()
    c1 = es.bootstrap_env_counts(E, 500, seed=0)
    c2 = es.bootstrap_env_counts(E, 500, seed=0)
    c3 = es.bootstrap_env_counts(E, 500, seed=1)
    assert np.array_equal(c1, c2) and not np.array_equal(c1, c3)
    assert np.all(c1.sum(axis=1) == E)
    m1 = es.compute_metrics(pos, pe, neg, ne, E, c1)
    m2 = es.compute_metrics(pos, pe, neg, ne, E, c2)
    assert m1 == m2
    assert m1["ci_lo"] <= m1["auc"] <= m1["ci_hi"]
    assert m1["ci_lo"] > 0.8 and m1["ci_hi"] <= 1.0
    assert m1["boot_valid"] + m1["boot_skipped"] == 500
    assert m1["auc_scorer"] == ref_auc(pos, neg)


def test_bootstrap_matrix_equals_explicit_resampling():
    from sklearn.metrics import roc_auc_score

    pos, pe, neg, ne, E = _sep_data(seed=3, shift=0.4)
    pos, neg = np.round(pos, 1), np.round(neg, 1)  # include ties
    counts = es.bootstrap_env_counts(E, 30, seed=7)
    vals, skipped = es.bootstrap_auc(pos, pe, neg, ne, E, counts)
    explicit = []
    for c in counts:
        wp = np.repeat(np.arange(len(pos)), c[pe].astype(int))
        wn = np.repeat(np.arange(len(neg)), c[ne].astype(int))
        if len(wp) == 0 or len(wn) == 0:
            continue
        y = np.r_[np.ones(len(wp)), np.zeros(len(wn))]
        explicit.append(roc_auc_score(y, np.r_[pos[wp], neg[wn]]))
    assert skipped == 30 - len(explicit)
    np.testing.assert_allclose(vals, explicit, rtol=0, atol=1e-12)


def test_bootstrap_skips_replicates_without_positives():
    E = 64
    pos = np.array([1.0, 2.0])
    pe = np.array([3, 3])  # all positives in one env
    neg = np.linspace(0, 1.5, 500)
    ne = np.arange(500) % E
    counts = es.bootstrap_env_counts(E, 400, seed=0)
    m = es.compute_metrics(pos, pe, neg, ne, E, counts)
    expected_skip = int((counts[:, 3] == 0).sum())
    assert expected_skip > 0 and m["boot_skipped"] == expected_skip
    assert m["boot_valid"] == 400 - expected_skip


# ======================================================================================
# kNN
# ======================================================================================
def brute_knn(q, t, ks):
    d = np.sqrt(((q[:, None, :] - t[None, :, :]) ** 2).sum(-1))
    d.sort(axis=1)
    return {k: d[:, :k].mean(axis=1) for k in ks}


@needs_torch
@pytest.mark.parametrize("qb,tc", [(7, 13), (5, 3), (1000, 1000), (16, 64)])
def test_knn_chunked_equals_bruteforce(qb, tc):
    rng = np.random.default_rng(0)
    t = rng.normal(size=(301, 9))
    q = np.r_[rng.normal(size=(40, 9)), t[:5] + 1e-9, t[5:7]]  # near-duplicates and exact duplicates
    ks = [1, 5, 10, 50]
    got = es.knn_mean_distances(q, t, ks, "cpu", query_batch=qb, train_chunk=tc)
    ref = brute_knn(q, t, ks)
    for k in ks:
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-12, atol=1e-12)


def test_pair_and_query_features_offset():
    N = 20
    state = np.tile(np.arange(N, dtype=np.float32)[:, None], (1, 45))
    action = np.tile(1000 + np.arange(N, dtype=np.float32)[:, None], (1, 12))
    f1 = es.pair_features(state, action, 1, "state_action")
    assert f1.shape == (N - 1, 57)
    assert np.all(f1[:, 0] == np.arange(N - 1)) and np.all(f1[:, 45] == 1000 + np.arange(1, N))
    f0 = es.pair_features(state, action, 0, "state_action")
    assert f0.shape == (N, 57) and np.all(f0[:, 45] == 1000 + np.arange(N))
    fs = es.pair_features(state, action, 1, "state")
    assert fs.shape == (N, 45)
    idx = np.array([5, 10, 19])
    q1 = es.query_features(state, action, idx, 1, "state_action")
    assert np.all(q1[:, 0] == idx - 1) and np.all(q1[:, 45] == 1000 + idx)  # state f-1, action f
    q0 = es.query_features(state, action, idx, 0, "state_action")
    assert np.all(q0[:, 45] == 1000 + idx - 1)
    qs = es.query_features(state, action, idx, 1, "state")
    assert qs.shape == (3, 45) and np.all(qs[:, 0] == idx - 1)
    # a query feature of the trace equals the training feature of the same row pair
    assert np.array_equal(q1[0], f1[4])
    with pytest.raises(ValueError):
        es.query_features(state, action, np.array([19]), 2, "state_action")


def test_read_csv_and_train_file_resolution(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 66)).astype(np.float32)
    pa = tmp_path / "a.csv"
    pd.DataFrame(a).to_csv(pa, header=False, index=False)
    ref = pd.read_csv(pa, header=None).values.astype(np.float32)
    assert np.array_equal(es.read_numeric_csv(str(pa)), ref)
    pd.DataFrame({"name": ["a"], "steps": [1], "rows": [50]}).to_csv(tmp_path / "manifest.csv", index=False)
    os.symlink(pa, tmp_path / "b_link.csv")
    (tmp_path / "empty.csv").write_text("")
    files = es.resolve_train_files([str(tmp_path / "*.csv")])
    assert files == [str(pa)]  # manifest skipped, symlink duplicate skipped, empty skipped
    # non-numeric cell coerced to NaN, dropped from features
    b = a.copy().astype(object)
    b[3, 7] = "oops"
    pb = tmp_path / "sub"
    pb.mkdir()
    pd.DataFrame(b).to_csv(pb / "b.csv", header=False, index=False)
    x = es.read_numeric_csv(str(pb / "b.csv"))
    assert np.isnan(x[3, 7]) and x.dtype == np.float32
    feats, info = es.load_train_features([str(pb / "b.csv")], 1, "state_action")
    assert len(feats) == 49 - 1 and info[0]["dropped_nonfinite"] == 1


# ======================================================================================
# WM parity (scorer path) + end-to-end
# ======================================================================================
def make_wm_ckpt(path, seed=0, normalized=True):
    from rsl_rl.modules import SystemDynamicsEnsemble
    from configs.go2_flat_cfg import Go2FlatConfig

    torch.manual_seed(seed)
    device = "cpu"
    cfg = Go2FlatConfig()
    mac = cfg.model_architecture_config
    Hc = getattr(mac, "history_horizon", 32)
    sd = SystemDynamicsEnsemble(
        45, 12, getattr(mac, "extension_dim", 0), getattr(mac, "contact_dim", 8),
        getattr(mac, "termination_dim", 1), device,
        ensemble_size=getattr(mac, "ensemble_size", None) or getattr(mac, "num_models", None) or 5,
        history_horizon=Hc, architecture_config=mac.architecture_config, freeze_auxiliary=False,
    ).to(device)
    # make heads differ more than default init so epi has spread; termination bias towards ~0.5
    rng = np.random.default_rng(seed)
    ck = {"system_dynamics_state_dict": sd.state_dict(), "iter": 7, "normalized": normalized}
    if normalized:
        ck.update(
            state_mean=torch.tensor(rng.normal(0, 0.5, 45), dtype=torch.float32),
            state_std=torch.tensor(rng.uniform(0.5, 2.0, 45), dtype=torch.float32),
            action_mean=torch.tensor(rng.normal(0, 0.5, 12), dtype=torch.float32),
            action_std=torch.tensor(rng.uniform(0.5, 2.0, 12), dtype=torch.float32),
        )
    torch.save(ck, path)
    return ck


def make_trace(path, num_envs=4, steps=150, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(num_envs * steps, 66)).astype(np.float32)
    # smooth random walk so windows look like trajectories
    data[:, :57] = np.cumsum(data[:, :57] * 0.1, axis=0)
    data[:, 57:65] = (rng.random((num_envs * steps, 8)) < 0.5).astype(np.float32)
    data[:, 65] = 0.0
    term_rows = [0 * steps + 100, 0 * steps + 120, 1 * steps + 40, 1 * steps + 90,
                 2 * steps + steps - 1, 3 * steps + 70, 3 * steps + 140]
    data[term_rows, 65] = 1.0
    pd.DataFrame(data).to_csv(path, header=False, index=False, float_format="%.7g")
    return data


def ref_scorer_scores(wm_path, trace_path, num_envs, steps_per_env, num_neg, device="cpu"):
    """Copy of the scorer's main() up to the score arrays."""
    from rsl_rl.modules import SystemDynamicsEnsemble
    from configs.go2_flat_cfg import Go2FlatConfig

    cfg = Go2FlatConfig()
    mac = cfg.model_architecture_config
    H = getattr(mac, "history_horizon", 32)
    ckpt = torch.load(wm_path, map_location=device)
    sd = SystemDynamicsEnsemble(
        45, 12, getattr(mac, "extension_dim", 0), getattr(mac, "contact_dim", 8),
        getattr(mac, "termination_dim", 1), device,
        ensemble_size=getattr(mac, "ensemble_size", None) or getattr(mac, "num_models", None) or 5,
        history_horizon=H, architecture_config=mac.architecture_config, freeze_auxiliary=False,
    ).to(device)
    sd.load_state_dict(ckpt["system_dynamics_state_dict"], strict=True)
    sd.eval()
    data = pd.read_csv(trace_path, header=None).values.astype(np.float32)
    state_all = np.ascontiguousarray(data[:, 0:45])
    action_all = np.ascontiguousarray(data[:, 45:57])
    term_all = np.ascontiguousarray(data[:, 65] > 0.5)
    if ckpt.get("normalized", False) and "state_mean" in ckpt:
        sm = ckpt["state_mean"].detach().cpu().numpy().reshape(-1)
        ss = ckpt["state_std"].detach().cpu().numpy().reshape(-1)
        am = ckpt["action_mean"].detach().cpu().numpy().reshape(-1)
        astd = ckpt["action_std"].detach().cpu().numpy().reshape(-1)
        state_all = ((state_all - sm) / ss).astype(np.float32)
        action_all = ((action_all - am) / astd).astype(np.float32)
    fall_idx, pre1_idx, pre5_idx, neg_cand, *_ , rng = ref_scorer_loop(term_all, num_envs, steps_per_env, H)
    neg_idx = rng.choice(neg_cand, size=min(num_neg, len(neg_cand)), replace=False)

    def window(f):
        xs = torch.from_numpy(state_all[f - H:f]).to(device).unsqueeze(0)
        xa = torch.from_numpy(action_all[f - H + 1:f + 1]).to(device).unsqueeze(0)
        return xs, xa

    @torch.no_grad()
    def score_one(f):
        sd.reset()
        xs, xa = window(int(f))
        state_pred, alea, epi, ext, contact, term = sd.forward(xs, xa)
        epi_v = float(epi.detach().cpu().reshape(-1)[0].item())
        alea_v = float(alea.detach().cpu().reshape(-1)[0].item())
        term_p = float(torch.sigmoid(term).detach().cpu().reshape(-1)[0].item()) if term is not None else float("nan")
        return epi_v, alea_v, term_p

    def score_many(idx):
        return np.array([score_one(f) for f in idx], dtype=np.float64)

    return {
        "fall": (fall_idx, score_many(fall_idx)),
        "neg": (neg_idx, score_many(neg_idx)),
        "pre1": (pre1_idx, score_many(pre1_idx) if len(pre1_idx) else np.zeros((0, 3))),
        "pre5": (pre5_idx, score_many(pre5_idx) if len(pre5_idx) else np.zeros((0, 3))),
    }


@pytest.fixture(scope="module")
def wm_setup(tmp_path_factory):
    if not HAVE_RSL:
        pytest.skip("torch/rsl_rl not importable")
    d = tmp_path_factory.mktemp("wm")
    wm = str(d / "wm.pt")
    trace = str(d / "trace.csv")
    make_wm_ckpt(wm)
    make_trace(trace)
    train_dir = d / "train"
    train_dir.mkdir()
    rng = np.random.default_rng(5)
    for i in range(3):
        x = rng.normal(size=(200 + 37 * i, 66)).astype(np.float32)
        x[:, :57] = np.cumsum(x[:, :57] * 0.1, axis=0)
        pd.DataFrame(x).to_csv(train_dir / f"seg_{i}.csv", header=False, index=False)
    pd.DataFrame({"name": ["seg_0"], "steps": [1], "rows": [200]}).to_csv(train_dir / "manifest.csv", index=False)
    return d, wm, trace, str(train_dir / "*.csv")


def _run(out, wm=None, trace=None, knn=None, extra=()):
    argv = ["--trace", trace, "--num_envs", "4", "--steps_per_env", "150", "--device", "cpu",
            "--num_neg", "60", "--bootstrap", "200", "--out", str(out)]
    if wm:
        argv += ["--wm", wm]
    if knn:
        argv += ["--knn_train", knn, "--knn_name", "synthetic", "--knn_k", "1,5,10,50",
                 "--knn_query_batch", "17", "--knn_train_chunk", "101"]
    argv += list(extra)
    return es.run(es.build_parser().parse_args(argv))


@needs_rsl
def test_wm_parity_with_scorer_logic(wm_setup):
    d, wm, trace, knn = wm_setup
    out = d / "out_wm"
    summary = _run(out, wm=wm, trace=trace)
    ref = ref_scorer_scores(wm, trace, 4, 150, 60)
    z = np.load(out / "scores.npz")
    pos = {int(f): i for i, f in enumerate(z["idx"])}
    sig = np.stack([z["sig_epi"], z["sig_alea"], z["sig_term"]], axis=1)
    n_checked = 0
    for name, (idx, vals) in ref.items():
        assert len(idx) > 0, name
        got = sig[[pos[int(f)] for f in idx]]
        np.testing.assert_allclose(got, vals, rtol=0, atol=0)
        n_checked += len(idx)
    assert n_checked > 60
    # set memberships match the scorer
    assert np.array_equal(z["idx"][z["in_fall"]], np.sort(ref["fall"][0]))
    assert np.array_equal(z["neg_scorer_order"], ref["neg"][0])
    # metrics: auc_scorer equals the scorer's AUC on the same arrays
    m = summary["metrics"]["epi"]
    assert m["pre5"]["scorer"]["auc_scorer"] == ref_auc(ref["pre5"][1][:, 0], ref["neg"][1][:, 0])
    assert m["fall"]["scorer"]["auc_scorer"] == ref_auc(ref["fall"][1][:, 0], ref["neg"][1][:, 0])
    assert summary["metrics"]["term"]["fall"]["scorer"]["auc_scorer"] == ref_auc(ref["fall"][1][:, 2],
                                                                                  ref["neg"][1][:, 2])


@needs_rsl
def test_scorer_stdout_parity_subprocess(wm_setup):
    """Run the real scorer script and diff its stdout against scorer_parity.txt."""
    d, wm, trace, knn = wm_setup
    out = d / "out_parity"
    _run(out, wm=wm, trace=trace)
    scorer = os.path.join(MODEL_BASED_DIR, "score_go2_exploit_trace_uncertainty.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([RSL_RL_PATH, MODEL_BASED_DIR, env.get("PYTHONPATH", "")])
    res = subprocess.run(
        [sys.executable, scorer, "--wm", wm, "--trace", trace, "--num_envs", "4", "--steps_per_env", "150",
         "--num_neg", "60", "--device", "cpu"],
        cwd=MODEL_BASED_DIR, env=env, capture_output=True, text=True, check=True)
    parity = (out / "scorer_parity.txt").read_text()
    assert res.stdout == parity
    assert "ROC-AUC epi(pre5 > walking)" in parity


@needs_rsl
def test_end_to_end_knn_and_outputs(wm_setup):
    d, wm, trace, knn = wm_setup
    out = d / "out_knn"
    summary = _run(out, wm=wm, trace=trace, knn=knn)
    z = np.load(out / "scores.npz")
    # independent brute-force kNN on the same eval indices
    data = pd.read_csv(trace, header=None).values.astype(np.float32)
    files = sorted(f for f in glob.glob(knn) if not f.endswith("manifest.csv"))
    train = []
    for f in files:
        x = pd.read_csv(f, header=None).values.astype(np.float32).astype(np.float64)
        train.append(np.c_[x[:-1, :45], x[1:, 45:57]])
    train = np.concatenate(train)
    mu, sd = train.mean(0), np.maximum(train.std(0), 1e-6)
    idx = z["idx"]
    q = np.c_[data[idx - 1, :45].astype(np.float64), data[idx, 45:57].astype(np.float64)]
    ref = brute_knn((q - mu) / sd, (train - mu) / sd, [1, 5, 10, 50])
    for k in (1, 5, 10, 50):
        np.testing.assert_allclose(z[f"sig_knn{k}"], ref[k], rtol=1e-10, atol=1e-12)
    assert summary["knn"]["n_train"] == len(train)
    assert len(summary["knn"]["files"]) == 3
    # summary.json is valid strict JSON with the expected structure
    with open(out / "summary.json") as fh:
        js = json.loads(fh.read(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    for s in ("epi", "alea", "term", "knn1", "knn5", "knn10", "knn50"):
        for p in ("fall", "pre1", "pre5", "lead0", "lead1", "lead5", "lead50"):
            for n in ("scorer", "clean"):
                m = js["metrics"][s][p][n]
                assert {"auc", "auc_scorer", "ratio", "ci_lo", "ci_hi", "n_pos", "n_neg",
                        "boot_skipped", "boot_valid"} <= set(m)
    assert js["counts"]["n_fall"] == int(z["in_fall"].sum())
    assert js["inputs"]["trace"]["bytes"] == os.path.getsize(trace)
    # metrics of lead1/lead5 identical to pre1/pre5
    for s in js["metrics"]:
        assert js["metrics"][s]["lead1"] == js["metrics"][s]["pre1"]
        assert js["metrics"][s]["lead5"] == js["metrics"][s]["pre5"]
        assert js["metrics"][s]["lead0"] == js["metrics"][s]["fall"]
    # clean negatives sampled with seed+1 and never within clean_gap before a termination
    term = data[:, 65] > 0.5
    for f in z["neg_clean_order"]:
        end = (f // 150 + 1) * 150
        assert f + 25 < end and not term[f + 1:f + 26].any()


@needs_rsl
def test_knn_variants_run(wm_setup):
    d, wm, trace, knn = wm_setup
    s0 = _run(d / "v0", trace=trace, knn=knn, extra=["--knn_pair_offset", "0"])
    s1 = _run(d / "v1", trace=trace, knn=knn, extra=["--knn_features", "state"])
    s2 = _run(d / "v2", wm=wm, trace=trace, knn=knn, extra=["--knn_norm", "wm"])
    assert s0["knn"]["dim"] == 57 and s1["knn"]["dim"] == 45 and s2["knn"]["norm"] == "wm"
    assert "epi" not in s0["signals"] and "knn10" in s0["signals"]
    with pytest.raises(SystemExit):
        _run(d / "v3", trace=trace, knn=knn, extra=["--knn_norm", "wm"])


@needs_rsl
def test_summarizer(wm_setup):
    d, wm, trace, knn = wm_setup
    root = d / "results"
    _run(root / "curated_t9", wm=wm, trace=trace, knn=knn)
    _run(root / "plusfail_t9", wm=wm, trace=trace)
    import summarize_p1_eval as sp

    txt = sp.summarize(str(root))
    assert "curated_t9" in txt and "plusfail_t9" in txt and "pre5" in txt
