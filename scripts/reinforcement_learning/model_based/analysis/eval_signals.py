#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate every candidate pre-fall signal on IDENTICAL index sets of a Go2 exploit trace.

This script supersedes the ad-hoc numbers printed by
``score_go2_exploit_trace_uncertainty.py`` ("the scorer") while reproducing them
exactly: the scorer's index sets, negative sample, WM forward path and AUC function
are copied verbatim, and its headline lines are printed (and written to
``scorer_parity.txt``) so that a plain ``diff`` against the scorer's stdout is the
parity check.

Data layout
-----------
A trace is a headerless 66-column float CSV::

    state 0:45   (base_lin_vel 0:3, base_ang_vel 3:6, projected_gravity 6:9,
                  joint_pos 9:21, joint_vel 21:33, joint_torque 33:45)
    action 45:57, contact 57:65, termination 65 (> 0.5 means terminal)

Traces are env-major: ``num_envs`` consecutive blocks of ``steps_per_env`` rows; row
``f`` belongs to env ``f // steps_per_env`` at local time ``f % steps_per_env``. The
last row of each block is the "seam" (end of recording, not necessarily a fall).

Index convention
----------------
Every evaluated *index* ``f`` is a trace row, interpreted as the transition that ends
in row ``f``: the WM sees state rows ``f-H .. f-1`` and action rows ``f-H+1 .. f``
(``window(f)``, copied from the scorer) and predicts state row ``f``. ``term[f]``
means "this transition ends in a failure". ``H`` is ``history_horizon`` of
``Go2FlatConfig`` (32).

Index sets
----------
Scorer sets (primary; bit-identical to the scorer, see ``select_indices``). For each
env block, for ``local_t`` in ``[H, steps_per_env)``, with ``f = base + local_t``:

* ``hist_clean(f)``: no termination in rows ``f-H .. f-1``.
* ``fall``: ``term[f]`` and ``f`` is not the seam and ``hist_clean(f)``.
* ``pre1``: row ``f-1`` for every fall ``f`` with ``local_t-1 >= H`` and no
  termination in rows ``f-H-1 .. f-1``.
* ``pre5``: row ``f-5`` for every fall ``f`` with ``local_t-5 >= H`` and no
  termination in rows ``f-H-5 .. f-5``.
* ``neg_cand``: ``not term[f]`` and ``hist_clean(f)`` (this includes seam rows that
  are not terminal, and also pre-fall rows such as the pre1/pre5 rows).
* ``scorer`` negatives: ``np.random.default_rng(0).choice(neg_cand,
  size=min(num_neg, len(neg_cand)), replace=False)``. The seed 0 is hard-coded as in
  the scorer and does NOT follow ``--seed``.

Secondary sets:

* ``lead{k}`` (lead-time positives, one per ``k`` in ``--lead_ks``): row ``f-k`` for
  every fall ``f`` such that (i) ``local_t-k >= H``, (ii) no termination in rows
  ``f-H-k .. f-k`` (the generalisation of the pre1/pre5 condition:
  ``term[f-H-k : f-k+1].sum() == 0``), and (iii) no termination in rows
  ``f-k+1 .. f-1`` (same-episode guard). ``lead0`` is the fall set itself. For
  ``k <= H+1`` condition (iii) is implied by ``hist_clean(f)``, so ``lead1 == pre1``
  and ``lead5 == pre5`` exactly (asserted at runtime); (iii) only matters for
  ``k > H+1`` where it prevents labelling a row from an *earlier* episode (one that
  ended at an interior termination between ``f-k`` and ``f``) as a lead-k positive
  of the later fall.
* ``clean`` negatives: candidates from ``neg_cand`` with no termination (interior or
  seam) in rows ``f+1 .. f+clean_gap`` of the same env block. With
  ``--clean_censor_end 1`` (default) candidates whose look-ahead window runs past the
  end of their block (``local_t + clean_gap >= steps_per_env``) are also dropped,
  because their future is unobserved. Sampled with
  ``np.random.default_rng(seed + 1)``, size ``min(num_neg, len(candidates))``.

Signals (all oriented so that larger = more anomalous)
------------------------------------------------------
* ``epi``, ``alea``, ``term`` (``--wm``): per-sample forward of the
  ``SystemDynamicsEnsemble`` exactly as in the scorer: same construction, strict
  checkpoint load, same normalisation (ckpt keys ``normalized``, ``state_mean``,
  ``state_std``, ``action_mean``, ``action_std``), ``sd.reset()`` before each sample,
  ``window(f)`` input, batch size 1. ``epi`` = sum over state dims of the across-head
  std of the predicted means; ``alea`` = sum over dims of the head-mean predicted std;
  ``term`` = sigmoid of the head-mean termination logit.
* ``term_logit`` (``--wm``): the head-mean termination logit from the same forward
  pass; same ranking as ``term`` but without float32 saturation ties at p = 1.0.
* ``knn{k}`` (``--knn_train``): mean Euclidean distance from the query feature to its
  ``k`` nearest training features, in a normalised feature space.

  - Query feature of index ``f``: ``[state row f-1, action row f-1+offset]``
    (``--knn_pair_offset``, default 1: the last state of the WM window paired with the
    last action of the window). ``--knn_features state``: state row ``f-1`` only.
  - Training features, per training file with ``N`` rows: ``[state r, action
    r+offset]`` for ``r = 0 .. N-1-offset``; state-only features use every row
    ``r = 0 .. N-1``. Pairs that straddle an env-block boundary inside a training file
    are kept (deliberately not handled): one per block boundary, e.g. 63 of 64k rows
    for a 64 x 1000 file, i.e. ~0.1 %. Pairs straddling an episode reset inside a
    block (terminal state + first action of the next episode) are kept as well; they
    are logged transitions like any other.
  - ``--knn_norm own``: z-score with mean/std (ddof=0) of the training features, std
    floored at 1e-6 (the number of floored dims is printed). ``--knn_norm wm``: the
    WM checkpoint's state/action mean/std (requires ``--wm``).
  - Exact distances: float64 ``torch.cdist`` over (query batch x training chunk)
    tiles with a running top-k (``torch.topk``), then the selected neighbours'
    distances are recomputed directly from coordinate differences.

Caveat: the ``fall`` / ``lead0`` set is contaminated
-------------------------------------------------
IsaacLab resets a terminated env before computing observations, and the reset zeroes
the last action. So a logged termination row ``f`` holds the *spawn* state of the next
episode and an all-zero action. ``window(f)`` ends with that zero action, and the kNN
query of ``f`` uses it too, so any "fall" AUC partly measures "is the last action zero"
(code review 1.2, 2026-10-08). Use ``pre1`` / ``pre5`` / ``lead{k>=1}`` as the primary
sets; ``fall`` is kept only for parity with the scorer.

Metrics (per signal x positive set x negative set)
--------------------------------------------------
* ``auc_scorer``: the scorer's AUC verbatim (argsort ranks, ties broken by sort order,
  no tie averaging) -- for parity only.
* ``auc``: tie-aware Mann-Whitney AUC = P(pos > neg) + 0.5 P(pos == neg), i.e. the
  average-rank AUC (equals ``sklearn.metrics.roc_auc_score``). Primary going forward.
* ``ratio``: ``mean(pos) / max(mean(neg), 1e-12)`` (scorer formula).
* ``ci_lo``, ``ci_hi``: 95 % percentile bootstrap CI of ``auc`` resampling env blocks
  (trajectories) with replacement: each of ``--bootstrap`` replicates draws
  ``num_envs`` env ids with replacement (rng ``default_rng(--seed)``, the same draws
  for every signal/set so comparisons are paired); positives and negatives of a drawn
  env enter the replicate with multiplicity equal to its draw count. Replicates with
  zero positives or zero negatives are skipped and counted (``boot_skipped``).
  Computed exactly via a per-env pairwise-comparison matrix ``M[e1, e2]`` so that
  ``auc_b = c_b^T M c_b / ((c_b . npos_e) (c_b . nneg_e))``.
* ``n_pos``, ``n_neg``, ``mean_pos``, ``mean_neg``.

Outputs (``--out``)
-------------------
``summary.json`` (args, git commit, inputs with sizes/sha256, counts, metrics),
``scores.npz`` (per evaluated index: row, env, local_t, set memberships, signals),
``scorer_parity.txt`` (the scorer-format lines when ``--wm`` is given).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - torch is required at runtime, optional for some tests
    torch = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_BASED_DIR = os.path.dirname(_THIS_DIR)
if _MODEL_BASED_DIR not in sys.path:  # so `configs` imports like in the scorer
    sys.path.insert(0, _MODEL_BASED_DIR)

STATE_DIM = 45
ACTION_DIM = 12
NUM_COLS = 66
TERM_COL = 65
SCORER_NEG_SEED = 0  # hard-coded in the scorer


# --------------------------------------------------------------------------------------
# CSV reading
# --------------------------------------------------------------------------------------
def read_numeric_csv(path: str) -> np.ndarray:
    """``pd.read_csv(header=None)``, coerce non-numeric columns, return float32.

    For an all-numeric file this is exactly the scorer's
    ``pd.read_csv(path, header=None).values.astype(np.float32)``.
    """
    df = pd.read_csv(path, header=None)
    obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if obj_cols:
        print(f"[warn] {path}: coercing {len(obj_cols)} non-numeric column(s) to numeric")
        df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors="coerce")
    return df.values.astype(np.float32)


def csv_num_cols(path: str) -> int:
    try:
        return int(pd.read_csv(path, header=None, nrows=1).shape[1])
    except pd.errors.EmptyDataError:
        return 0


def resolve_train_files(patterns: Sequence[str]) -> List[str]:
    """Expand globs (sorted), drop duplicates by realpath and files with != 66 columns."""
    files: List[str] = []
    seen: Dict[str, str] = {}
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches:
            print(f"[warn] knn_train pattern matched no files: {pat}")
        for p in matches:
            if os.path.isdir(p):
                continue
            rp = os.path.realpath(p)
            if rp in seen:
                print(f"[warn] skipping duplicate training file {p} (same file as {seen[rp]})")
                continue
            ncols = csv_num_cols(p)
            if ncols != NUM_COLS:
                print(f"[warn] skipping {p}: {ncols} columns != {NUM_COLS} (e.g. manifest.csv)")
                continue
            seen[rp] = p
            files.append(p)
    return files


# --------------------------------------------------------------------------------------
# Index selection
# --------------------------------------------------------------------------------------
@dataclass
class ScorerIndexSets:
    fall_idx: List[int]
    pre1_idx: List[int]
    pre5_idx: List[int]
    neg_cand: List[int]
    raw_terms: int
    seam_terms: int
    interior_terms: int


def select_indices(term_all: np.ndarray, num_envs: int, steps_per_env: int, H: int) -> ScorerIndexSets:
    """The scorer's selection loop, verbatim (only ``args.`` removed)."""
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

    return ScorerIndexSets(fall_idx, pre1_idx, pre5_idx, neg_cand, raw_terms, seam_terms, interior_terms)


def sample_scorer_negatives(neg_cand: Sequence[int], num_neg: int) -> np.ndarray:
    """Exactly the scorer's negative sample (seed 0 hard-coded)."""
    rng = np.random.default_rng(SCORER_NEG_SEED)
    return rng.choice(neg_cand, size=min(num_neg, len(neg_cand)), replace=False)


def lead_time_sets(
    term_all: np.ndarray, fall_idx: Sequence[int], steps_per_env: int, H: int, ks: Sequence[int]
) -> Dict[int, np.ndarray]:
    """``lead{k}`` positives (see module docstring). ``k = 0`` returns the fall set."""
    out: Dict[int, np.ndarray] = {}
    for k in ks:
        if k < 0:
            raise ValueError(f"lead k must be >= 0, got {k}")
        if k == 0:
            out[k] = np.asarray(fall_idx, dtype=np.int64)
            continue
        idx = []
        for f in fall_idx:
            local_t = f % steps_per_env
            if local_t - k < H:
                continue
            if term_all[f - H - k:f - k + 1].sum() != 0:  # pre-k window + row clean (scorer rule)
                continue
            if term_all[f - k + 1:f].sum() != 0:  # same episode as the fall (only binds for k > H+1)
                continue
            idx.append(f - k)
        out[k] = np.asarray(idx, dtype=np.int64)
    return out


def clean_negative_candidates(
    term_all: np.ndarray, neg_cand: Sequence[int], steps_per_env: int, clean_gap: int, censor_end: bool = True
) -> np.ndarray:
    """Subset of ``neg_cand`` with no termination in rows f+1..f+clean_gap of the same block."""
    keep = []
    for f in neg_cand:
        f = int(f)
        block_end = (f // steps_per_env + 1) * steps_per_env  # exclusive
        if censor_end and f + clean_gap >= block_end:
            continue
        if term_all[f + 1:min(f + clean_gap + 1, block_end)].any():
            continue
        keep.append(f)
    return np.asarray(keep, dtype=np.int64)


def sample_clean_negatives(clean_cand: np.ndarray, num_neg: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    return rng.choice(clean_cand, size=min(num_neg, len(clean_cand)), replace=False)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def auc_scorer(pos, neg):
    """The scorer's ``auc`` verbatim (argsort ranks, no tie averaging)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _pairwise_counts(pos: np.ndarray, neg_sorted: np.ndarray) -> np.ndarray:
    """For each pos value: #(neg < pos) + 0.5 #(neg == pos)."""
    lo = np.searchsorted(neg_sorted, pos, side="left")
    hi = np.searchsorted(neg_sorted, pos, side="right")
    return lo + 0.5 * (hi - lo)


def auc_tie(pos, neg) -> float:
    """Tie-aware Mann-Whitney AUC (average ranks). NaN if a set is empty or has NaN."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0 or np.isnan(pos).any() or np.isnan(neg).any():
        return float("nan")
    c = _pairwise_counts(pos, np.sort(neg))
    return float(c.sum() / (len(pos) * len(neg)))


def env_pair_matrix(
    pos: np.ndarray, pos_env: np.ndarray, neg: np.ndarray, neg_env: np.ndarray, num_envs: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M[e1, e2] = sum_{i in pos(e1), j in neg(e2)} 1[s_i > s_j] + 0.5 1[s_i == s_j]."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    M = np.zeros((num_envs, num_envs), dtype=np.float64)
    for e2 in np.unique(neg_env):
        ns = np.sort(neg[neg_env == e2])
        c = _pairwise_counts(pos, ns)
        M[:, e2] = np.bincount(pos_env, weights=c, minlength=num_envs)
    npos_e = np.bincount(pos_env, minlength=num_envs).astype(np.float64)
    nneg_e = np.bincount(neg_env, minlength=num_envs).astype(np.float64)
    return M, npos_e, nneg_e


def bootstrap_env_counts(num_envs: int, B: int, seed: int) -> np.ndarray:
    """(B, num_envs) draw counts of env blocks resampled with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, num_envs, size=(B, num_envs))
    counts = np.zeros((B, num_envs), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(B), num_envs), draws.reshape(-1)), 1.0)
    return counts


def bootstrap_auc(
    pos: np.ndarray, pos_env: np.ndarray, neg: np.ndarray, neg_env: np.ndarray, num_envs: int, counts: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Env-block bootstrap replicates of ``auc_tie``. Returns (valid replicates, n_skipped)."""
    M, npos_e, nneg_e = env_pair_matrix(pos, pos_env, neg, neg_env, num_envs)
    wpos = counts @ npos_e
    wneg = counts @ nneg_e
    num = np.einsum("be,ef,bf->b", counts, M, counts)
    valid = (wpos > 0) & (wneg > 0)
    vals = num[valid] / (wpos[valid] * wneg[valid])
    return vals, int((~valid).sum())


def compute_metrics(
    pos: np.ndarray, pos_env: np.ndarray, neg: np.ndarray, neg_env: np.ndarray, num_envs: int, counts: np.ndarray
) -> Dict[str, object]:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    m: Dict[str, object] = {"n_pos": int(len(pos)), "n_neg": int(len(neg))}
    nan = float("nan")
    if len(pos) == 0 or len(neg) == 0 or np.isnan(pos).any() or np.isnan(neg).any():
        m.update(auc=nan, auc_scorer=nan, ratio=nan, ci_lo=nan, ci_hi=nan, boot_valid=0,
                 boot_skipped=int(len(counts)), mean_pos=float(pos.mean()) if len(pos) else nan,
                 mean_neg=float(neg.mean()) if len(neg) else nan)
        return m
    m["auc"] = auc_tie(pos, neg)
    m["auc_scorer"] = float(auc_scorer(pos, neg))
    m["ratio"] = float(pos.mean() / max(neg.mean(), 1e-12))
    m["mean_pos"] = float(pos.mean())
    m["mean_neg"] = float(neg.mean())
    vals, skipped = bootstrap_auc(pos, pos_env, neg, neg_env, num_envs, counts)
    m["boot_valid"] = int(len(vals))
    m["boot_skipped"] = skipped
    if len(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        m["ci_lo"], m["ci_hi"] = float(lo), float(hi)
    else:
        m["ci_lo"] = m["ci_hi"] = nan
    return m


# --------------------------------------------------------------------------------------
# World-model signals (scorer path, verbatim)
# --------------------------------------------------------------------------------------
def history_horizon() -> int:
    from configs.go2_flat_cfg import Go2FlatConfig

    mac = Go2FlatConfig().model_architecture_config
    return getattr(mac, "history_horizon", 32)


def load_wm(path: str, device: str):
    """Scorer's checkpoint load + model construction + strict state-dict load."""
    from rsl_rl.modules import SystemDynamicsEnsemble
    from configs.go2_flat_cfg import Go2FlatConfig

    cfg = Go2FlatConfig()
    mac = cfg.model_architecture_config
    H = getattr(mac, "history_horizon", 32)

    ckpt = torch.load(path, map_location=device)

    sd = SystemDynamicsEnsemble(
        45,
        12,
        getattr(mac, "extension_dim", 0),
        getattr(mac, "contact_dim", 8),
        getattr(mac, "termination_dim", 1),
        device,
        ensemble_size=getattr(mac, "ensemble_size", None) or getattr(mac, "num_models", None) or 5,
        history_horizon=H,
        architecture_config=mac.architecture_config,
        freeze_auxiliary=False,
    ).to(device)

    sd.load_state_dict(ckpt["system_dynamics_state_dict"], strict=True)
    sd.eval()
    return sd, ckpt, H


def wm_normalize(state_all: np.ndarray, action_all: np.ndarray, ckpt) -> Tuple[np.ndarray, np.ndarray, str]:
    """Scorer's normalisation, verbatim. Returns (state, action, message line)."""
    if ckpt.get("normalized", False) and "state_mean" in ckpt:
        sm = ckpt["state_mean"].detach().cpu().numpy().reshape(-1)
        ss = ckpt["state_std"].detach().cpu().numpy().reshape(-1)
        am = ckpt["action_mean"].detach().cpu().numpy().reshape(-1)
        astd = ckpt["action_std"].detach().cpu().numpy().reshape(-1)
        state_all = ((state_all - sm) / ss).astype(np.float32)
        action_all = ((action_all - am) / astd).astype(np.float32)
        msg = "[score] normalized inputs to WM training space"
    else:
        msg = "[score] WM has no saved normalizer -> using raw inputs"
    return state_all, action_all, msg


def wm_score_indices(sd, state_all: np.ndarray, action_all: np.ndarray, idx: Sequence[int], H: int, device: str,
                     progress_every: int = 0, with_logit: bool = False) -> np.ndarray:
    """Per-sample scorer path: returns (n, 3) float64 array of (epi, alea, term_p).

    With ``with_logit=True`` a 4th column holds the raw head-mean termination logit from
    the same forward pass (no saturation at p = 1.0, so its AUC has no float32 ties).
    """

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
        if with_logit:
            term_l = float(term.detach().cpu().reshape(-1)[0].item()) if term is not None else float("nan")
            return epi_v, alea_v, term_p, term_l
        return epi_v, alea_v, term_p

    out = []
    t0 = time.time()
    for i, f in enumerate(idx):
        out.append(score_one(f))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[wm] {i + 1}/{len(idx)} samples ({time.time() - t0:.1f}s)")
    return np.array(out, dtype=np.float64).reshape(-1, 4 if with_logit else 3)


def scorer_stats_line(name, x) -> str:
    x = np.asarray(x, dtype=np.float64)
    return (
        f"{name:18s} n={len(x):5d} "
        f"mean={x.mean():.6f} median={np.median(x):.6f} "
        f"p90={np.quantile(x, 0.90):.6f} p99={np.quantile(x, 0.99):.6f} "
        f"min={x.min():.6f} max={x.max():.6f}"
    )


def scorer_parity_lines(
    norm_msg: str, wm_arg: str, trace_arg: str, n_rows: int, num_envs: int, steps_per_env: int, H: int,
    sets: ScorerIndexSets, neg_idx: np.ndarray, fall: np.ndarray, neg: np.ndarray, pre1: np.ndarray, pre5: np.ndarray,
) -> List[str]:
    """The scorer's stdout, line by line, from the given (n, 3) score arrays."""
    L: List[str] = [norm_msg]
    L.append(f"[score] wm={wm_arg}")
    L.append(f"[score] trace={trace_arg}")
    L.append(f"[score] rows={n_rows} num_envs={num_envs} steps_per_env={steps_per_env}")
    L.append(f"[score] H={H} raw_terms={sets.raw_terms} seam_terms={sets.seam_terms} interior_terms={sets.interior_terms}")
    L.append(f"[score] valid fall transitions={len(sets.fall_idx)}")
    L.append(f"[score] valid pre1 transitions={len(sets.pre1_idx)}")
    L.append(f"[score] valid pre5 transitions={len(sets.pre5_idx)}")
    L.append(f"[score] negative candidates={len(sets.neg_cand)} sampled={len(neg_idx)}")
    if len(sets.fall_idx) == 0 or len(neg_idx) == 0:
        L.append("[score] not enough positives or negatives")
        return L

    fall_epi, fall_alea, fall_term = fall[:, 0], fall[:, 1], fall[:, 2]
    neg_epi, neg_alea, neg_term = neg[:, 0], neg[:, 1], neg[:, 2]
    auc = auc_scorer
    stats = scorer_stats_line
    # The scorer's literal "\\n" prints a backslash-n; kept for a byte-exact diff.
    L.append("\\n=== epistemic uncertainty: transition into failure vs walking ===")
    L.append(stats("fall_transition", fall_epi))
    L.append(stats("walking", neg_epi))
    L.append(f"ratio mean fall/walking = {fall_epi.mean() / max(neg_epi.mean(), 1e-12):.3f}")
    L.append(f"ROC-AUC epi(fall > walking) = {auc(fall_epi, neg_epi):.3f}")

    if len(pre1):
        L.append("\\n=== epistemic uncertainty: one step before failure ===")
        L.append(stats("pre1_failure", pre1[:, 0]))
        L.append(stats("walking", neg_epi))
        L.append(f"ratio mean pre1/walking = {pre1[:,0].mean() / max(neg_epi.mean(), 1e-12):.3f}")
        L.append(f"ROC-AUC epi(pre1 > walking) = {auc(pre1[:,0], neg_epi):.3f}")

    if len(pre5):
        L.append("\\n=== epistemic uncertainty: five steps before failure ===")
        L.append(stats("pre5_failure", pre5[:, 0]))
        L.append(stats("walking", neg_epi))
        L.append(f"ratio mean pre5/walking = {pre5[:,0].mean() / max(neg_epi.mean(), 1e-12):.3f}")
        L.append(f"ROC-AUC epi(pre5 > walking) = {auc(pre5[:,0], neg_epi):.3f}")

    L.append("\\n=== aleatoric uncertainty ===")
    L.append(stats("fall_transition", fall_alea))
    L.append(stats("walking", neg_alea))
    L.append(f"ROC-AUC alea(fall > walking) = {auc(fall_alea, neg_alea):.3f}")

    L.append("\\n=== termination probability ===")
    L.append(stats("fall_transition", fall_term))
    L.append(stats("walking", neg_term))
    L.append(f"ROC-AUC term(fall > walking) = {auc(fall_term, neg_term):.3f}")
    return L


# --------------------------------------------------------------------------------------
# kNN signals
# --------------------------------------------------------------------------------------
def pair_features(state: np.ndarray, action: np.ndarray, offset: int, features: str) -> np.ndarray:
    """Training features of one file: [state r, action r+offset] for r = 0..N-1-offset (float64)."""
    state = np.asarray(state, dtype=np.float64)
    if features == "state":
        return state
    action = np.asarray(action, dtype=np.float64)
    n = len(state) - offset
    if n <= 0:
        return np.zeros((0, state.shape[1] + action.shape[1]), dtype=np.float64)
    return np.concatenate([state[:n], action[offset:offset + n]], axis=1)


def query_features(state: np.ndarray, action: np.ndarray, idx: np.ndarray, offset: int, features: str) -> np.ndarray:
    """Query feature of index f: [state row f-1, action row f-1+offset] (float64)."""
    idx = np.asarray(idx, dtype=np.int64)
    s = np.asarray(state[idx - 1], dtype=np.float64)
    if features == "state":
        return s
    ai = idx - 1 + offset
    if len(ai) and (ai.min() < 0 or ai.max() >= len(action)):
        raise ValueError("knn query action row out of range; check --knn_pair_offset")
    return np.concatenate([s, np.asarray(action[ai], dtype=np.float64)], axis=1)


def load_train_features(files: Sequence[str], offset: int, features: str) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    feats, info = [], []
    for p in files:
        data = read_numeric_csv(p)
        x = pair_features(data[:, 0:STATE_DIM], data[:, STATE_DIM:STATE_DIM + ACTION_DIM], offset, features)
        bad = ~np.isfinite(x).all(axis=1)
        if bad.any():
            print(f"[warn] {p}: dropping {int(bad.sum())} feature rows with NaN/inf")
            x = x[~bad]
        print(f"[knn] {p}: rows={len(data)} features={len(x)}")
        info.append({"path": p, "realpath": os.path.realpath(p), "bytes": os.path.getsize(p),
                     "rows": int(len(data)), "features": int(len(x)), "dropped_nonfinite": int(bad.sum())})
        feats.append(x)
    if not feats:
        raise RuntimeError("no usable kNN training files")
    return np.concatenate(feats, axis=0), info


def own_norm_stats(train: np.ndarray, floor: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, int]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    n_floored = int((sd < floor).sum())
    return mu, np.maximum(sd, floor), n_floored


def wm_norm_stats(ckpt, features: str) -> Tuple[np.ndarray, np.ndarray]:
    if not (ckpt.get("normalized", False) and "state_mean" in ckpt):
        raise RuntimeError("--knn_norm wm requires a WM checkpoint with saved normalizer stats")
    g = lambda k: ckpt[k].detach().cpu().numpy().reshape(-1).astype(np.float64)
    if features == "state":
        return g("state_mean"), g("state_std")
    return np.concatenate([g("state_mean"), g("action_mean")]), np.concatenate([g("state_std"), g("action_std")])


def knn_mean_distances(
    query: np.ndarray, train: np.ndarray, ks: Sequence[int], device: str = "cpu",
    query_batch: int = 1024, train_chunk: int = 131072,
) -> Dict[int, np.ndarray]:
    """Exact kNN: for each k, mean Euclidean distance to the k nearest training rows."""
    ks = sorted(set(int(k) for k in ks))
    kmax = ks[-1]
    if kmax > len(train):
        raise ValueError(f"k={kmax} > number of training rows {len(train)}")
    dt = torch.float64
    T = torch.from_numpy(np.ascontiguousarray(train, dtype=np.float64)).to(device=device, dtype=dt)
    Qall = np.ascontiguousarray(query, dtype=np.float64)
    out = {k: np.empty(len(Qall), dtype=np.float64) for k in ks}
    with torch.no_grad():
        for qs in range(0, len(Qall), query_batch):
            Q = torch.from_numpy(Qall[qs:qs + query_batch]).to(device=device, dtype=dt)
            best_d: Optional[torch.Tensor] = None
            best_i: Optional[torch.Tensor] = None
            for ts in range(0, T.shape[0], train_chunk):
                D = torch.cdist(Q, T[ts:ts + train_chunk])
                kk = min(kmax, D.shape[1])
                d, i = torch.topk(D, kk, dim=1, largest=False)
                i = i + ts
                if best_d is None:
                    best_d, best_i = d, i
                else:
                    cd = torch.cat([best_d, d], dim=1)
                    ci = torch.cat([best_i, i], dim=1)
                    kk2 = min(kmax, cd.shape[1])
                    best_d, sel = torch.topk(cd, kk2, dim=1, largest=False)
                    best_i = torch.gather(ci, 1, sel)
            # recompute the selected neighbours' distances directly (no |a|^2+|b|^2-2ab cancellation)
            nb = T[best_i]  # (q, kmax, dim)
            exact = torch.sqrt(((nb - Q[:, None, :]) ** 2).sum(dim=-1))
            exact, _ = torch.sort(exact, dim=1)
            ex = exact.cpu().numpy()
            for k in ks:
                out[k][qs:qs + len(Q)] = ex[:, :k].mean(axis=1)
    return out


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def git_info(path: str) -> Dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", path, "status", "--porcelain", "--", "."],
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def file_info(path: str, sha: bool = True) -> Dict[str, object]:
    d: Dict[str, object] = {"path": path, "realpath": os.path.realpath(path), "bytes": os.path.getsize(path)}
    if sha:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        d["sha256"] = h.hexdigest()
    return d


def _json_clean(o):
    if isinstance(o, dict):
        return {str(k): _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return _json_clean(o.tolist())
    return o


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def resolve_device(dev: str) -> str:
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] {dev} requested but CUDA unavailable -> using cpu (WM numbers will not match a GPU scorer run)")
        return "cpu"
    return dev


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trace", required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps_per_env", type=int, default=1000)
    p.add_argument("--wm", default=None, help="WM checkpoint; enables epi/alea/term signals")
    p.add_argument("--knn_train", action="append", default=None,
                   help="glob of kNN training CSVs (repeatable); enables knn signals")
    p.add_argument("--knn_name", default="", help="label of the kNN training set")
    p.add_argument("--knn_pair_offset", type=int, default=1)
    p.add_argument("--knn_features", choices=["state_action", "state"], default="state_action")
    p.add_argument("--knn_norm", choices=["own", "wm"], default="own")
    p.add_argument("--knn_k", type=parse_int_list, default=[1, 5, 10, 50])
    p.add_argument("--knn_primary_k", type=int, default=10)
    p.add_argument("--knn_query_batch", type=int, default=1024)
    p.add_argument("--knn_train_chunk", type=int, default=131072)
    p.add_argument("--num_neg", type=int, default=5000)
    p.add_argument("--clean_gap", type=int, default=25)
    p.add_argument("--clean_censor_end", type=int, choices=[0, 1], default=1,
                   help="1: drop clean-negative candidates whose look-ahead runs past the block end")
    p.add_argument("--lead_ks", type=parse_int_list, default=[0, 1, 2, 3, 5, 10, 15, 20, 25, 50])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, object]:
    t_start = time.time()
    if torch is None:
        raise RuntimeError("torch is required")
    os.makedirs(args.out, exist_ok=True)
    device = resolve_device(args.device)
    if args.knn_norm == "wm" and args.knn_train and not args.wm:
        raise SystemExit("--knn_norm wm requires --wm")
    if args.knn_train and args.knn_primary_k not in args.knn_k:
        raise SystemExit("--knn_primary_k must be one of --knn_k")
    if args.knn_pair_offset < 0:
        raise SystemExit("--knn_pair_offset must be >= 0")

    H = history_horizon()

    # ---- trace
    data = read_numeric_csv(args.trace)
    assert data.shape[1] == NUM_COLS, data.shape
    assert data.shape[0] == args.num_envs * args.steps_per_env, data.shape
    state_raw = np.ascontiguousarray(data[:, 0:45])
    action_raw = np.ascontiguousarray(data[:, 45:57])
    term_all = np.ascontiguousarray(data[:, 65] > 0.5)

    # ---- index sets
    sets = select_indices(term_all, args.num_envs, args.steps_per_env, H)
    neg_idx = sample_scorer_negatives(sets.neg_cand, args.num_neg)
    lead = lead_time_sets(term_all, sets.fall_idx, args.steps_per_env, H, args.lead_ks)
    for k, ref in ((0, sets.fall_idx), (1, sets.pre1_idx), (5, sets.pre5_idx)):
        if k in lead and not np.array_equal(lead[k], np.asarray(ref, dtype=np.int64)):
            raise AssertionError(f"lead{k} != scorer set; this is a bug")
    clean_cand = clean_negative_candidates(term_all, sets.neg_cand, args.steps_per_env, args.clean_gap,
                                           bool(args.clean_censor_end))
    clean_idx = sample_clean_negatives(clean_cand, args.num_neg, args.seed)

    pos_sets: Dict[str, np.ndarray] = {
        "fall": np.asarray(sets.fall_idx, dtype=np.int64),
        "pre1": np.asarray(sets.pre1_idx, dtype=np.int64),
        "pre5": np.asarray(sets.pre5_idx, dtype=np.int64),
    }
    for k in args.lead_ks:
        pos_sets[f"lead{k}"] = lead[k]
    neg_sets: Dict[str, np.ndarray] = {"scorer": np.asarray(neg_idx, dtype=np.int64),
                                       "clean": np.asarray(clean_idx, dtype=np.int64)}

    counts = {
        "rows": int(len(data)), "H": int(H), "raw_terms": sets.raw_terms, "seam_terms": sets.seam_terms,
        "interior_terms": sets.interior_terms, "neg_cand": len(sets.neg_cand), "clean_cand": int(len(clean_cand)),
        **{f"n_{k}": int(len(v)) for k, v in pos_sets.items()},
        **{f"n_neg_{k}": int(len(v)) for k, v in neg_sets.items()},
    }
    print(f"[eval] trace={args.trace} rows={len(data)} num_envs={args.num_envs} steps_per_env={args.steps_per_env} H={H}")
    print(f"[eval] falls={len(sets.fall_idx)} pre1={len(sets.pre1_idx)} pre5={len(sets.pre5_idx)} "
          f"neg_cand={len(sets.neg_cand)} scorer_neg={len(neg_idx)} clean_cand={len(clean_cand)} clean_neg={len(clean_idx)}")
    print("[eval] lead sets: " + " ".join(f"lead{k}={len(lead[k])}" for k in args.lead_ks))

    eval_idx = np.unique(np.concatenate([*pos_sets.values(), *neg_sets.values()]).astype(np.int64))
    pos_in_eval = {f: i for i, f in enumerate(eval_idx.tolist())}
    signals: Dict[str, np.ndarray] = {}
    inputs: Dict[str, object] = {"trace": file_info(args.trace)}
    knn_meta: Dict[str, object] = {}
    parity_lines: List[str] = []

    # ---- WM signals
    ckpt = None
    if args.wm:
        t0 = time.time()
        sd, ckpt, H_wm = load_wm(args.wm, device)
        assert H_wm == H
        inputs["wm"] = file_info(args.wm)
        state_wm, action_wm, norm_msg = wm_normalize(state_raw, action_raw, ckpt)
        print(norm_msg)
        vals = wm_score_indices(sd, state_wm, action_wm, eval_idx, H, device, progress_every=2000, with_logit=True)
        signals["epi"], signals["alea"], signals["term"] = vals[:, 0], vals[:, 1], vals[:, 2]
        signals["term_logit"] = vals[:, 3]
        print(f"[wm] scored {len(eval_idx)} indices in {time.time() - t0:.1f}s")

        def take(idx):
            if len(idx) == 0:
                return np.zeros((0, 3))
            return vals[[pos_in_eval[int(f)] for f in idx]][:, :3]

        parity_lines = scorer_parity_lines(
            norm_msg, args.wm, args.trace, len(data), args.num_envs, args.steps_per_env, H, sets, neg_idx,
            take(sets.fall_idx), take(neg_idx),
            take(sets.pre1_idx) if len(sets.pre1_idx) else np.zeros((0, 3)),
            take(sets.pre5_idx) if len(sets.pre5_idx) else np.zeros((0, 3)),
        )
        with open(os.path.join(args.out, "scorer_parity.txt"), "w") as fh:
            fh.write("\n".join(parity_lines) + "\n")

    # ---- kNN signals
    if args.knn_train:
        t0 = time.time()
        files = resolve_train_files(args.knn_train)
        if not files:
            raise SystemExit("no usable kNN training files")
        train, finfo = load_train_features(files, args.knn_pair_offset, args.knn_features)
        q = query_features(state_raw, action_raw, eval_idx, args.knn_pair_offset, args.knn_features)
        if args.knn_norm == "own":
            mu, sdv, n_floored = own_norm_stats(train)
            print(f"[knn] own normalisation: {n_floored} of {len(mu)} dims had std < 1e-6 and were floored")
        else:
            mu, sdv = wm_norm_stats(ckpt, args.knn_features)
            n_floored = 0
            print("[knn] normalisation with WM checkpoint stats")
        train = (train - mu) / sdv
        q = (q - mu) / sdv
        print(f"[knn] train features {train.shape}, queries {q.shape}, device {device}")
        dists = knn_mean_distances(q, train, args.knn_k, device, args.knn_query_batch, args.knn_train_chunk)
        for k in args.knn_k:
            signals[f"knn{k}"] = dists[k]
        print(f"[knn] done in {time.time() - t0:.1f}s")
        knn_meta = {
            "name": args.knn_name, "patterns": args.knn_train, "files": finfo, "n_train": int(len(train)),
            "dim": int(train.shape[1]), "norm": args.knn_norm, "n_std_floored": n_floored,
            "features": args.knn_features, "pair_offset": args.knn_pair_offset, "ks": args.knn_k,
            "primary": f"knn{args.knn_primary_k}", "mean": mu, "std": sdv,
        }
        inputs["knn_train"] = finfo
        del train

    # ---- metrics
    boot_counts = bootstrap_env_counts(args.num_envs, args.bootstrap, args.seed)
    env_of = lambda idx: (np.asarray(idx, dtype=np.int64) // args.steps_per_env).astype(np.int64)
    metrics: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for sname, sv in signals.items():
        lookup = lambda idx: sv[np.searchsorted(eval_idx, np.asarray(idx, dtype=np.int64))]
        metrics[sname] = {}
        for pname, pidx in pos_sets.items():
            metrics[sname][pname] = {}
            for nname, nidx in neg_sets.items():
                metrics[sname][pname][nname] = compute_metrics(
                    lookup(pidx), env_of(pidx), lookup(nidx), env_of(nidx), args.num_envs, boot_counts)

    # ---- outputs
    npz = {"idx": eval_idx, "env": eval_idx // args.steps_per_env, "local_t": eval_idx % args.steps_per_env,
           "neg_scorer_order": neg_sets["scorer"], "neg_clean_order": neg_sets["clean"]}
    for name, idx in {**pos_sets, **{f"neg_{k}": v for k, v in neg_sets.items()}}.items():
        npz[f"in_{name}"] = np.isin(eval_idx, idx)
    npz["in_neg_cand"] = np.isin(eval_idx, np.asarray(sets.neg_cand, dtype=np.int64))
    npz["in_clean_cand"] = np.isin(eval_idx, clean_cand)
    for sname, sv in signals.items():
        npz[f"sig_{sname}"] = sv
    np.savez_compressed(os.path.join(args.out, "scores.npz"), **npz)

    summary = {
        "script": os.path.abspath(__file__),
        "git": git_info(_THIS_DIR),
        "args": vars(args),
        "device_used": device,
        "torch_version": torch.__version__,
        "inputs": inputs,
        "counts": counts,
        "signals": list(signals.keys()),
        "positive_sets": list(pos_sets.keys()),
        "negative_sets": list(neg_sets.keys()),
        "knn": knn_meta,
        "metrics": metrics,
        "runtime_s": time.time() - t_start,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(_json_clean(summary), fh, indent=1)

    # ---- printing
    print_table(metrics, pos_sets, neg_sets)
    if parity_lines:
        print("\n===== BEGIN SCORER-PARITY BLOCK (same lines as score_go2_exploit_trace_uncertainty.py) =====")
        for line in parity_lines:
            print(line)
        print("===== END SCORER-PARITY BLOCK =====")
    print(f"[eval] wrote {args.out}/summary.json, scores.npz ({time.time() - t_start:.1f}s)")
    return summary


def _fmt(m: Dict[str, object]) -> str:
    a, lo, hi = m.get("auc"), m.get("ci_lo"), m.get("ci_hi")
    if a is None or not np.isfinite(a):
        return f"{'n/a':>21s}"
    return f"{a:.3f} [{lo:.3f},{hi:.3f}]"


def print_table(metrics, pos_sets, neg_sets) -> None:
    for sname, by_pos in metrics.items():
        print(f"\n=== signal {sname}: tie-aware AUC [95% env-bootstrap CI] | scorer-AUC | ratio ===")
        hdr = f"{'positives':10s} {'n_pos':>6s}"
        for nname in neg_sets:
            hdr += f" | {'neg=' + nname:21s} {'aucS':>6s} {'ratio':>7s} {'n_neg':>6s}"
        print(hdr)
        for pname in pos_sets:
            row = by_pos[pname]
            first = next(iter(row.values()))
            line = f"{pname:10s} {first['n_pos']:6d}"
            for nname in neg_sets:
                m = row[nname]
                aucs = m["auc_scorer"]
                ratio = m["ratio"]
                line += (f" | {_fmt(m):21s} {aucs:6.3f} {ratio:7.3f} {m['n_neg']:6d}"
                         if np.isfinite(aucs) else f" | {_fmt(m):21s} {'n/a':>6s} {'n/a':>7s} {m['n_neg']:6d}")
            print(line)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
