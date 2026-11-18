#!/usr/bin/env python3
"""Plan v6 RQ1, check C0: reproduce the 27 Oct paired replay, then measure the healing index on
~270 falls and matched non-fall windows. CPU only, no simulator, no training.

Step 1 (reproduction). `replay_fall_paths.run` is called unchanged on the 27 Oct trace and model.
Its real-context uncertainty at step 25 (9.78) and imagined uncertainty at steps 1 and 25 (5.11, 2.66)
must match the 27 Oct JSON within 5 %.

Step 2 (battery). Same replay rules as `replay_fall_paths.py` (32-step real context, recorded actions,
fixed driving head, mean-state feedback, forced continuation), with HORIZON = 26 predictions so that
depth d = number of imagined states fed back runs 0..25 and the last target is the terminal row.
* u_real(d): one-step prediction from 32 real rows ending at the row before the target (teacher forcing).
* u_imag(h, d): prediction on the path driven by head h's own mean (all heads read the same trunk).
* u = std over heads of the head means in normalized units (the model's epistemic uncertainty),
  summed over all 45 dimensions and over each component of GROUPS.
* HI(d) = mean over (window, driving head) pairs of log(u_imag / u_real); HR = exp(HI); healing
  deficit H = -HI (plan §11). HR(0) must lie in 0.9-1.1 (both arms see the same inputs at d = 0).
* Matched non-falls: per fall, the nearest eligible non-fall window of the same trace in
  (command vx, vy, wz, base speed at the window start), standardized, without replacement. A non-fall
  window has no reset from 31 rows before it to 25 rows after its end.
* CI: cluster bootstrap over episodes, stratified by source (policy/batch), 10,000 resamples, 95 %.
* C0 rule (plan §2): fall-specific if H_fall(25) >= 1.5 x H_nonfall(25) with the ratio CI excluding 1
  (if H_nonfall(25) <= 0: H_fall(25) with CI above 0); not supported if the ratio CI lies below 1.5;
  otherwise inconclusive. Components are reported for the "only in gravity -> C6" reading.
u_real and u_imag are reported separately (doubt E18: H can shrink because u_real drops).

Cause-test outputs (plan §2), stored per window in c0_windows.npz and read by c0_compare.py:
* prediction error per depth: real-context one-step error and the driving head's own imagined error,
  in normalized units (45 dims) and as base-velocity error in m/s;
* C1a: per-member trajectory spread = std over driving heads of each head's own imagined state,
  against the shared-path disagreement u_imag, and its Spearman with the imagined error;
* C1c: --wm a.pt,b.pt,... reads several single-head models as one ensemble on a shared path
  (independent trunks; each model keeps its own memory, all see the same inputs);
* C7: --imag-context K starts imagination from the last K real rows (cold memory: K = 1);
  the real-context arm keeps 32 rows, so HR(0) is no longer 1 by construction;
* C8: --head-subsets 2,5,10 adds the healing index of 50 random head subsets per size;
* C5 / C9: --feedback head (default) | sampled (mean + aleatoric std x N(0,1)) | random_step | mean |
  most_tilted chooses the state fed back in imagination; the report gives the share of paths that reach
  the fall (termination > 0.5 anywhere; tilt >= 30 deg), aleatoric sigma by depth, 1-sigma coverage
  of |error| by the head spread, the scale s(d) fitted on non-falls and the fall coverage after it,
  and the Spearman of spread vs error by depth;
* C2 free run: --actions held keeps the last real action after the context (the real arm keeps the
  recorded actions, so H is not meaningful there; c2_attractor.py reads the imagined states).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pre_reset_transitions import GROUPS, gather, sha256, validate_trace

HISTORY = 32
HORIZON = 26
DEPTHS = (0, 1, 8, 16, 25)
REPRO_TOLERANCE = 0.05
COMPONENTS = {"sum": (0, 45), **GROUPS}
BINS_NOTE = "depth d = imagined states fed back before the prediction; d = 0 is the first prediction"


# ---------------------------------------------------------------- model passes
class ModelSet:
    """Several models read as one ensemble on a shared path (C1c: independent trunks)."""

    def __init__(self, models):
        self.models = list(models)
        self.ensemble_size = sum(m.ensemble_size for m in self.models)

    def reset(self):
        for m in self.models:
            m.reset()


@torch.no_grad()
def forward_all(model, xs, xa):
    if isinstance(model, ModelSet):
        parts = [forward_all(m, xs, xa) for m in model.models]
        return tuple(torch.cat(group, 0) for group in zip(*parts))
    return _forward_one(model, xs, xa)


@torch.no_grad()
def _forward_one(model, xs, xa):
    """One recurrent step of SystemDynamicsEnsemble.forward, returning every head's state mean.

    Same calls in the same order as the library forward (state trunk, state heads, auxiliary trunk,
    auxiliary heads), so the hidden states advance exactly as in `model(xs, xa, ids)`.
    Returns means [heads, B, 45], contact logits [heads, B, 8], termination logits [heads, B, 1],
    aleatoric stds [heads, B, 45] (normalized units).
    """
    base = model.state_base(xs, xa)
    means, stds = [], []
    for head in model.state_heads:
        mean, std = head(base, xs)
        if model.prediction_type == "sequence":
            mean, std = mean[:, -1], std[:, -1]
        means.append(mean)
        stds.append(std)
    aux = model.auxiliary_base(xs, xa)
    contacts, terms = [], []
    for head in model.auxiliary_heads:
        _, contact, term = head(aux, xs)
        if model.prediction_type == "sequence":
            contact, term = contact[:, -1], term[:, -1]
        contacts.append(contact)
        terms.append(term)
    return torch.stack(means), torch.stack(contacts), torch.stack(terms), torch.stack(stds)


def head_spread(means):
    """Per-dimension std over heads (unbiased, as the library's epistemic uncertainty) [B, 45]."""
    return means.std(dim=0)


def _normalized(contract, data, origins, lo, hi, key):
    stats = contract.statistics
    values = torch.from_numpy(gather(data[key], origins, lo, hi)).float()
    return (values - stats[f"{key}_mean"]) / stats[f"{key}_std"]


def make_subsets(heads, sizes, n=50, seed=0):
    """Random head subsets {K: int array [n, K]} (sizes >= heads are skipped)."""
    rng = np.random.default_rng(seed)
    return {k: np.stack([np.sort(rng.choice(heads, k, replace=False)) for _ in range(n)])
            for k in sizes if 2 <= k < heads}


def subset_spread(means, subsets):
    """Summed std over the heads of each subset: {K: [n, B]}."""
    return {k: means[torch.from_numpy(idx)].std(dim=1).sum(-1).numpy() for k, idx in subsets.items()}


def _errors(pred, target, stats):
    """Squared errors: mean over dims in normalized units, and over base velocity (dims 0-2) in m/s."""
    d = pred - target
    vel = d[..., 0:3] * stats["state_std"][0:3]
    return (d ** 2).mean(-1).numpy(), (vel ** 2).mean(-1).numpy()


@torch.no_grad()
def real_context_pass(model, contract, data, origins, horizon, subsets=None, batch=256):
    """Teacher forcing: the prediction of step k from 32 real rows.
    Returns spread [H, W, 45], se_norm [H, W], se_vel [H, W] (mean over heads), subset spreads
    {K: [n, len(DEPTHS), W]} at DEPTHS."""
    stats = contract.statistics
    rows = np.asarray([(t + k, n) for k in range(horizon) for t, n in origins], dtype=int).reshape(-1, 2)
    target = (torch.from_numpy(data["next_state"][rows[:, 0], rows[:, 1]]).float() - stats["state_mean"]) / stats["state_std"]
    spread, sigma, se_n, se_v, sub = [], [], [], [], {k: [] for k in (subsets or {})}
    for b in range(0, len(rows), batch):
        chunk = rows[b:b + batch]
        model.reset()
        means, _, _, stds = forward_all(model, _normalized(contract, data, chunk, -31, 1, "state"),
                                        _normalized(contract, data, chunk, -31, 1, "action"))
        spread.append(head_spread(means).numpy())
        sigma.append(stds.sum(-1).mean(0).numpy())
        n_err, v_err = _errors(means, target[b:b + batch][None], stats)
        se_n.append(n_err.mean(0))
        se_v.append(v_err.mean(0))
        for k, v in subset_spread(means, subsets or {}).items():
            sub[k].append(v)
    W = len(origins)
    out = dict(spread=np.concatenate(spread).reshape(horizon, W, -1), sigma=np.concatenate(sigma).reshape(horizon, W),
               se_norm=np.concatenate(se_n).reshape(horizon, W), se_vel=np.concatenate(se_v).reshape(horizon, W))
    depths = [d for d in DEPTHS if d < horizon]
    out["subsets"] = {k: np.concatenate(v, axis=1).reshape(len(v[0]), horizon, W)[:, depths] for k, v in sub.items()}
    return out


def tilt_deg(gravity):
    """Tilt angle in degrees between a (physical) gravity vector and straight down, for any vector length:
    atan2(|g_xy|, -g_z). Upright 0, on its side 90, upside down 180 (review 18 Nov: |g_xy| alone is not an
    angle for an unconstrained predicted vector)."""
    g = torch.as_tensor(gravity) if not isinstance(gravity, torch.Tensor) else gravity
    return torch.rad2deg(torch.atan2(torch.linalg.norm(g[..., :2], dim=-1), -g[..., 2]))


FEEDBACK = ("head", "random_step", "mean", "most_tilted", "sampled")
ACTIONS = ("recorded", "held")
GRAVITY = slice(6, 9)


@torch.no_grad()
def imagined_pass(model, contract, data, origins, horizon, context=HISTORY, subsets=None, feedback="head", seed=0,
                  actions_mode="recorded"):
    """Imagination started from the last `context` real rows; the state fed back at each step depends on
    `feedback` (plan C5, C9):
      head         each head drives its own path with its mean (RWM-U imagination; one path per head)
      random_step  a new random head at every step (one path per seed, as many paths as heads)
      mean         the ensemble mean (one path)
      most_tilted  the head whose state is most tilted (largest tilt angle, physical units; one path)
      sampled      each head's mean + its aleatoric std x N(0, 1) (one path per head)
    Per path p: spread [P, H, W, 45] (disagreement of all heads on that path), own [P, H, W, 45] (fed
    state, normalized), stop [P, H, W] (termination probability of the head whose state is fed; the
    ensemble mean only in mean mode), stop_mean [P, H, W] (ensemble mean probability, every mode, for a
    like-for-like comparison), se_norm / se_vel [P, H, W] (fed state vs target), sigma [P, H, W]
    (aleatoric std summed over dims, mean over heads), ratio [P, H, W, 45] = |fed - target| / spread
    (1-sigma coverage; a zero spread counts as not covered), tilt [P, H, W] (fed tilt angle in degrees),
    gnorm [P, H, W] (fed gravity vector length) and subset spreads at DEPTHS."""
    if not 1 <= context <= HISTORY:
        raise ValueError(f"context must be 1..{HISTORY}")
    if feedback not in FEEDBACK:
        raise ValueError(f"feedback must be one of {FEEDBACK}")
    if actions_mode not in ACTIONS:
        raise ValueError(f"actions must be one of {ACTIONS}")
    stats = contract.statistics
    states = _normalized(contract, data, origins, -31, 1, "state")[:, HISTORY - context:]
    actions = _normalized(contract, data, origins, -31, horizon, "action")
    if actions_mode == "held":                    # C2 free run: the last real action is held after the context
        actions = actions.clone()
        actions[:, HISTORY:] = actions[:, HISTORY - 1:HISTORY]
    target = (torch.from_numpy(gather(data["next_state"], origins, 0, horizon)).float()
              - stats["state_mean"]) / stats["state_std"]                                   # [W, H, 45]
    depths = [d for d in DEPTHS if d < horizon]
    heads = model.ensemble_size
    paths = 1 if feedback in ("mean", "most_tilted") else heads
    gstd, gmean = stats["state_std"][GRAVITY], stats["state_mean"][GRAVITY]
    keys = ("spread", "own", "stop", "stop_mean", "se_norm", "se_vel", "sigma", "ratio", "tilt", "gnorm")
    acc = {k: [] for k in keys}
    sub = {k: np.full((len(idx), paths, len(depths), len(origins)), np.nan) for k, idx in (subsets or {}).items()}
    for p in range(paths):
        rng = torch.Generator().manual_seed(seed * 1000 + p)
        model.reset()
        xs, xa = states, actions[:, HISTORY - context:HISTORY]
        rec = {k: [] for k in keys}
        for k in range(horizon):
            means, _, terms, stds = forward_all(model, xs, xa)
            probs = torch.sigmoid(terms).squeeze(-1)                                           # [heads, B]
            if feedback == "head":
                fed, stop = means[p], probs[p]
            elif feedback == "sampled":
                fed = means[p] + stds[p] * torch.randn(means[p].shape, generator=rng)
                stop = probs[p]
            elif feedback == "random_step":
                pick = torch.randint(heads, (means.shape[1],), generator=rng)
                cols = torch.arange(means.shape[1])
                fed, stop = means[pick, cols], probs[pick, cols]                               # same head for state and stop
            elif feedback == "mean":
                fed, stop = means.mean(0), probs.mean(0)                                        # no single head: mean of both
            else:                                                                               # most_tilted
                tilt_h = tilt_deg(means[..., GRAVITY] * gstd + gmean)                           # [heads, B], degrees
                pick, cols = tilt_h.argmax(0), torch.arange(means.shape[1])
                fed, stop = means[pick, cols], probs[pick, cols]
            spread = head_spread(means)
            rec["spread"].append(spread.numpy())
            rec["own"].append(fed.numpy())
            rec["stop"].append(stop.numpy())
            n_err, v_err = _errors(fed, target[:, k], stats)
            rec["se_norm"].append(n_err)
            rec["se_vel"].append(v_err)
            rec["sigma"].append(stds.sum(-1).mean(0).numpy())
            rec["ratio"].append(((fed - target[:, k]).abs() / spread.clamp_min(1e-12)).numpy())
            g_fed = fed[:, GRAVITY] * gstd + gmean
            rec["tilt"].append(tilt_deg(g_fed).numpy())
            rec["gnorm"].append(torch.linalg.norm(g_fed, dim=-1).numpy())
            rec["stop_mean"].append(probs.mean(0).numpy())
            if k in depths:
                for K, v in subset_spread(means, subsets or {}).items():
                    member = (subsets[K] == p).any(axis=1) if feedback in ("head", "sampled") else np.ones(len(v), bool)
                    sub[K][member, p, depths.index(k)] = v[member]
            xs, xa = fed[:, None], actions[:, HISTORY + k:HISTORY + k + 1]
        for key in keys:
            acc[key].append(np.stack(rec[key]))
    out = {k: np.stack(v) for k, v in acc.items()}
    out["subsets"] = sub
    return out


def real_context_spread(model, contract, data, origins, horizon, batch=256):
    """Spread [H, W, 45] of the real-context pass (kept for the 27 Oct equality test)."""
    return real_context_pass(model, contract, data, origins, horizon, batch=batch)["spread"]


def imagined_spread(model, contract, data, origins, horizon):
    """Spread [heads, H, W, 45] and stop probability of the imagined pass (32-row context)."""
    out = imagined_pass(model, contract, data, origins, horizon)
    return out["spread"], out["stop"]


# ---------------------------------------------------------------- windows
def episode_index(done):
    """Episode number per row and env: increments on the row after a reset."""
    ep = np.zeros(done.shape, dtype=np.int64)
    ep[1:] = np.cumsum(done[:-1], axis=0)
    return ep


def fall_windows(trace, horizon=HORIZON, history=HISTORY):
    """Window starts (t, n) whose last target is a physical fall, with clean 32-row history.
    Same rule as replay_fall_paths.fall_origins, for any horizon."""
    done = trace["done"]
    falls = done & trace["terminated"] & ~trace["time_out"]
    origins, excluded = [], 0
    for end, n in np.argwhere(falls):
        start = int(end) - horizon + 1
        if start < history - 1 or done[start - history + 1:end, n].any():
            excluded += 1
        else:
            origins.append((start, int(n)))
    return np.asarray(origins, dtype=int).reshape(-1, 2), dict(
        physical_falls=int(falls.sum()), included=len(origins), short_history_excluded=excluded)


def nonfall_candidates(done, horizon=HORIZON, history=HISTORY, survive=25, stride=5):
    """Window starts with no reset from t-31 to t+horizon-1+survive (all rows must exist)."""
    T, N = done.shape
    cum = np.vstack([np.zeros((1, N), np.int64), np.cumsum(done, axis=0)])
    starts = np.arange(history - 1, T - horizon - survive + 1)[::stride]
    lo, hi = starts - (history - 1), starts + horizon + survive      # rows lo .. hi-1 must hold no reset
    clean = (cum[hi] - cum[lo]) == 0                                 # [S, N]
    s, n = np.nonzero(clean)
    return np.column_stack([starts[s], n]).astype(int)


def match_features(trace, origins):
    t, n = origins[:, 0], origins[:, 1]
    cmd = trace["command"][t, n, :3].astype(float)
    speed = np.linalg.norm(trace["state"][t, n, 0:2].astype(float), axis=-1)
    return np.column_stack([cmd, speed])


def match_nonfalls(trace, falls, candidates, seed=0):
    """Nearest candidate per fall in standardized (command, speed), without replacement; falls are
    processed in a seeded random order. Returns origins [F, 2] and distances [F]."""
    if len(candidates) < len(falls):
        raise ValueError(f"only {len(candidates)} non-fall candidates for {len(falls)} falls")
    fc, cc = match_features(trace, falls), match_features(trace, candidates)
    scale = cc.std(axis=0)
    scale[scale == 0] = 1.0
    fc, cc = fc / scale, cc / scale
    free = np.ones(len(candidates), bool)
    chosen = np.zeros(len(falls), dtype=int)
    dist = np.zeros(len(falls))
    for i in np.random.default_rng(seed).permutation(len(falls)):
        d = np.linalg.norm(cc - fc[i], axis=1)
        d[~free] = np.inf
        j = int(np.argmin(d))
        chosen[i], dist[i], free[j] = j, d[j], False
    return candidates[chosen], dist


# ---------------------------------------------------------------- statistics
class ZeroDisagreement(ValueError):
    """Rule (review 18 Nov): a zero summed head spread makes the log ratio undefined; the run stops and names
    the windows instead of dropping or clamping them."""


def log_ratio(u_imag, u_real, component):
    """Per window and depth: mean over driving heads of log(u_imag / u_real) for one component.
    u_imag [heads, H, W, 45], u_real [H, W, 45] -> [H, W]."""
    lo, hi = COMPONENTS[component]
    ui, ur = u_imag[..., lo:hi].sum(-1), u_real[..., lo:hi].sum(-1)
    if (ui <= 0).any() or (ur <= 0).any():
        bad = sorted(set(np.flatnonzero((ui <= 0).any(axis=(0, 1)) | (ur <= 0).any(axis=0)).tolist()))
        raise ZeroDisagreement(f"zero head spread in component {component}, windows {bad[:20]}")
    return np.log(ui / ur[None]).mean(axis=0)


def cluster_bootstrap(values, clusters, strata, n_boot=10_000, seed=0):
    """Mean of `values` [W] (or [W, K]) with a cluster bootstrap stratified by `strata`.
    Returns the point estimate and the resampled means [n_boot(, K)]."""
    values = np.asarray(values, float)
    flat = values.reshape(len(values), -1)
    rng = np.random.default_rng(seed)
    total_sum = np.zeros((n_boot, flat.shape[1]))
    total_cnt = np.zeros(n_boot)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        keys, inv = np.unique(clusters[idx], return_inverse=True)
        sums = np.zeros((len(keys), flat.shape[1]))
        np.add.at(sums, inv, flat[idx])
        cnts = np.bincount(inv, minlength=len(keys)).astype(float)
        draw = rng.integers(0, len(keys), size=(n_boot, len(keys)))
        total_sum += sums[draw].sum(axis=1)
        total_cnt += cnts[draw].sum(axis=1)
    boot = total_sum / total_cnt[:, None]
    point = flat.mean(axis=0)
    shape = values.shape[1:]
    return point.reshape(shape), boot.reshape((n_boot,) + shape)


def pair_bootstrap(fall_values, nonfall_values, strata, n_boot=10_000, seed=0):
    """Joint resampling of matched pairs (fall window i with its matched non-fall window i), stratified by
    source: the same draw picks both members of a pair. Each fall window is its own episode, so pairs are the
    clusters on the fall side; non-fall windows that share an episode across pairs stay a stated limit.
    Returns the bootstrap means of both groups [n_boot(, K)] each."""
    f, n = np.asarray(fall_values, float), np.asarray(nonfall_values, float)
    if f.shape != n.shape:
        raise ValueError("fall and non-fall values must be paired (same shape)")
    flat_f, flat_n = f.reshape(len(f), -1), n.reshape(len(n), -1)
    rng = np.random.default_rng(seed)
    sum_f = np.zeros((n_boot, flat_f.shape[1]))
    sum_n = np.zeros((n_boot, flat_n.shape[1]))
    count = 0
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        draw = idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))]
        sum_f += flat_f[draw].sum(axis=1)
        sum_n += flat_n[draw].sum(axis=1)
        count += len(idx)
    shape = f.shape[1:]
    return (sum_f / count).reshape((n_boot,) + shape), (sum_n / count).reshape((n_boot,) + shape)


def ci(boot):
    return [float(np.percentile(boot, 2.5, axis=0)), float(np.percentile(boot, 97.5, axis=0))]


def c0_verdict(h_fall, h_fall_boot, h_non, h_non_boot, threshold=1.5):
    """Plan §2 C0 rule at d = 25 (H = -HI)."""
    if h_non <= 0:
        lo, hi = ci(h_fall_boot)
        verdict = "supported" if lo > 0 else ("not supported" if hi < 0 else "inconclusive")
        return dict(reading="H_nonfall <= 0: H_fall CI above 0", h_fall_ci=[lo, hi], verdict=verdict)
    ratio_boot = h_fall_boot / np.where(h_non_boot == 0, np.nan, h_non_boot)
    ratio_boot = ratio_boot[np.isfinite(ratio_boot)]
    lo, hi = ci(ratio_boot)
    ratio = h_fall / h_non
    if ratio >= threshold and lo > 1:
        verdict = "supported"
    elif hi < threshold:
        verdict = "not supported"
    else:
        verdict = "inconclusive"
    return dict(reading=f"ratio H_fall/H_nonfall >= {threshold} with CI excluding 1", ratio=ratio,
                ratio_ci=[lo, hi], verdict=verdict)


def reproduction_check(new, old, tolerance=REPRO_TOLERANCE):
    """Compare the three 27 Oct numbers (dicts from paired_fall_replay.json)."""
    def pick(report):
        real, imag = report["real_context"]["steps"], report["imagined_context"]["steps"]
        return {"real_step25": real[24]["uncertainty_mean"], "imagined_step1": imag[0]["uncertainty_mean"],
                "imagined_step25": imag[24]["uncertainty_mean"]}
    a, b = pick(new), pick(old)
    rows = {k: dict(old=b[k], new=a[k], rel_diff=abs(a[k] / b[k] - 1)) for k in a}
    same_inputs = new.get("trace_sha256") == old.get("trace_sha256") and new.get("wm_sha256") == old.get("wm_sha256")
    return dict(rows=rows, same_input_hashes=bool(same_inputs),
                passed=bool(all(r["rel_diff"] <= tolerance for r in rows.values())), tolerance=tolerance)


# ---------------------------------------------------------------- battery
def battery_source(model, contract, name, trace, seed, context=HISTORY, subsets=None, feedback="head",
                   actions_mode="recorded"):
    data = {k: trace[k] for k in ("state", "action", "next_state", "command", "done", "terminated", "time_out")}
    falls, counts = fall_windows(data)
    if not len(falls):
        raise ValueError(f"{name}: no fall with 32 + {HORIZON} clean rows")
    cands = nonfall_candidates(data["done"])
    nonfalls, dist = match_nonfalls(data, falls, cands, seed)
    ep = episode_index(data["done"])
    out = {}
    for kind, origins in (("fall", falls), ("nonfall", nonfalls)):
        real = real_context_pass(model, contract, data, origins, HORIZON, subsets)
        imag = imagined_pass(model, contract, data, origins, HORIZON, context, subsets, feedback, seed, actions_mode)
        first = np.abs(imag["spread"][:, 0] - real["spread"][None, 0]).max()
        stats = contract.statistics
        target = (gather(data["next_state"], origins, 0, HORIZON).transpose(1, 0, 2)
                  - stats["state_mean"].numpy()) / stats["state_std"].numpy()
        member = imag["own"].std(axis=0, ddof=1) if imag["own"].shape[0] > 1 else np.full(imag["own"].shape[1:], np.nan)
        out[kind] = dict(origins=origins, u_real=real["spread"], u_imag=imag["spread"],
                         u_member=member, stop=imag["stop"], own=imag["own"].astype(np.float32),
                         target=target.astype(np.float32), sigma_real=real["sigma"], sigma_imag=imag["sigma"],
                         ratio=imag["ratio"].astype(np.float32), tilt=imag["tilt"], gnorm=imag["gnorm"],
                         stop_mean=imag["stop_mean"],
                         tilt_real=tilt_deg(gather(data["next_state"], origins, 0, HORIZON).transpose(1, 0, 2)[..., 6:9]).numpy(),
                         se_real_norm=real["se_norm"], se_real_vel=real["se_vel"],
                         se_imag_norm=imag["se_norm"], se_imag_vel=imag["se_vel"],
                         sub_real=real["subsets"], sub_imag=imag["subsets"],
                         clusters=np.asarray([f"{name}/{n}/{ep[t, n]}" for t, n in origins]),
                         first_step_max_abs_delta=float(first))
    out["counts"] = dict(counts, nonfall_candidates=int(len(cands)),
                         match_distance_mean=float(dist.mean()), match_distance_max=float(dist.max()))
    out["match_features"] = dict(fall=match_features(data, falls).mean(0).tolist(),
                                 nonfall=match_features(data, nonfalls).mean(0).tolist(),
                                 names=["cmd_vx", "cmd_vy", "cmd_wz", "speed_xy"])
    return out


def average_ranks(x):
    """Ranks 1..n with ties given their average rank (as scipy.stats.rankdata 'average')."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman correlation with average ranks for ties; undefined (NaN) for fewer than 3 finite pairs or a
    constant input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def summarize(sources, n_boot, seed, context=HISTORY, feedback="head"):
    """HI/H/HR and u_real/u_imag per kind, component and depth, pooled and per source; errors;
    C1a member spread; C8 subsets."""
    report = {"depth_definition": BINS_NOTE, "depths": list(DEPTHS), "components": list(COMPONENTS),
              "imag_context": context, "feedback": feedback}
    names = list(sources)
    d25, s_idx = DEPTHS.index(25), list(COMPONENTS).index("sum")
    for scope in ["pooled"] + names:
        use = names if scope == "pooled" else [scope]
        block, boots, c5_parts, pair_vals, pair_strata = {}, {}, {}, {}, {}
        for kind in ("fall", "nonfall"):
            cat = lambda key, axis: np.concatenate([sources[s][kind][key] for s in use], axis=axis)
            u_real, u_imag, u_mem = cat("u_real", 1), cat("u_imag", 2), cat("u_member", 1)
            lr = np.stack([log_ratio(u_imag, u_real, c) for c in COMPONENTS], axis=-1)       # [H, W, C]
            clusters = cat("clusters", 0)
            strata = np.concatenate([[s] * len(sources[s][kind]["clusters"]) for s in use])
            point, boot = cluster_bootstrap(lr[list(DEPTHS)].transpose(1, 0, 2), clusters, strata, n_boot, seed)
            boots[kind] = boot
            pair_vals[kind] = lr[25, :, list(COMPONENTS).index("sum")]
            pair_strata[kind] = strata
            rows = {}
            for di, d in enumerate(DEPTHS):
                for ci_, c in enumerate(COMPONENTS):
                    lo, hi = COMPONENTS[c]
                    ur, ui = u_real[d, :, lo:hi].sum(-1), u_imag[:, d, :, lo:hi].sum(-1).mean(0)
                    hi_b = boot[:, di, ci_]
                    rows[f"d{d}/{c}"] = dict(
                        HI=float(point[di, ci_]), HI_ci=ci(hi_b), HR=float(np.exp(point[di, ci_])),
                        H=float(-point[di, ci_]), H_ci=sorted(float(-x) for x in ci(hi_b)),
                        u_real_mean=float(ur.mean()), u_imag_mean=float(ui.mean()),
                        u_real_geomean=float(np.exp(np.log(ur).mean())), u_imag_geomean=float(np.exp(np.log(ui).mean())))
            se_rn, se_rv = cat("se_real_norm", 1), cat("se_real_vel", 1)
            se_in, se_iv = cat("se_imag_norm", 2), cat("se_imag_vel", 2)
            errors = {f"d{d}": dict(real_norm_rmse=float(np.sqrt(se_rn[d].mean())), real_vel_rmse=float(np.sqrt(se_rv[d].mean())),
                                    imag_norm_rmse=float(np.sqrt(se_in[:, d].mean())), imag_vel_rmse=float(np.sqrt(se_iv[:, d].mean())))
                      for d in DEPTHS}
            # C1a: per-member trajectory spread vs shared-path disagreement at d = 25 (sums)
            mem25, shared25 = u_mem[25].sum(-1), u_imag[:, 25].sum(-1).mean(0)
            pm, pb = cluster_bootstrap(np.stack([mem25, shared25], 1), clusters, strata, n_boot, seed)
            c1a = dict(member_spread_d25=float(pm[0]), shared_disagreement_d25=float(pm[1]),
                       ratio=float(pm[0] / pm[1]), ratio_ci=ci(pb[:, 0] / pb[:, 1]),
                       spearman_member_spread_vs_imag_vel_error_d25=spearman(mem25, np.sqrt(se_iv[:, 25].mean(0))),
                       rule="member spread >= 2x shared disagreement AND Spearman >= 0.5 (on falls)")
            # C8: head subsets (mean over subsets of exp(HI_S) at d = 25, driving heads inside the subset)
            c8 = {}
            for K in sources[use[0]][kind]["sub_imag"]:
                sr = np.concatenate([sources[s][kind]["sub_real"][K] for s in use], axis=-1)       # [n, D, W]
                si = np.concatenate([sources[s][kind]["sub_imag"][K] for s in use], axis=-1)       # [n, heads, D, W]
                his = np.nanmean(np.log(si[:, :, d25] / sr[:, None, d25]), axis=(1, 2))           # [n]
                c8[f"K{K}"] = dict(HR_mean=float(np.exp(his).mean()), HR_p10=float(np.percentile(np.exp(his), 10)),
                                   HR_p90=float(np.percentile(np.exp(his), 90)), subsets=int(len(his)))
            c8["K_all"] = dict(HR_mean=rows["d25/sum"]["HR"], heads=int(u_imag.shape[0]))
            stop, tilt, ratio = cat("stop", 2), cat("tilt", 2), cat("ratio", 2)
            stop_m, gnorm, tilt_r = cat("stop_mean", 2), cat("gnorm", 2), cat("tilt_real", 1)
            sig_r, sig_i = cat("sigma_real", 1), cat("sigma_imag", 2)
            c5 = dict(reach_termination=float((stop > 0.5).any(axis=1).mean()),
                      reach_termination_ensemble_mean=float((stop_m > 0.5).any(axis=1).mean()),
                      reach_tilt_30deg=float((tilt >= 30).any(axis=1).mean()),
                      reach_tilt_30deg_real=float((tilt_r >= 30).any(axis=0).mean()),
                      gravity_norm={f"d{d}": float(gnorm[:, d].mean()) for d in DEPTHS},
                      sigma={f"d{d}": dict(real=float(sig_r[d].mean()), imagined=float(sig_i[:, d].mean())) for d in DEPTHS},
                      coverage_raw={f"d{d}": float((ratio[:, d] <= 1).mean()) for d in DEPTHS},
                      spearman_spread_vs_error={f"d{d}": spearman(u_imag[:, d].sum(-1).mean(0), np.sqrt(se_iv[:, d].mean(0)))
                                                for d in DEPTHS})
            c5_parts[kind] = (ratio, u_imag)
            block[kind] = dict(windows=int(len(clusters)), clusters=int(len(np.unique(clusters))), rows=rows,
                               errors=errors, c1a=c1a, c8=c8, c5=c5)
        # C5 scale: s(d) = sqrt(sum e^2 / sum u^2) on non-falls, then 1-sigma coverage on falls with u * s(d)
        r_n, u_n = c5_parts["nonfall"]
        r_f, _ = c5_parts["fall"]
        scale, cov_after = {}, {}
        for d in DEPTHS:
            e2 = ((r_n[:, d] * u_n[:, d]) ** 2).sum()
            s_d = float(np.sqrt(e2 / (u_n[:, d] ** 2).sum()))
            scale[f"d{d}"] = s_d
            cov_after[f"d{d}"] = dict(fall=float((r_f[:, d] <= s_d).mean()), nonfall=float((r_n[:, d] <= s_d).mean()))
        block["c5_scale"] = dict(s=scale, coverage_after=cov_after,
                                 rule="coverage < 30 % brought to 55-80 % by s(d) -> separate scale defect; "
                                      "Spearman < 0.5 -> scale is not the cause")
        # fall / non-fall ratio: joint resampling of matched pairs (review 18 Nov: separate same-seed
        # resampling coupled unrelated episodes)
        pf, pn = pair_bootstrap(pair_vals["fall"], pair_vals["nonfall"], pair_strata["fall"], n_boot, seed)
        block["c0_rule_d25_sum"] = c0_verdict(-block["fall"]["rows"]["d25/sum"]["HI"], -pf,
                                              -block["nonfall"]["rows"]["d25/sum"]["HI"], -pn)
        block["c0_rule_d25_sum"]["resampling"] = "matched pairs, stratified by source"
        hr0 = {k: block[k]["rows"]["d0/sum"]["HR"] for k in ("fall", "nonfall")}
        block["hr0_check"] = dict(HR0_fall=hr0["fall"], HR0_nonfall=hr0["nonfall"],
                                  passed=bool(all(0.9 <= v <= 1.1 for v in hr0.values())) if context == HISTORY else None,
                                  note=None if context == HISTORY else f"not applicable: imagination starts from {context} rows")
        report[scope] = block
    return report


def text_report(report, repro, counts):
    lines = ["Plan v6 C0 battery (CPU). " + BINS_NOTE,
             f"imagination context: {report['imag_context']} rows; feedback: {report.get('feedback', 'head')}; "
             f"actions: {report.get('actions', 'recorded')}", ""]
    if repro is not None:
        lines.append(f"Reproduction of 27 Oct (tolerance {repro['tolerance']:.0%}): "
                     f"{'PASS' if repro['passed'] else 'FAIL'}; same input hashes: {repro['same_input_hashes']}")
        for k, r in repro["rows"].items():
            lines.append(f"  {k:16s} old {r['old']:.4f}  new {r['new']:.4f}  diff {r['rel_diff']:.2%}")
        lines.append("")
    for name, c in counts.items():
        lines.append(f"{name}: {c}")
    for scope in [k for k in report if k not in ("depth_definition", "depths", "components", "imag_context", "feedback", "actions")]:
        b = report[scope]
        lines.append(f"\n== {scope}: falls {b['fall']['windows']} windows / {b['fall']['clusters']} episodes; "
                     f"non-falls {b['nonfall']['windows']} / {b['nonfall']['clusters']}")
        h0 = b["hr0_check"]
        lines.append(f"HR(0) check: fall {h0['HR0_fall']:.4f}, non-fall {h0['HR0_nonfall']:.4f} -> "
                     + (h0["note"] if h0["passed"] is None else ("PASS" if h0["passed"] else "FAIL")))
        lines.append(f"C0 rule (d=25, sum): {b['c0_rule_d25_sum']}")
        sc = b["c5_scale"]
        lines.append("C5 scale s(d) from non-falls: " + ", ".join(f"{d} {v:.2f}" for d, v in sc["s"].items())
                     + "; fall coverage after s(d): " + ", ".join(f"{d} {v['fall']:.3f}" for d, v in sc["coverage_after"].items()))
        for kind in ("fall", "nonfall"):
            k = b[kind]
            lines.append(f"  {kind}: d  component      H [95% CI]            HR      u_real   u_imag")
            for d in DEPTHS:
                for comp in COMPONENTS:
                    if d not in (0, 25) and comp != "sum":
                        continue
                    r = k["rows"][f"d{d}/{comp}"]
                    lines.append(f"    {d:2d} {comp:15s} {r['H']:+.3f} [{r['H_ci'][0]:+.3f}, {r['H_ci'][1]:+.3f}]  "
                                 f"{r['HR']:.3f}  {r['u_real_mean']:.3f}  {r['u_imag_mean']:.3f}")
            lines.append("    errors (RMSE)  d: real-context vel m/s | imagined vel m/s | real norm | imagined norm")
            for d in DEPTHS:
                e = k["errors"][f"d{d}"]
                lines.append(f"      {d:2d}: {e['real_vel_rmse']:.4f} | {e['imag_vel_rmse']:.4f} | "
                             f"{e['real_norm_rmse']:.4f} | {e['imag_norm_rmse']:.4f}")
            c = k["c1a"]
            lines.append(f"    C1a d=25 ({kind}): member spread {c['member_spread_d25']:.3f} / shared {c['shared_disagreement_d25']:.3f} "
                         f"= {c['ratio']:.2f} [{c['ratio_ci'][0]:.2f}, {c['ratio_ci'][1]:.2f}]; "
                         f"Spearman vs imagined error {c['spearman_member_spread_vs_imag_vel_error_d25']:+.2f}")
            c5 = k["c5"]
            lines.append(f"    C5/C9 ({kind}): reach termination {c5['reach_termination']:.3f} (ensemble-mean prob "
                         f"{c5['reach_termination_ensemble_mean']:.3f}), reach tilt>=30deg imagined {c5['reach_tilt_30deg']:.3f} / "
                         f"real {c5['reach_tilt_30deg_real']:.3f}; gravity norm d0/d25 {c5['gravity_norm']['d0']:.3f}/"
                         f"{c5['gravity_norm']['d25']:.3f}; sigma real/imag d1 {c5['sigma']['d1']['real']:.3f}/"
                         f"{c5['sigma']['d1']['imagined']:.3f}, d25 {c5['sigma']['d25']['real']:.3f}/{c5['sigma']['d25']['imagined']:.3f}")
            lines.append("      1-sigma coverage raw by depth: " + ", ".join(f"{d} {v:.3f}" for d, v in c5["coverage_raw"].items())
                         + "; Spearman spread vs error: " + ", ".join(
                             f"{d} {v:+.2f}" if v == v else f"{d} n/a" for d, v in c5["spearman_spread_vs_error"].items()))
            if len(k["c8"]) > 1:
                lines.append(f"    C8 d=25 ({kind}) HR by head count: " + "; ".join(
                    f"{K} {v['HR_mean']:.3f}" + (f" [{v['HR_p10']:.3f}-{v['HR_p90']:.3f}]" if "HR_p10" in v else "")
                    for K, v in k["c8"].items()))
    return "\n".join(lines) + "\n"


def json_safe(obj):
    """NaN / inf -> None (e.g. a Spearman over two windows), so the JSON stays strict."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def load_models(spec):
    """One checkpoint, or several comma-separated ones read as one ensemble (C1c)."""
    from analyse_transition_fidelity import load_model
    paths = [Path(p) for p in str(spec).split(",") if p]
    if len(paths) == 1:
        model, contract = load_model(paths[0])
        return model, contract, paths
    loaded = [load_model(p) for p in paths]
    contract = loaded[0][1]
    for _, other in loaded[1:]:
        for key, value in contract.statistics.items():
            if not torch.equal(value, other.statistics[key]):
                raise ValueError(f"models in {spec} use different normalization ({key})")
    return ModelSet(m for m, _ in loaded), contract, paths


def run(wm, traces, out, repro_trace=None, repro_json=None, n_boot=10_000, seed=0, context=HISTORY, head_subsets=(),
        feedback="head", actions_mode="recorded"):
    import replay_fall_paths as replay
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=False)
    repro = None
    if repro_trace is not None:
        replay.run(repro_trace, wm, out / "repro")
        new = json.loads((out / "repro" / "paired_fall_replay.json").read_text())
        repro = reproduction_check(new, json.loads(Path(repro_json).read_text()))
        print(f"[c0] reproduction {'PASS' if repro['passed'] else 'FAIL'}: {repro['rows']}", flush=True)
    model, contract, paths = load_models(wm)
    subsets = make_subsets(model.ensemble_size, head_subsets, seed=seed) if head_subsets else None
    sources, counts, inputs = {}, {}, {}
    for name, path in traces:
        with np.load(path, allow_pickle=False) as trace:
            validate_trace(trace)
            sources[name] = battery_source(model, contract, name, trace, seed, context, subsets, feedback, actions_mode)
        counts[name] = sources[name]["counts"]
        inputs[name] = dict(path=str(path), sha256=sha256(path))
        print(f"[c0] {name}: {counts[name]}", flush=True)
    report = summarize(sources, n_boot, seed, context, feedback)
    report["actions"] = actions_mode
    meta = dict(wm=[str(p) for p in paths], wm_sha256=[sha256(p) for p in paths], heads=model.ensemble_size,
                inputs=inputs, counts=counts, horizon=HORIZON, history=HISTORY, imag_context=context, feedback=feedback, actions=actions_mode,
                head_subsets={str(k): v.tolist() for k, v in (subsets or {}).items()},
                bootstrap=n_boot, seed=seed, analysis_sha256=sha256(__file__),
                match_features={k: v["match_features"] for k, v in sources.items()},
                first_step_max_abs_delta={k: {kind: v[kind]["first_step_max_abs_delta"] for kind in ("fall", "nonfall")}
                                          for k, v in sources.items()},
                normalization=contract.metadata(), reproduction=repro)
    with (out / "c0_report.json").open("x") as f:
        json.dump(json_safe(dict(meta=meta, report=report)), f, indent=1, allow_nan=False)
    arrays = {"meta/state_mean": contract.statistics["state_mean"].numpy(),
              "meta/state_std": contract.statistics["state_std"].numpy()}
    for name, src in sources.items():
        for kind in ("fall", "nonfall"):
            for key in ("origins", "u_real", "u_imag", "u_member", "stop", "se_real_norm", "se_real_vel",
                        "se_imag_norm", "se_imag_vel", "own", "target", "sigma_real", "sigma_imag", "ratio", "tilt",
                        "gnorm", "stop_mean", "tilt_real"):
                arrays[f"{name}/{kind}/{key}"] = src[kind][key]
            for K, v in src[kind]["sub_real"].items():
                arrays[f"{name}/{kind}/sub_real_K{K}"] = v
                arrays[f"{name}/{kind}/sub_imag_K{K}"] = src[kind]["sub_imag"][K]
            arrays[f"{name}/{kind}/clusters"] = src[kind]["clusters"].astype("U")
    with (out / "c0_windows.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    text = text_report(report, repro, counts)
    (out / "c0_report.txt").write_text(text)
    print(text, flush=True)
    failed = []
    if repro is not None and not repro["passed"]:
        failed.append("reproduction of 27 Oct outside tolerance")
    if any(report[k]["hr0_check"]["passed"] is False for k in report if isinstance(report[k], dict) and "hr0_check" in report[k]):
        failed.append("HR(0) outside 0.9-1.1")
    if failed:
        raise GateFailed("; ".join(failed) + f" (reports kept in {out})")
    return report, repro


class GateFailed(RuntimeError):
    """A pipeline gate (27 Oct reproduction, HR(0)) failed: the analysis stops (review 18 Nov)."""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", required=True,
                   help="world model checkpoint (ETH: assets/models/pretrain_rnn_ens.pt), or a,b,c (C1c: one ensemble)")
    p.add_argument("--trace", action="append", required=True, metavar="NAME=PATH",
                   help="pre-reset trace (repeat): batchA=..., batchC=..., round0_1701=...")
    p.add_argument("--out", type=Path, required=True, help="fresh output folder")
    p.add_argument("--repro-trace", type=Path, help="27 Oct trace (transition_fidelity.*/baseline.npz)")
    p.add_argument("--repro-json", type=Path, help="27 Oct paired_fall_replay.json")
    p.add_argument("--imag-context", type=int, default=HISTORY, help="real rows that start imagination (C7 cold: 1)")
    p.add_argument("--head-subsets", default="", help="C8: subset sizes, e.g. 2,5,10 (50 random subsets each)")
    p.add_argument("--actions", default="recorded", choices=ACTIONS,
                   help="actions after the context: recorded, or held at the last real action (C2 free run)")
    p.add_argument("--feedback", default="head", choices=FEEDBACK,
                   help="state fed back in imagination (C5 sampled; C9 random_step, mean, most_tilted)")
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists; use a fresh one")
    if (a.repro_trace is None) != (a.repro_json is None):
        p.error("--repro-trace and --repro-json go together")
    if not 1 <= a.imag_context <= HISTORY:
        p.error(f"--imag-context must be 1..{HISTORY}")
    try:
        sizes = tuple(int(x) for x in a.head_subsets.split(",") if x)
    except ValueError:
        p.error("--head-subsets takes integers, e.g. 2,5,10")
    traces = []
    for item in a.trace:
        name, _, path = item.partition("=")
        if not name or not path or "/" in name:
            p.error(f"bad --trace {item!r}; use NAME=PATH")
        traces.append((name, Path(path)))
    try:
        run(a.wm, traces, a.out, a.repro_trace, a.repro_json, a.bootstrap, a.seed, a.imag_context, sizes, a.feedback,
            a.actions)
    except (GateFailed, ZeroDisagreement) as exc:
        raise SystemExit(f"[c0] STOPPED: {exc}")


if __name__ == "__main__":
    main()
