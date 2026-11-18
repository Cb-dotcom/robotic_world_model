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
@torch.no_grad()
def forward_all(model, xs, xa):
    """One recurrent step of SystemDynamicsEnsemble.forward, returning every head's state mean.

    Same calls in the same order as the library forward (state trunk, state heads, auxiliary trunk,
    auxiliary heads), so the hidden states advance exactly as in `model(xs, xa, ids)`.
    Returns means [heads, B, 45], contact logits [heads, B, 8], termination logits [heads, B, 1].
    """
    base = model.state_base(xs, xa)
    means = []
    for head in model.state_heads:
        mean, _ = head(base, xs)
        if model.prediction_type == "sequence":
            mean = mean[:, -1]
        means.append(mean)
    aux = model.auxiliary_base(xs, xa)
    contacts, terms = [], []
    for head in model.auxiliary_heads:
        _, contact, term = head(aux, xs)
        if model.prediction_type == "sequence":
            contact, term = contact[:, -1], term[:, -1]
        contacts.append(contact)
        terms.append(term)
    return torch.stack(means), torch.stack(contacts), torch.stack(terms)


def head_spread(means):
    """Per-dimension std over heads (unbiased, as the library's epistemic uncertainty) [B, 45]."""
    return means.std(dim=0)


def _normalized(contract, data, origins, lo, hi, key):
    stats = contract.statistics
    values = torch.from_numpy(gather(data[key], origins, lo, hi)).float()
    return (values - stats[f"{key}_mean"]) / stats[f"{key}_std"]


@torch.no_grad()
def real_context_spread(model, contract, data, origins, horizon, batch=256):
    """Teacher forcing: spread [horizon, W, 45] of the prediction of step k from 32 real rows."""
    rows = np.asarray([(t + k, n) for k in range(horizon) for t, n in origins], dtype=int).reshape(-1, 2)
    out = []
    for b in range(0, len(rows), batch):
        chunk = rows[b:b + batch]
        model.reset()
        means, _, _ = forward_all(model, _normalized(contract, data, chunk, -31, 1, "state"),
                                  _normalized(contract, data, chunk, -31, 1, "action"))
        out.append(head_spread(means).numpy())
    return np.concatenate(out).reshape(horizon, len(origins), -1)


@torch.no_grad()
def imagined_spread(model, contract, data, origins, horizon):
    """Imagination driven by each head's own mean: spread [heads, horizon, W, 45] and the driving
    head's termination probability [heads, horizon, W]."""
    states = _normalized(contract, data, origins, -31, 1, "state")
    actions = _normalized(contract, data, origins, -31, horizon, "action")
    spread, stop = [], []
    for h in range(model.ensemble_size):
        model.reset()
        xs, xa = states, actions[:, :32]
        sp, st = [], []
        for k in range(horizon):
            means, _, terms = forward_all(model, xs, xa)
            sp.append(head_spread(means).numpy())
            st.append(torch.sigmoid(terms[h]).squeeze(-1).numpy())
            xs, xa = means[h][:, None], actions[:, 32 + k:33 + k]
        spread.append(np.stack(sp))
        stop.append(np.stack(st))
    return np.stack(spread), np.stack(stop)


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
def log_ratio(u_imag, u_real, component):
    """Per window and depth: mean over driving heads of log(u_imag / u_real) for one component.
    u_imag [heads, H, W, 45], u_real [H, W, 45] -> [H, W]."""
    lo, hi = COMPONENTS[component]
    ui, ur = u_imag[..., lo:hi].sum(-1), u_real[..., lo:hi].sum(-1)
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
def battery_source(model, contract, name, trace, seed):
    data = {k: trace[k] for k in ("state", "action", "next_state", "command", "done", "terminated", "time_out")}
    falls, counts = fall_windows(data)
    if not len(falls):
        raise ValueError(f"{name}: no fall with 32 + {HORIZON} clean rows")
    cands = nonfall_candidates(data["done"])
    nonfalls, dist = match_nonfalls(data, falls, cands, seed)
    ep = episode_index(data["done"])
    out = {}
    for kind, origins in (("fall", falls), ("nonfall", nonfalls)):
        u_real = real_context_spread(model, contract, data, origins, HORIZON)
        u_imag, stop = imagined_spread(model, contract, data, origins, HORIZON)
        first = np.abs(u_imag[:, 0] - u_real[None, 0]).max()
        out[kind] = dict(origins=origins, u_real=u_real, u_imag=u_imag, stop=stop,
                         clusters=np.asarray([f"{name}/{n}/{ep[t, n]}" for t, n in origins]),
                         first_step_max_abs_delta=float(first))
    out["counts"] = dict(counts, nonfall_candidates=int(len(cands)),
                         match_distance_mean=float(dist.mean()), match_distance_max=float(dist.max()))
    out["match_features"] = dict(fall=match_features(data, falls).mean(0).tolist(),
                                 nonfall=match_features(data, nonfalls).mean(0).tolist(),
                                 names=["cmd_vx", "cmd_vy", "cmd_wz", "speed_xy"])
    return out


def summarize(sources, n_boot, seed):
    """HI/H/HR and u_real/u_imag per kind, component and depth, pooled and per source."""
    report = {"depth_definition": BINS_NOTE, "depths": list(DEPTHS), "components": list(COMPONENTS)}
    names = list(sources)
    for scope in ["pooled"] + names:
        use = names if scope == "pooled" else [scope]
        block = {}
        boots = {}
        for kind in ("fall", "nonfall"):
            lr = np.concatenate([np.stack([log_ratio(sources[s][kind]["u_imag"], sources[s][kind]["u_real"], c)
                                           for c in COMPONENTS], axis=-1) for s in use], axis=1)  # [H, W, C]
            clusters = np.concatenate([sources[s][kind]["clusters"] for s in use])
            strata = np.concatenate([[s] * len(sources[s][kind]["clusters"]) for s in use])
            vals = lr[list(DEPTHS)].transpose(1, 0, 2)                                         # [W, D, C]
            point, boot = cluster_bootstrap(vals, clusters, strata, n_boot, seed)
            boots[kind] = boot
            rows = {}
            for di, d in enumerate(DEPTHS):
                for ci_, c in enumerate(COMPONENTS):
                    lo, hi = COMPONENTS[c]
                    ur = np.concatenate([sources[s][kind]["u_real"][d, :, lo:hi].sum(-1) for s in use])
                    ui = np.concatenate([sources[s][kind]["u_imag"][:, d, :, lo:hi].sum(-1).mean(0) for s in use])
                    hi_b = boot[:, di, ci_]
                    rows[f"d{d}/{c}"] = dict(
                        HI=float(point[di, ci_]), HI_ci=ci(hi_b), HR=float(np.exp(point[di, ci_])),
                        H=float(-point[di, ci_]), H_ci=sorted(float(-x) for x in ci(hi_b)),
                        u_real_mean=float(ur.mean()), u_imag_mean=float(ui.mean()),
                        u_real_geomean=float(np.exp(np.log(ur).mean())), u_imag_geomean=float(np.exp(np.log(ui).mean())))
            block[kind] = dict(windows=int(len(clusters)), clusters=int(len(np.unique(clusters))), rows=rows)
        d25, s = DEPTHS.index(25), list(COMPONENTS).index("sum")
        block["c0_rule_d25_sum"] = c0_verdict(-block["fall"]["rows"]["d25/sum"]["HI"], -boots["fall"][:, d25, s],
                                              -block["nonfall"]["rows"]["d25/sum"]["HI"], -boots["nonfall"][:, d25, s])
        block["hr0_check"] = dict(HR0_fall=block["fall"]["rows"]["d0/sum"]["HR"],
                                  HR0_nonfall=block["nonfall"]["rows"]["d0/sum"]["HR"],
                                  passed=bool(all(0.9 <= block[k]["rows"]["d0/sum"]["HR"] <= 1.1 for k in ("fall", "nonfall"))))
        report[scope] = block
    return report


def text_report(report, repro, counts):
    lines = ["Plan v6 C0 battery (CPU). " + BINS_NOTE, ""]
    if repro is not None:
        lines.append(f"Reproduction of 27 Oct (tolerance {repro['tolerance']:.0%}): "
                     f"{'PASS' if repro['passed'] else 'FAIL'}; same input hashes: {repro['same_input_hashes']}")
        for k, r in repro["rows"].items():
            lines.append(f"  {k:16s} old {r['old']:.4f}  new {r['new']:.4f}  diff {r['rel_diff']:.2%}")
        lines.append("")
    for name, c in counts.items():
        lines.append(f"{name}: {c}")
    for scope in [k for k in report if k not in ("depth_definition", "depths", "components")]:
        b = report[scope]
        lines.append(f"\n== {scope}: falls {b['fall']['windows']} windows / {b['fall']['clusters']} episodes; "
                     f"non-falls {b['nonfall']['windows']} / {b['nonfall']['clusters']}")
        lines.append(f"HR(0) check: fall {b['hr0_check']['HR0_fall']:.4f}, non-fall {b['hr0_check']['HR0_nonfall']:.4f} "
                     f"-> {'PASS' if b['hr0_check']['passed'] else 'FAIL'}")
        lines.append(f"C0 rule (d=25, sum): {b['c0_rule_d25_sum']}")
        for kind in ("fall", "nonfall"):
            lines.append(f"  {kind}: d  component      H [95% CI]            HR      u_real   u_imag")
            for d in DEPTHS:
                for comp in COMPONENTS:
                    if d not in (0, 25) and comp != "sum":
                        continue
                    r = b[kind]["rows"][f"d{d}/{comp}"]
                    lines.append(f"    {d:2d} {comp:15s} {r['H']:+.3f} [{r['H_ci'][0]:+.3f}, {r['H_ci'][1]:+.3f}]  "
                                 f"{r['HR']:.3f}  {r['u_real_mean']:.3f}  {r['u_imag_mean']:.3f}")
    return "\n".join(lines) + "\n"


def run(wm, traces, out, repro_trace=None, repro_json=None, n_boot=10_000, seed=0):
    from analyse_transition_fidelity import load_model
    import replay_fall_paths as replay
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=False)
    repro = None
    if repro_trace is not None:
        replay.run(repro_trace, wm, out / "repro")
        new = json.loads((out / "repro" / "paired_fall_replay.json").read_text())
        repro = reproduction_check(new, json.loads(Path(repro_json).read_text()))
        print(f"[c0] reproduction {'PASS' if repro['passed'] else 'FAIL'}: {repro['rows']}", flush=True)
    model, contract = load_model(wm)
    sources, counts, inputs = {}, {}, {}
    for name, path in traces:
        with np.load(path, allow_pickle=False) as trace:
            validate_trace(trace)
            sources[name] = battery_source(model, contract, name, trace, seed)
        counts[name] = sources[name]["counts"]
        inputs[name] = dict(path=str(path), sha256=sha256(path))
        print(f"[c0] {name}: {counts[name]}", flush=True)
    report = summarize(sources, n_boot, seed)
    meta = dict(wm=str(wm), wm_sha256=sha256(wm), inputs=inputs, counts=counts, horizon=HORIZON,
                history=HISTORY, bootstrap=n_boot, seed=seed, analysis_sha256=sha256(__file__),
                match_features={k: v["match_features"] for k, v in sources.items()},
                first_step_max_abs_delta={k: {kind: v[kind]["first_step_max_abs_delta"] for kind in ("fall", "nonfall")}
                                          for k, v in sources.items()},
                normalization=contract.metadata(), reproduction=repro)
    with (out / "c0_report.json").open("x") as f:
        json.dump(dict(meta=meta, report=report), f, indent=1, allow_nan=False)
    arrays = {}
    for name, src in sources.items():
        for kind in ("fall", "nonfall"):
            for key in ("origins", "u_real", "u_imag", "stop"):
                arrays[f"{name}/{kind}/{key}"] = src[kind][key]
            arrays[f"{name}/{kind}/clusters"] = src[kind]["clusters"].astype("U")
    with (out / "c0_windows.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    text = text_report(report, repro, counts)
    (out / "c0_report.txt").write_text(text)
    print(text, flush=True)
    return report, repro


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True, help="world model checkpoint (ETH: assets/models/pretrain_rnn_ens.pt)")
    p.add_argument("--trace", action="append", required=True, metavar="NAME=PATH",
                   help="pre-reset trace (repeat): batchA=..., batchC=..., round0_1701=...")
    p.add_argument("--out", type=Path, required=True, help="fresh output folder")
    p.add_argument("--repro-trace", type=Path, help="27 Oct trace (transition_fidelity.*/baseline.npz)")
    p.add_argument("--repro-json", type=Path, help="27 Oct paired_fall_replay.json")
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists; use a fresh one")
    if (a.repro_trace is None) != (a.repro_json is None):
        p.error("--repro-trace and --repro-json go together")
    traces = []
    for item in a.trace:
        name, _, path = item.partition("=")
        if not name or not path or "/" in name:
            p.error(f"bad --trace {item!r}; use NAME=PATH")
        traces.append((name, Path(path)))
    run(a.wm, traces, a.out, a.repro_trace, a.repro_json, a.bootstrap, a.seed)


if __name__ == "__main__":
    main()
