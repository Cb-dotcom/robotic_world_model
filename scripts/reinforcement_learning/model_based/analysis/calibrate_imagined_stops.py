#!/usr/bin/env python3
"""Choose an imagined-episode stop rule on IMAGINED 25-step paths (27 Oct 2026).

Why: the paired fall replay (analysis/replay_fall_paths.py) showed that inside
imagination the termination probability and the disagreement fade while the real
error grows. A rule calibrated on real-context windows would fire too rarely in
training. Here every path is rolled out as imagination does: full 32-step real
context once, then the model's own mean state with the recorded actions, one fixed
head per path (as model_ids), the other heads evaluated on that same input.

Paths (25 imagined steps each):
  fall      every baseline fall with a reset-safe 32+25 context; the path ends at the
            true terminal transition (same origins as replay_fall_paths.py)
  surviving up to 2000 per actor (sampling seed 0) whose endpoint is >= 25 steps
            before any episode end (windows(..., horizon=25) group "surviving_25")
A path "stops" if the rule fires at ANY of its 25 steps.

Candidate rules: stop if head-rule(agg, tau) OR disagreement > u.
  agg in {selected, max, mean}; tau in TAUS or none; u in US or none (not both none).
Selection (fixed 27 Oct before any output was seen):
  feasible = surviving-path stop fraction <= MAX_FALSE_PATH for BOTH actors;
  choose the highest fall-path recall (pooled over path heads); ties -> the least
  intervention: no uncertainty stop, then larger u, then no head rule, then larger
  tau, then agg order selected < mean < max.
  No feasible candidate -> exit code 3.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from pre_reset_transitions import gather, sha256, validate_trace, windows  # noqa: E402

HORIZON = 25
N_SURVIVING = 2000
MAX_FALSE_PATH = 0.005          # per 25-step path (~2e-4 per step)
AGGS = ("selected", "mean", "max")
TAUS = (None, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01)
US = (None, 6.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0)


def fall_origins(trace, horizon=HORIZON, history=32):
    """Same definition as replay_fall_paths.fall_origins."""
    falls = trace["done"] & trace["terminated"] & ~trace["time_out"]
    origins = []
    for end, n in np.argwhere(falls):
        start = int(end) - horizon + 1
        if start >= history - 1 and not trace["done"][start - history + 1:end, n].any():
            origins.append((start, int(n)))
    return np.asarray(origins, dtype=int).reshape(-1, 2)


@torch.no_grad()
def imagined_paths(model, stats, trace, origins, device, batch=1024):
    """Returns term [path_head, step, n, ensemble] probabilities and epi [path_head, step, n]."""
    import termination_rule as tr
    E = model.ensemble_size
    terms, epis = [], []
    sm, ss = stats["state_mean"].to(device), stats["state_std"].to(device)
    am, ast = stats["action_mean"].to(device), stats["action_std"].to(device)
    for b in range(0, len(origins), batch):
        o = origins[b:b + batch]
        st = (torch.from_numpy(gather(trace["state"], o, -31, 1)).float().to(device) - sm) / ss
        ac = (torch.from_numpy(gather(trace["action"], o, -31, HORIZON)).float().to(device) - am) / ast
        t_heads, e_heads = [], []
        for head in range(E):
            model.reset()
            xs, xa = st, ac[:, :32]
            ids = torch.full((1, len(o), 1), head, dtype=torch.long, device=device)
            t_path, e_path = [], []
            for k in range(HORIZON):
                mean, _, epi, _, _, _ = tr.ensemble_forward_with_head_terminations(model, xs, xa, ids)
                t_path.append(torch.sigmoid(model.last_head_termination_logits).squeeze(-1).T.cpu().numpy())
                e_path.append(epi.reshape(-1).cpu().numpy())
                xs, xa = mean[:, None], ac[:, 32 + k:33 + k]
            t_heads.append(np.stack(t_path)); e_heads.append(np.stack(e_path))
        terms.append(np.stack(t_heads)); epis.append(np.stack(e_heads))
    if not terms:
        return np.zeros((E, HORIZON, 0, E)), np.zeros((E, HORIZON, 0))
    return np.concatenate(terms, axis=2), np.concatenate(epis, axis=2)


def stops(term, epi, agg, tau, u):
    """[path_head, n] bool: rule fires anywhere on the path."""
    fire = np.zeros(epi.shape, dtype=bool)
    if tau is not None:
        if agg == "selected":
            E = term.shape[0]
            p = np.stack([term[h, :, :, h] for h in range(E)])
        elif agg == "max":
            p = term.max(-1)
        else:
            p = term.mean(-1)
        fire |= p > tau
    if u is not None:
        fire |= epi > u
    return fire.any(axis=1)


def evaluate(paths):
    rows = []
    for agg, tau, u in itertools.product(AGGS, TAUS, US):
        if tau is None and u is None:
            continue
        if tau is None and agg != "selected":
            continue  # agg is irrelevant without a head rule
        row = {"agg": agg, "tau": tau, "u": u,
               "recall_fall": float(stops(*paths["fall"], agg, tau, u).mean())}
        for arm in ("baseline", "reference"):
            row[f"false_path_{arm}"] = float(stops(*paths[f"surv_{arm}"], agg, tau, u).mean())
        rows.append(row)
    eth = {"agg": "selected", "tau": 0.5, "u": None, "recall_fall": float(stops(*paths["fall"], "selected", .5, None).mean())}
    for arm in ("baseline", "reference"):
        eth[f"false_path_{arm}"] = float(stops(*paths[f"surv_{arm}"], "selected", .5, None).mean())
    return rows, eth


def select(rows):
    feasible = [r for r in rows if np.isfinite(r["recall_fall"]) and r["false_path_baseline"] <= MAX_FALSE_PATH and r["false_path_reference"] <= MAX_FALSE_PATH]
    if not feasible:
        return None
    key = lambda r: (-r["recall_fall"], r["u"] is not None, -(r["u"] or 0.0), r["tau"] is not None,
                     -(r["tau"] or 0.0), AGGS.index(r["agg"]))
    return sorted(feasible, key=key)[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output JSON (must not exist)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_surviving", type=int, default=N_SURVIVING)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.n_surviving < 1:
        ap.error("Require at least one surviving path per actor")
    from analyse_transition_fidelity import load_model
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.set_num_threads(2)
    model, contract = load_model(a.wm)
    model = model.to(device)
    model.device = device
    stats = contract.statistics
    paths, counts = {}, {}
    for arm, path in (("baseline", a.baseline), ("reference", a.reference)):
        with np.load(path, allow_pickle=False) as z:
            trace = {k: z[k] for k in z.files}
        validate_trace(trace)
        o, lab, cnt = windows(trace, 32, HORIZON, per_group=a.n_surviving)
        surv = o[lab == "surviving_25"]
        # windows() origin = last context row and endpoint = origin + HORIZON - 1, the same
        # convention as fall_origins (path predicts next_state of rows origin .. origin+24).
        paths[f"surv_{arm}"] = imagined_paths(model, stats, trace, surv, device)
        counts[f"surv_{arm}"] = int(len(surv))
        if arm == "baseline":
            fo = fall_origins(trace)
            paths["fall"] = imagined_paths(model, stats, trace, fo, device)
            counts["fall"] = int(len(fo))
        print(f"[calibrate-paths] {arm}: {counts}", flush=True)
    if counts["fall"] == 0:
        raise SystemExit("No fall path with reset-safe context")
    rows, eth = evaluate(paths)
    chosen = select(rows)
    if chosen is not None and chosen["recall_fall"] <= eth["recall_fall"]:
        chosen = None
    summary = {k: {"epi_step1_mean": float(v[1][:, 0].mean()), "epi_step25_mean": float(v[1][:, -1].mean()),
                   "epi_path_max_median": float(np.median(v[1].max(1))),
                   "term_max_heads_path_max_median": float(np.median(v[0].max(-1).max(1)))}
               for k, v in paths.items()}
    report = {"schema": "imagined_stop_paths_v2", "normalization": contract.metadata(),
              "world_model_sha256": sha256(a.wm), "horizon": HORIZON, "counts": counts,
              "recording_sha256": {"baseline": sha256(a.baseline), "reference": sha256(a.reference)},
              "scope": "Empirical rule selection on these recordings, not held-out validation. Overlapping paths and "
                       "heads are not independent replications. The measured 25-step false-stop fraction does not "
                       "certify new 256-step episode risk. Only a strict recall gain over ETH justifies training.",
              "max_false_path": MAX_FALSE_PATH, "aggs": AGGS, "taus": TAUS, "us": US,
              "selection": __doc__.split("Selection")[1].split("No feasible")[0].strip(),
              "path_summary": summary, "eth_rule": eth, "rows": rows, "chosen": chosen}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(report, f, indent=1)
    print("\npath summary:", json.dumps(summary, indent=1))
    print(f"\nETH rule (selected > 0.5): recall {eth['recall_fall']:.3f}; false base/ref "
          f"{eth['false_path_baseline']:.4f}/{eth['false_path_reference']:.4f}")
    feas = [r for r in rows if r["false_path_baseline"] <= MAX_FALSE_PATH and r["false_path_reference"] <= MAX_FALSE_PATH]
    print(f"feasible candidates: {len(feas)} of {len(rows)}; top 12 by recall:")
    print("agg       tau    u     recall  false_base  false_ref")
    for r in sorted(feas, key=lambda r: -r["recall_fall"])[:12]:
        print(f"{r['agg']:9s} {str(r['tau']):6s} {str(r['u']):5s} {r['recall_fall']:.3f}   "
              f"{r['false_path_baseline']:.4f}      {r['false_path_reference']:.4f}")
    print(f"\nCHOSEN: {chosen}")
    return 0 if chosen else 3


if __name__ == "__main__":
    raise SystemExit(main())
