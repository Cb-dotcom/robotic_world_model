#!/usr/bin/env python3
"""Choose an imagination termination rule from true pre-reset transitions (27 Oct 2026).

Input: the two recordings of run_transition_diagnostic.py (baseline = falling offline
policy, reference = ETH's walking actor) and ETH's world model. Uses ALL eligible
windows (not the 256-window sample of the fidelity report). Read-only.

For horizons 1 and 8 (32-step real history, then fixed-head mean forecasts with the
recorded actions, as in analyse_transition_fidelity.py) it computes every head's
termination probability at the window end and, for each rule in {max, mean} and each
threshold in GRID, the per-step false-termination rate on "surviving_25" windows of
each actor and the recall on the baseline's "fall_now" and "pre_fall_1_25" windows.

Selection rule (fixed 27 Oct before any calibration output was seen):
  horizon 8; feasible = false rate <= MAX_FALSE per step on BOTH actors' surviving
  windows; choose the feasible (rule, threshold) with the highest recall on the pooled
  baseline fall_now + pre_fall_1_25 windows; ties -> larger threshold, then "mean".
  No feasible candidate -> exit code 3 and no rule (no training should follow).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from pre_reset_transitions import gather, sha256, validate_trace, windows  # noqa: E402

GRID = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005)
RULES = ("max", "mean")
MAX_FALSE = 2e-4   # per imagined step: ~5% of 256-step imagined episodes cut falsely
HORIZONS = (1, 8)


@torch.no_grad()
def head_term_probs(model, stats, trace, origins, horizon, device, batch=2048):
    """[ensemble, n] termination probability at the window end, per fixed head."""
    out = []
    sm, ss = stats["state_mean"].to(device), stats["state_std"].to(device)
    am, ast = stats["action_mean"].to(device), stats["action_std"].to(device)
    for b in range(0, len(origins), batch):
        o = origins[b:b + batch]
        st = (torch.from_numpy(gather(trace["state"], o, -31, 1)).float().to(device) - sm) / ss
        ac = (torch.from_numpy(gather(trace["action"], o, -31, horizon)).float().to(device) - am) / ast
        heads = []
        for head in range(model.ensemble_size):
            model.reset()
            xs, xa = st, ac[:, :32]
            ids = torch.full((1, len(o), 1), head, dtype=torch.long, device=device)
            for k in range(horizon):
                mean, _, _, _, _, term = model(xs, xa, ids)
                if k < horizon - 1:
                    xs, xa = mean[:, None], ac[:, 32 + k:33 + k]
            heads.append(torch.sigmoid(term).squeeze(-1).float().cpu().numpy())
        out.append(np.stack(heads))
    return np.concatenate(out, axis=1) if out else np.zeros((model.ensemble_size, 0))


def table(probs_by_arm, labels_by_arm):
    rows = []
    surv = {a: probs_by_arm[a][:, labels_by_arm[a] == "surviving_25"] for a in probs_by_arm}
    pos_mask = np.isin(labels_by_arm["baseline"], ("fall_now", "pre_fall_1_25"))
    groups = {g: probs_by_arm["baseline"][:, labels_by_arm["baseline"] == g] for g in ("fall_now", "pre_fall_1_25")}
    pos = probs_by_arm["baseline"][:, pos_mask]
    agg = {"max": lambda p: p.max(0), "mean": lambda p: p.mean(0)}
    for rule in RULES:
        for tau in GRID:
            row = {"rule": rule, "threshold": tau}
            for arm, p in surv.items():
                row[f"false_rate_{arm}"] = float((agg[rule](p) > tau).mean()) if p.shape[1] else float("nan")
                row[f"n_surv_{arm}"] = int(p.shape[1])
            row["recall_pooled"] = float((agg[rule](pos) > tau).mean()) if pos.shape[1] else float("nan")
            for g, p in groups.items():
                row[f"recall_{g}"] = float((agg[rule](p) > tau).mean()) if p.shape[1] else float("nan")
            rows.append(row)
    # ETH's deployed rule for reference: a random selected head > 0.5 (average over heads)
    eth = {"rule": "selected(eth)", "threshold": 0.5}
    for arm, p in surv.items():
        eth[f"false_rate_{arm}"] = float((p > 0.5).mean()) if p.shape[1] else float("nan")
        eth[f"n_surv_{arm}"] = int(p.shape[1])
    eth["recall_pooled"] = float((pos > 0.5).mean()) if pos.shape[1] else float("nan")
    for g, p in groups.items():
        eth[f"recall_{g}"] = float((p > 0.5).mean()) if p.shape[1] else float("nan")
    return rows, eth


def select(rows):
    feasible = [r for r in rows if r["false_rate_baseline"] <= MAX_FALSE and r["false_rate_reference"] <= MAX_FALSE]
    if not feasible:
        return None
    return sorted(feasible, key=lambda r: (-r["recall_pooled"], -r["threshold"], r["rule"] != "mean"))[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output JSON (must not exist)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_windows_per_group", type=int, default=10**9)
    a = ap.parse_args(argv)
    from analyse_transition_fidelity import load_model
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.set_num_threads(2)
    model, contract = load_model(a.wm)
    model = model.to(device)
    model.device = device
    stats = contract.statistics
    report = {"world_model_sha256": sha256(a.wm), "grid": GRID, "rules": RULES, "max_false_rate": MAX_FALSE,
              "selection": "horizon 8; false rate <= max on both actors' surviving windows; max pooled recall; "
                           "ties -> larger threshold, then mean", "horizons": {}}
    for horizon in HORIZONS:
        probs, labels, counts = {}, {}, {}
        for arm, path in (("baseline", a.baseline), ("reference", a.reference)):
            with np.load(path, allow_pickle=False) as z:
                trace = {k: z[k] for k in z.files}
            validate_trace(trace)
            origins, lab, cnt = windows(trace, 32, horizon, per_group=a.max_windows_per_group)
            probs[arm] = head_term_probs(model, stats, trace, origins, horizon, device)
            labels[arm], counts[arm] = lab, cnt
            print(f"[calibrate] h={horizon} {arm}: {cnt['selected']}", flush=True)
        rows, eth = table(probs, labels)
        report["horizons"][str(horizon)] = {"counts": counts, "eth_rule": eth, "rows": rows,
                                            "selected": select(rows)}
    chosen = report["horizons"]["8"]["selected"]
    report["chosen"] = chosen
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(report, f, indent=1)
    for h in HORIZONS:
        r = report["horizons"][str(h)]
        e = r["eth_rule"]
        print(f"\n== horizon {h} == ETH rule (selected head > 0.5): false base/ref "
              f"{e['false_rate_baseline']:.2e}/{e['false_rate_reference']:.2e}; recall fall_now "
              f"{e['recall_fall_now']:.3f}, pre_fall_1_25 {e['recall_pre_fall_1_25']:.3f}")
        print("rule  tau    false_base  false_ref  recall_pooled  fall_now  pre_1_25")
        for row in r["rows"]:
            print(f"{row['rule']:5s} {row['threshold']:<6} {row['false_rate_baseline']:.2e}    {row['false_rate_reference']:.2e}   "
                  f"{row['recall_pooled']:.3f}          {row['recall_fall_now']:.3f}     {row['recall_pre_fall_1_25']:.3f}")
    print(f"\nCHOSEN (horizon 8): {chosen}")
    return 0 if chosen else 3


if __name__ == "__main__":
    raise SystemExit(main())
