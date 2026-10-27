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
from types import MethodType

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from pre_reset_transitions import gather, sha256, validate_trace, windows  # noqa: E402

GRID = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005)
RULES = ("max", "mean")
MAX_FALSE = 2e-4   # Empirical window false rate; not a certified new-episode risk bound.
HORIZONS = (1, 8)


@torch.no_grad()
def head_term_probs(model, stats, trace, origins, horizon, device, batch=2048):
    """[selected state-path head, classifier head, window] endpoint probability.

    Every classifier must see the SAME selected state path, as at deployment.
    Taking max/mean across different state paths is not the deployed rule.
    """
    from termination_rule import ensemble_forward_with_head_terminations
    model.forward = MethodType(ensemble_forward_with_head_terminations, model)
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
            heads.append(torch.sigmoid(model.last_head_termination_logits).squeeze(-1).float().cpu().numpy())
        out.append(np.stack(heads))
        print(f"[calibrate] h={horizon} windows {min(b+batch,len(origins))}/{len(origins)}", flush=True)
    return np.concatenate(out, axis=2) if out else np.zeros((model.ensemble_size, model.ensemble_size, 0))


def table(probs_by_arm, labels_by_arm):
    # A 2D input is a one-state-path test fixture; real calibration is 3D.
    probs_by_arm = {a: p[None] if p.ndim == 2 else p for a, p in probs_by_arm.items()}
    rows = []
    surv = {a: probs_by_arm[a][:, :, labels_by_arm[a] == "surviving_25"] for a in probs_by_arm}
    pos_mask = np.isin(labels_by_arm["baseline"], ("fall_now", "pre_fall_1_25"))
    groups = {g: probs_by_arm["baseline"][:, :, labels_by_arm["baseline"] == g] for g in ("fall_now", "pre_fall_1_25")}
    pos = probs_by_arm["baseline"][:, :, pos_mask]
    agg = {"max": lambda p: p.max(1), "mean": lambda p: p.mean(1)}
    for rule in RULES:
        for tau in GRID:
            row = {"rule": rule, "threshold": tau}
            for arm, p in surv.items():
                row[f"false_rate_{arm}"] = float((agg[rule](p) > tau).mean()) if p.shape[2] else float("nan")
                row[f"n_surv_{arm}"] = int(p.shape[2])
            row["recall_pooled"] = float((agg[rule](pos) > tau).mean()) if pos.shape[2] else float("nan")
            for g, p in groups.items():
                row[f"recall_{g}"] = float((agg[rule](p) > tau).mean()) if p.shape[2] else float("nan")
            rows.append(row)
    # ETH's deployed rule for reference: a random selected head > 0.5 (average over heads)
    eth = {"rule": "selected(eth)", "threshold": 0.5}
    def selected(p):
        if p.shape[0] == 1:  # simple one-path fixtures: average over random heads
            return p[0]
        return np.stack([p[h, h] for h in range(p.shape[0])])
    for arm, p in surv.items():
        eth[f"false_rate_{arm}"] = float((selected(p) > 0.5).mean()) if p.shape[2] else float("nan")
        eth[f"n_surv_{arm}"] = int(p.shape[2])
    eth["recall_pooled"] = float((selected(pos) > 0.5).mean()) if pos.shape[2] else float("nan")
    for g, p in groups.items():
        eth[f"recall_{g}"] = float((selected(p) > 0.5).mean()) if p.shape[2] else float("nan")
    return rows, eth


def select(rows):
    feasible = [r for r in rows if np.isfinite(r["recall_pooled"]) and r["false_rate_baseline"] <= MAX_FALSE and r["false_rate_reference"] <= MAX_FALSE]
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
    if a.out.exists():
        raise FileExistsError(a.out)
    from analyse_transition_fidelity import load_model
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.set_num_threads(2)
    model, contract = load_model(a.wm)
    model = model.to(device)
    model.device = device
    stats = contract.statistics
    report = {"schema": "same_path_all_head_termination_v2", "world_model_sha256": sha256(a.wm), "grid": GRID, "rules": RULES, "max_false_rate": MAX_FALSE,
              "normalization": contract.metadata(),
              "scope": "Empirical selection on these recordings; no held-out calibration claim. Windows overlap. "
                       "Aggregate classifiers within each selected state-head path, then average uniformly over paths/windows. "
                       "Endpoint window rates do not certify cumulative rollout risk or future-policy generalization.",
              "selection": "horizon 8; false rate <= max on both actors' surviving windows; max pooled recall; "
                           "ties -> larger threshold, then mean", "horizons": {}}
    for horizon in HORIZONS:
        probs, labels, counts = {}, {}, {}
        for arm, path in (("baseline", a.baseline), ("reference", a.reference)):
            with np.load(path, allow_pickle=False) as z:
                trace = {k: z[k] for k in z.files}
            validate_trace(trace)
            report.setdefault("recordings", {})[arm] = {"sha256": sha256(path), "metadata": json.loads(str(trace["metadata"]))}
            origins, lab, cnt = windows(trace, 32, horizon, per_group=a.max_windows_per_group)
            probs[arm] = head_term_probs(model, stats, trace, origins, horizon, device)
            labels[arm], counts[arm] = lab, cnt
            print(f"[calibrate] h={horizon} {arm}: {cnt['selected']}", flush=True)
        rows, eth = table(probs, labels)
        report["horizons"][str(horizon)] = {"counts": counts, "eth_rule": eth, "rows": rows,
                                            "selected": select(rows)}
    chosen = report["horizons"]["8"]["selected"]
    if chosen is not None and chosen["recall_pooled"] <= report["horizons"]["8"]["eth_rule"]["recall_pooled"]:
        chosen = None
        report["stop_reason"] = "No strict recall improvement over the deployed selected-head rule"
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
