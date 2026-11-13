#!/usr/bin/env python3
"""E1 model check: ETH model before vs fine-tuned model after, on held-out envs only. CPU.

Uses the frozen transition-fidelity analysis (H=32, horizons 1/8, up to 256 windows per
stratum, sampling seed 0), so before and after are scored on identical windows.

Pre-registered bars (plan v5.1, E1):
  A  8-step velocity RMSE, pre_fall_1_25 windows of the falling policy: after <= 0.70 x before
  B  8-step velocity RMSE, surviving_25 windows of the reference policy: after <= 1.50 x before
One velocity bar does not clear the whole model; all components are reported.
Plan v5.2 adds (reported, no bar): horizon 24 and per-head termination recall on fall_now.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

MB = Path(__file__).resolve().parent
sys.path.insert(0, str(MB / "analysis"))
from analyse_transition_fidelity import analyse, load_model  # noqa: E402
from pre_reset_transitions import GROUPS, sha256  # noqa: E402

BAR_A, BAR_B = 0.70, 1.50


def vel(report, arm, group, horizon="8"):
    g = report[arm]["horizons"][horizon]["groups"][group]
    return g["rmse_physical_units_per_head_mean"]["lin_vel_m_s"] if g.get("windows") else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eth", type=Path, required=True)
    p.add_argument("--finetuned", type=Path, required=True)
    p.add_argument("--baseline-test", type=Path, required=True)
    p.add_argument("--reference-test", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline-tag", default="baseline1601")
    p.add_argument("--horizons", default="1,8", help="comma list; must include 8 (E1-V: 1,8,24)")
    args = p.parse_args(argv)
    torch.set_num_threads(2)
    horizons = tuple(int(h) for h in args.horizons.split(","))
    if 8 not in horizons:
        raise SystemExit("--horizons must include 8 (the bars use 8 steps)")
    tag = args.baseline_tag
    reports, contracts = {}, {}
    for name, wm in (("before", args.eth), ("after", args.finetuned)):
        model, contract = load_model(wm)
        contracts[name] = contract.metadata()
        reports[name] = {arm: analyse(path, model, contract, horizons)
                         for arm, path in ((tag, args.baseline_test), ("reference", args.reference_test))}
    if contracts["before"]["norm_stats_sha256"] != contracts["after"]["norm_stats_sha256"]:
        raise SystemExit("Normalizer differs between ETH and fine-tuned model: the comparison is invalid")

    a0, a1 = vel(reports["before"], tag, "pre_fall_1_25"), vel(reports["after"], tag, "pre_fall_1_25")
    b0, b1 = vel(reports["before"], "reference", "surviving_25"), vel(reports["after"], "reference", "surviving_25")
    verdict = {
        "A_prefall_velocity_8step": {"before": a0, "after": a1, "ratio": (a1 / a0) if a0 and a1 is not None else None,
                                     "bar": BAR_A, "pass": (a1 is not None and a0 and a1 <= BAR_A * a0)},
        "B_reference_velocity_8step": {"before": b0, "after": b1, "ratio": (b1 / b0) if b0 and b1 is not None else None,
                                       "bar": BAR_B, "pass": (b1 is not None and b0 and b1 <= BAR_B * b0)},
    }
    verdict["model_check_pass"] = bool(verdict["A_prefall_velocity_8step"]["pass"] and verdict["B_reference_velocity_8step"]["pass"])

    lines = ["E1 model check (held-out envs; fixed-head mean replay of logged actions)",
             f"normalizer sha256: {contracts['after']['norm_stats_sha256']}"]
    for key in ("A_prefall_velocity_8step", "B_reference_velocity_8step"):
        v = verdict[key]
        lines.append(f"{key}: before={v['before']} after={v['after']} ratio={v['ratio']} bar={v['bar']} pass={v['pass']}")
    lines.append(f"MODEL CHECK PASS: {verdict['model_check_pass']}")
    lines.append("\nAll components, RMSE in physical units (before -> after):")
    for arm in (tag, "reference"):
        for horizon in (str(h) for h in horizons):
            for group in ("pre_fall_1_25", "fall_now", "surviving_25"):
                g0 = reports["before"][arm]["horizons"][horizon]["groups"][group]
                g1 = reports["after"][arm]["horizons"][horizon]["groups"][group]
                if not g0.get("windows"):
                    continue
                parts = [f"{k} {g0['rmse_physical_units_per_head_mean'][k]:.4f}->{g1['rmse_physical_units_per_head_mean'][k]:.4f}"
                         for k in GROUPS]
                lines.append(f"  {arm} h{horizon} {group} (n={g0['windows']}): " + "; ".join(parts)
                             + f"; term {g0['term_probability_mean']:.3f}->{g1['term_probability_mean']:.3f}"
                             + f"; epi {g0['epistemic_sum_std_normalized_mean']:.3f}->{g1['epistemic_sum_std_normalized_mean']:.3f}")
                if group == "fall_now":
                    lines.append(f"    term recall per head: {g0['term_recall_at_deployment_threshold_per_head']} -> "
                                 f"{g1['term_recall_at_deployment_threshold_per_head']}")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "report.json").open("x") as f:
        json.dump({"verdict": verdict, "contracts": contracts, "eth_sha256": sha256(args.eth),
                   "finetuned_sha256": sha256(args.finetuned), "reports": reports}, f, indent=2, allow_nan=False)
    with (args.out / "report.txt").open("x") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
