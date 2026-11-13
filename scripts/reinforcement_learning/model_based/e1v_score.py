#!/usr/bin/env python3
"""E1 validation feedback cycle (plan v5.2): score pre-reset traces on one or more world models. CPU.

Uses the frozen transition-fidelity analysis (H=32, up to 256 windows per stratum, sampling seed 0),
so every model is scored on identical windows of a trace. Reports per horizon and window group:
velocity RMSE (m/s), mean normalized epistemic disagreement, mean termination probability and
per-head termination recall on fall_now. No bars: the feedback cycle checks that the path works
and records the numbers.

  e1v_score.py --model eth=PATH --model updated=PATH --trace batch=PATH --horizons 1,8,24 --out DIR
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
from pre_reset_transitions import sha256  # noqa: E402


def pairs(values, flag):
    out = {}
    for item in values:
        name, sep, path = item.partition("=")
        if not sep or not name or name in out:
            raise SystemExit(f"{flag} needs unique NAME=PATH entries, got {item!r}")
        out[name] = Path(path)
    return out


def lines_for(report, horizons):
    lines = [f"  falls {report['fall_events']}, time-outs {report['timeouts']}"]
    for h in horizons:
        for group, g in report["horizons"][str(h)]["groups"].items():
            if not g.get("windows"):
                lines.append(f"  h{h} {group}: no windows")
                continue
            recall = g.get("term_recall_at_deployment_threshold_per_head")
            lines.append(
                f"  h{h} {group} (n={g['windows']}): vel RMSE {g['rmse_physical_units_per_head_mean']['lin_vel_m_s']:.4f} m/s"
                f"; epi {g['epistemic_sum_std_normalized_mean']:.3f}; term p {g['term_probability_mean']:.3f}"
                + (f"; term recall/head {[round(x, 2) for x in recall]}" if recall is not None and group == "fall_now" else ""))
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, help="NAME=PATH (repeatable)")
    p.add_argument("--trace", action="append", required=True, help="NAME=PATH (repeatable)")
    p.add_argument("--horizons", default="1,8,24")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists():
        p.error("--out must not exist")
    torch.set_num_threads(2)
    horizons = tuple(int(h) for h in args.horizons.split(","))
    models, traces = pairs(args.model, "--model"), pairs(args.trace, "--trace")
    result = {"horizons": horizons, "models": {}, "traces": {n: {"path": str(t), "sha256": sha256(t)} for n, t in traces.items()},
              "scores": {}}
    text = []
    norm = set()
    for mname, mpath in models.items():
        model, contract = load_model(mpath)
        norm.add(contract.metadata()["norm_stats_sha256"])
        result["models"][mname] = {"path": str(mpath), "sha256": sha256(mpath),
                                   "norm_stats_sha256": contract.metadata()["norm_stats_sha256"]}
        for tname, tpath in traces.items():
            report = analyse(tpath, model, contract, horizons)
            result["scores"][f"{mname}|{tname}"] = report
            text += [f"== model {mname} on trace {tname}"] + lines_for(report, horizons)
    if len(norm) != 1:
        raise SystemExit(f"Models use different normalizers: {sorted(norm)}")
    args.out.mkdir(parents=True)
    (args.out / "scores.json").write_text(json.dumps(result, indent=2, default=str))
    (args.out / "scores.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
