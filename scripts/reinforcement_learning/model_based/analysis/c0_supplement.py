#!/usr/bin/env python3
"""Plan v6 amendment A3 §1 (21 Nov 2026): two supplementary C0 analyses from the saved windows. No verdict.

Input: c0_windows.npz (and c0_report.json) of the registered C0 run (rerun 2, 18 Nov: ETH's checkpoint,
sources A, C and 1701 round 0). The registered statistic is unchanged: per window, LR(d) = mean over driving
heads of log(u_imag / u_ctx) on the summed spread; H(d) = -mean over windows of LR(d) (every window weight 1,
pooled over sources); the C0 rule reads H_fall / H_nonfall at d = 25 (the simulator terminal transition) with
the episode-cluster pair bootstrap (pair components, stratified by source, 10,000 draws, seed 0).

(a) Exact decomposition. Because log(u_imag / u_ctx) = log u_imag - log u_ctx,
       H = C_ctx - C_imag,  C_ctx = mean log u_ctx,  C_imag = mean over windows and driving heads of log u_imag,
       H_fall - H_nonfall = (C_ctx,fall - C_ctx,nonfall) - (C_imag,fall - C_imag,nonfall)
    i.e. the recorded-context part and the imagination part of the fall / non-fall difference, with
    bootstrap intervals from the SAME draws as the registered ratio.
(b) The same statistics at d = 16: 9 imagined steps (180 ms at 50 Hz) before the terminal target.
Geometric means are reported as exp(mean log): exp(-H) is the geometric mean of u_imag / u_ctx.
d = 25 stays the registered endpoint; these numbers are supplementary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c0_battery import ZeroDisagreement, ci, log_ratio, pair_bootstrap  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

DEPTHS = (25, 16)
STEP_MS = 20


def load_sources(path):
    """{source: {kind: (log_ctx [H, W], log_imag [H, W], clusters [W], registered LR [H, W])}} in the saved order.
    log_ctx and log_imag are float64 (the decomposition); the registered LR is c0_battery.log_ratio on the saved
    float32 spreads, exactly as the registered run computed it (the reproduction check)."""
    out = {}
    with np.load(path, allow_pickle=False) as z:
        names = []
        for key in z.files:
            parts = key.split("/")
            if len(parts) == 3 and parts[1] == "fall" and parts[2] == "u_real" and parts[0] not in names:
                names.append(parts[0])
        for name in names:
            out[name] = {}
            for kind in ("fall", "nonfall"):
                u_real, u_imag = z[f"{name}/{kind}/u_real"], z[f"{name}/{kind}/u_imag"]
                lr = log_ratio(u_imag, u_real, "sum")                          # registered path, float32 [H, W]
                ur = u_real.astype(np.float64).sum(-1)                          # [H, W]
                ui = u_imag.astype(np.float64).sum(-1)                          # [heads, H, W]
                if (ur <= 0).any() or (ui <= 0).any():
                    raise ZeroDisagreement(f"{name}/{kind}: zero summed spread")
                out[name][kind] = (np.log(ur), np.log(ui).mean(0), z[f"{name}/{kind}/clusters"].astype(str), lr)
    return out


def analyse(sources, n_boot=10_000, seed=0):
    names = list(sources)
    cat = lambda kind, i: np.concatenate([sources[s][kind][i] for s in names], axis=-1)
    strata = np.concatenate([[s] * len(sources[s]["fall"][2]) for s in names])
    res = {}
    for d in DEPTHS:
        # registered path (as c0_battery.summarize / c0_verdict): H = -mean of the float32 LR, the same pair draws
        reg = {kind: np.asarray(cat(kind, 3)[d], float) for kind in ("fall", "nonfall")}
        rf, rn = pair_bootstrap(reg["fall"], reg["nonfall"], strata, cat("fall", 2), cat("nonfall", 2), n_boot, seed)
        h_reg = {kind: float(-reg[kind].mean()) for kind in reg}
        rb_reg = (-rf) / np.where(rn == 0, np.nan, -rn)
        registered_path = dict(H_fall=h_reg["fall"], H_nonfall=h_reg["nonfall"], H_fall_ci=sorted(-x for x in ci(rf)),
                               H_nonfall_ci=sorted(-x for x in ci(rn)),
                               ratio=h_reg["fall"] / h_reg["nonfall"] if h_reg["nonfall"] != 0 else float("nan"),
                               ratio_ci=ci(rb_reg[np.isfinite(rb_reg)]))
        # decomposition: float64 logs of the same spreads, the same draws
        vals = {kind: np.stack([cat(kind, 0)[d], cat(kind, 1)[d]], axis=1) for kind in ("fall", "nonfall")}  # [W, 2]
        bf, bn = pair_bootstrap(vals["fall"], vals["nonfall"], strata, cat("fall", 2), cat("nonfall", 2), n_boot, seed)
        pt = {kind: vals[kind].mean(0) for kind in vals}
        row = {}
        for kind, b in (("fall", bf), ("nonfall", bn)):
            c_ctx, c_imag = pt[kind]
            h = c_ctx - c_imag
            row[kind] = dict(windows=int(len(vals[kind])), H=float(h), H_ci=ci(b[:, 0] - b[:, 1]),
                             C_ctx=float(c_ctx), C_imag=float(c_imag),
                             geomean_u_ctx=float(np.exp(c_ctx)), geomean_u_imag=float(np.exp(c_imag)),
                             geomean_imag_over_ctx=float(np.exp(-h)))
        d_ctx = pt["fall"][0] - pt["nonfall"][0]
        d_imag = pt["fall"][1] - pt["nonfall"][1]
        bd_ctx, bd_imag = bf[:, 0] - bn[:, 0], bf[:, 1] - bn[:, 1]
        h_f, h_n = row["fall"]["H"], row["nonfall"]["H"]
        rb = (bf[:, 0] - bf[:, 1]) / (bn[:, 0] - bn[:, 1])
        row["difference"] = dict(
            dH=float(h_f - h_n), dH_ci=ci(bd_ctx - bd_imag),
            context_part=float(d_ctx), context_part_ci=ci(bd_ctx),
            imagination_part=float(-d_imag), imagination_part_ci=ci(-bd_imag),
            identity_error=float((h_f - h_n) - (d_ctx - d_imag)),
            note="dH = context_part + imagination_part; context_part = C_ctx,fall - C_ctx,nonfall; "
                 "imagination_part = -(C_imag,fall - C_imag,nonfall)")
        row["ratio_H"] = dict(ratio=float(h_f / h_n) if h_n != 0 else float("nan"),
                              ratio_ci=ci(rb[np.isfinite(rb)]))
        row["registered_path"] = registered_path
        row["float64_vs_registered_max_rel"] = float(max(abs(h_f - h_reg["fall"]) / abs(h_reg["fall"]),
                                                         abs(h_n - h_reg["nonfall"]) / abs(h_reg["nonfall"])))
        row["ms_before_terminal_target"] = (25 - d) * STEP_MS
        res[f"d{d}"] = row
    return res


def check_registered(res, report_path, rtol=1e-9):
    """At d = 25 the registered path (c0_battery.log_ratio on the saved float32 spreads, float64 mean, the same
    pair-bootstrap draws) must reproduce the registered C0 numbers; the largest relative difference is reported."""
    rep = json.loads(Path(report_path).read_text())["report"]["pooled"]
    rule = rep["c0_rule_d25_sum"]
    reg = dict(H_fall=rep["fall"]["rows"]["d25/sum"]["H"], H_nonfall=rep["nonfall"]["rows"]["d25/sum"]["H"],
               ratio=rule.get("ratio"), ratio_ci=rule.get("ratio_ci"), h_fall_ci=rule.get("h_fall_ci"))
    new = res["d25"]["registered_path"]
    pairs = [(new["H_fall"], reg["H_fall"]), (new["H_nonfall"], reg["H_nonfall"])]
    if reg["ratio"] is not None:                 # the registered branch with H_nonfall > 0 (the real data)
        pairs += [(new["ratio"], reg["ratio"])] + list(zip(new["ratio_ci"], reg["ratio_ci"]))
    else:                                        # H_nonfall <= 0: the rule reads H_fall's interval
        pairs += list(zip(new["H_fall_ci"], reg["h_fall_ci"]))
    rel = max(abs(a - b) / max(abs(b), 1e-300) for a, b in pairs)
    return dict(registered=reg, reproduced=bool(rel <= rtol), max_rel_diff=float(rel), rtol=rtol)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True, help="c0_windows.npz of the registered C0 run")
    p.add_argument("--report", type=Path, required=True, help="its c0_report.json (reproduction check)")
    p.add_argument("--out", type=Path, required=True, help="fresh folder")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    res = analyse(load_sources(a.windows), a.n_boot, a.seed)
    chk = check_registered(res, a.report)
    payload = dict(protocol="plan v6 amendment A3 §1: supplementary C0 analyses (no verdict)", results=res,
                   reproduction=chk, windows=str(a.windows), windows_sha256=sha256(a.windows),
                   report_sha256=sha256(a.report), n_boot=a.n_boot, seed=a.seed, analysis_sha256=sha256(__file__))
    (a.out / "c0_supplement.json").write_text(json.dumps(payload, indent=1))
    lines = ["C0 supplementary analyses (A3 §1; registered endpoint d25 unchanged; no verdict)",
             f"registered numbers reproduced at d25: {chk['reproduced']} (registered arithmetic path; max relative "
             f"difference {chk['max_rel_diff']:.2e}, tolerance {chk['rtol']:.0e})",
             "decomposition in float64; it differs from the registered float32 path by at most "
             f"{max(res[f'd{d}']['float64_vs_registered_max_rel'] for d in DEPTHS):.1e} (relative)"]
    for d in DEPTHS:
        r = res[f"d{d}"]
        f_, n_, df = r["fall"], r["nonfall"], r["difference"]
        lines += [f"d{d} ({r['ms_before_terminal_target']} ms before the terminal target):",
                  f"  geometric mean u_imag/u_ctx: fall {f_['geomean_imag_over_ctx']:.3f} ({f_['windows']} windows), "
                  f"non-fall {n_['geomean_imag_over_ctx']:.3f}",
                  f"  registered path: H fall {r['registered_path']['H_fall']:.4f} [{r['registered_path']['H_fall_ci'][0]:.3f}, "
                  f"{r['registered_path']['H_fall_ci'][1]:.3f}]; non-fall {r['registered_path']['H_nonfall']:.4f} "
                  f"[{r['registered_path']['H_nonfall_ci'][0]:.3f}, {r['registered_path']['H_nonfall_ci'][1]:.3f}]; "
                  f"ratio {r['registered_path']['ratio']:.3f} [{r['registered_path']['ratio_ci'][0]:.3f}, "
                  f"{r['registered_path']['ratio_ci'][1]:.3f}]",
                  f"  geometric means: u_ctx fall {f_['geomean_u_ctx']:.3f} vs non-fall {n_['geomean_u_ctx']:.3f}; "
                  f"u_imag fall {f_['geomean_u_imag']:.3f} vs non-fall {n_['geomean_u_imag']:.3f}",
                  f"  dH = {df['dH']:.3f} [{df['dH_ci'][0]:.3f}, {df['dH_ci'][1]:.3f}] = context part "
                  f"{df['context_part']:.3f} [{df['context_part_ci'][0]:.3f}, {df['context_part_ci'][1]:.3f}] + "
                  f"imagination part {df['imagination_part']:.3f} [{df['imagination_part_ci'][0]:.3f}, "
                  f"{df['imagination_part_ci'][1]:.3f}]"]
    text = "\n".join(lines) + "\n"
    (a.out / "c0_supplement.txt").write_text(text)
    print(text, flush=True)
    if not chk["reproduced"]:
        raise SystemExit("[c0_supplement] the registered d25 numbers were NOT reproduced: check the inputs")


if __name__ == "__main__":
    main()
