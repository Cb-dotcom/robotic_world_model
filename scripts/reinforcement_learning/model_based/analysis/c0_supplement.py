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

from c0_battery import ZeroDisagreement, ci, pair_bootstrap  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

DEPTHS = (25, 16)
STEP_MS = 20


def load_sources(path):
    """{source: {kind: (log_ctx [H, W], log_imag [H, W], clusters [W])}} in the saved order."""
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
                ur = z[f"{name}/{kind}/u_real"].sum(-1)                       # [H, W]
                ui = z[f"{name}/{kind}/u_imag"].sum(-1)                       # [heads, H, W]
                if (ur <= 0).any() or (ui <= 0).any():
                    raise ZeroDisagreement(f"{name}/{kind}: zero summed spread")
                out[name][kind] = (np.log(ur), np.log(ui).mean(0), z[f"{name}/{kind}/clusters"].astype(str))
    return out


def analyse(sources, n_boot=10_000, seed=0):
    names = list(sources)
    cat = lambda kind, i: np.concatenate([sources[s][kind][i] for s in names], axis=-1)
    strata = np.concatenate([[s] * len(sources[s]["fall"][2]) for s in names])
    res = {}
    for d in DEPTHS:
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
        row["ms_before_terminal_target"] = (25 - d) * STEP_MS
        res[f"d{d}"] = row
    return res


def check_registered(res, report_path):
    """At d = 25 the H values and the ratio must reproduce the registered C0 numbers (to float32 precision: the
    saved spreads are float32, and the summation order differs; relative tolerance 1e-6)."""
    rep = json.loads(Path(report_path).read_text())["report"]["pooled"]
    rule = rep["c0_rule_d25_sum"]
    reg = dict(H_fall=rep["fall"]["rows"]["d25/sum"]["H"], H_nonfall=rep["nonfall"]["rows"]["d25/sum"]["H"],
               ratio=rule.get("ratio"), ratio_ci=rule.get("ratio_ci"), h_fall_ci=rule.get("h_fall_ci"))
    new = res["d25"]
    ok = (np.isclose(new["fall"]["H"], reg["H_fall"], rtol=1e-6, atol=1e-9)
          and np.isclose(new["nonfall"]["H"], reg["H_nonfall"], rtol=1e-6, atol=1e-9))
    if reg["ratio"] is not None:                 # the registered branch with H_nonfall > 0 (the real data)
        ok = ok and np.isclose(new["ratio_H"]["ratio"], reg["ratio"], rtol=1e-9) \
            and np.allclose(new["ratio_H"]["ratio_ci"], reg["ratio_ci"], rtol=1e-6)
    else:                                        # H_nonfall <= 0: the rule reads H_fall's interval
        ok = ok and np.allclose(new["fall"]["H_ci"], reg["h_fall_ci"], rtol=1e-6)
    return dict(registered=reg, reproduced=bool(ok))


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
             f"registered numbers reproduced at d25: {chk['reproduced']}"]
    for d in DEPTHS:
        r = res[f"d{d}"]
        f_, n_, df = r["fall"], r["nonfall"], r["difference"]
        lines += [f"d{d} ({r['ms_before_terminal_target']} ms before the terminal target):",
                  f"  geometric mean u_imag/u_ctx: fall {f_['geomean_imag_over_ctx']:.3f} ({f_['windows']} windows), "
                  f"non-fall {n_['geomean_imag_over_ctx']:.3f}",
                  f"  H fall {f_['H']:.3f} [{f_['H_ci'][0]:.3f}, {f_['H_ci'][1]:.3f}]; non-fall {n_['H']:.3f} "
                  f"[{n_['H_ci'][0]:.3f}, {n_['H_ci'][1]:.3f}]; ratio {r['ratio_H']['ratio']:.2f} "
                  f"[{r['ratio_H']['ratio_ci'][0]:.2f}, {r['ratio_H']['ratio_ci'][1]:.2f}]",
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
