#!/usr/bin/env python3
"""Paper figures for C0 (IROS draft, Sections III-IV), from the registered run's saved windows. No new statistic.

Input: c0_windows.npz and c0_report.json of the registered C0 run (rerun 2, 18 Nov: ETH's checkpoint, sources
A, C and 1701 round 0; logs/p2/v6w1_rerun2_1118/c0_eth/). Read with c0_supplement.load_sources, the reader that
reproduced the registered numbers exactly on 21 Nov.

Gate: c0_supplement.analyse + check_registered must reproduce the registered d25 numbers (the C0 rule's H_fall,
H_nonfall, ratio and interval) before any figure is drawn; the curves below must equal the supplement's d25 and d16
values. Otherwise the script stops and draws nothing.

For every depth d = 0..25 (d = imagined states fed back; d = 25 is the simulator's terminal transition), pooled over
sources, every window weight 1, as registered:
  geometric-mean disagreement with recorded context, exp(mean log u_ctx), and on the imagined path,
  exp(mean over windows and driving heads of log u_imag), for falls and matched walking windows;
  H(d) = mean log u_ctx - mean log u_imag (= the registered -mean LR, up to float32 rounding);
95 % bands from the same episode-cluster pair bootstrap as the registered rule (pair components, stratified by
source, seed 0). The bands at d other than 25 are descriptive.

Outputs: fig_c0_disagreement.{pdf,png} (Fig. 1), fig_c0_healing.{pdf,png} (Fig. 2), c0_curves.csv (the table view
of both), c0_figures.json (the gate and the numbers drawn). IEEE single column (3.5 in), 8 pt.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c0_battery import ci, pair_bootstrap  # noqa: E402
import c0_supplement as sup  # noqa: E402

FALL, WALK = "#eb6834", "#2a78d6"          # dataviz reference palette, slots 2 and 1 (validated: CVD dE 24.7)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEPTH_MAX, STEP_MS = 25, 20


class GateFailed(RuntimeError):
    pass


def curves(sources, n_boot=10_000, seed=0):
    names = list(sources)
    cat = lambda kind, i: np.concatenate([sources[s][kind][i] for s in names], axis=-1)
    strata = np.concatenate([[s] * len(sources[s]["fall"][2]) for s in names])
    cf, cn = cat("fall", 2), cat("nonfall", 2)
    rows = []
    for d in range(DEPTH_MAX + 1):
        vals = {kind: np.stack([cat(kind, 0)[d], cat(kind, 1)[d]], axis=1) for kind in ("fall", "nonfall")}
        bf, bn = pair_bootstrap(vals["fall"], vals["nonfall"], strata, cf, cn, n_boot, seed)
        row = dict(d=d, ms_before_terminal=(DEPTH_MAX - d) * STEP_MS)
        for kind, b in (("fall", bf), ("nonfall", bn)):
            c_ctx, c_imag = vals[kind].mean(0)
            row[kind] = dict(H=float(c_ctx - c_imag), H_ci=ci(b[:, 0] - b[:, 1]),
                             u_ctx=float(np.exp(c_ctx)), u_ctx_ci=[float(np.exp(x)) for x in ci(b[:, 0])],
                             u_imag=float(np.exp(c_imag)), u_imag_ci=[float(np.exp(x)) for x in ci(b[:, 1])],
                             windows=int(len(vals[kind])))
        rows.append(row)
    return rows


def gate(sources, report, rows, n_boot):
    sup_res = sup.analyse(sources, n_boot=n_boot, seed=0)
    chk = sup.check_registered(sup_res, report)
    if not chk["reproduced"]:
        raise GateFailed(f"the registered C0 numbers are not reproduced: {chk}")
    for key in ("d25", "d16"):
        d = int(key[1:])
        for kind in ("fall", "nonfall"):
            a, b = rows[d][kind], sup_res[key][kind]
            for x, y in ((a["H"], b["H"]), (a["u_ctx"], b["geomean_u_ctx"]), (a["u_imag"], b["geomean_u_imag"])):
                if abs(x - y) > 1e-12 * max(1.0, abs(y)):
                    raise GateFailed(f"{key} {kind}: curve {x} != supplement {y}")
            if [round(v, 12) for v in a["H_ci"]] != [round(v, 12) for v in b["H_ci"]]:
                raise GateFailed(f"{key} {kind}: band {a['H_ci']} != supplement {b['H_ci']}")
    return dict(check=chk, ratio_d25=sup_res["d25"]["registered_path"]["ratio"],
                ratio_d25_ci=sup_res["d25"]["registered_path"]["ratio_ci"])


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d", "ms_before_terminal", "kind", "u_ctx", "u_ctx_lo", "u_ctx_hi", "u_imag", "u_imag_lo",
                    "u_imag_hi", "H", "H_lo", "H_hi", "windows"])
        for r in rows:
            for kind in ("fall", "nonfall"):
                k = r[kind]
                w.writerow([r["d"], r["ms_before_terminal"], kind, k["u_ctx"], *k["u_ctx_ci"], k["u_imag"],
                            *k["u_imag_ci"], k["H"], *k["H_ci"], k["windows"]])


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(0, DEPTH_MAX + 0.6)
    ax.set_xticks([0, 5, 10, 15, 20, 25])


def draw(rows, g, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.labelcolor": INK, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    d = np.array([r["d"] for r in rows])
    get = lambda kind, key: np.array([r[kind][key] for r in rows])
    band = lambda kind, key: np.array([r[kind][key] for r in rows]).T

    # Fig. 1: disagreement with recorded context vs on the imagined path
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    _style(ax)
    series = [("fall", "u_ctx", FALL, "-", "falls, recorded context"), ("fall", "u_imag", FALL, "--", "falls, imagined"),
              ("nonfall", "u_ctx", WALK, "-", "walking, recorded context"),
              ("nonfall", "u_imag", WALK, "--", "walking, imagined")]
    for kind, key, color, ls, label in series:
        lo, hi = band(kind, key + "_ci")
        ax.fill_between(d, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(d, get(kind, key), color=color, linestyle=ls, linewidth=1.4, label=label)
    ax.set_yscale("log")
    from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("imagined steps fed back, d")
    ax.set_ylabel("disagreement u (geometric mean)")
    ax.set_title("d = 25 is the terminal transition (the fall)", fontsize=6.5, color=MUTED, loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="best", handlelength=2.2)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_c0_disagreement.{ext}", dpi=300)
    plt.close(fig)

    # Fig. 2: healing deficit H(d), falls vs matched walking windows
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    _style(ax)
    for kind, color, label in (("fall", FALL, "falls"), ("nonfall", WALK, "matched walking")):
        lo, hi = band(kind, "H_ci")
        ax.fill_between(d, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(d, get(kind, "H"), color=color, linewidth=1.4, label=label)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.axvline(DEPTH_MAX, color=MUTED, linewidth=0.6, linestyle=":")
    lo, hi = g["ratio_d25_ci"]
    ax.set_title(f"d = 25 (dotted, registered): H ratio {g['ratio_d25']:.2f} [{lo:.2f}, {hi:.2f}]",
                 fontsize=6.5, color=MUTED, loc="left")
    ax.set_xlabel("imagined steps fed back, d")
    ax.set_ylabel("healing deficit H(d)")
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_c0_healing.{ext}", dpi=300)
    plt.close(fig)


def run(windows, report, out, n_boot=10_000):
    sources = sup.load_sources(windows)
    rows = curves(sources, n_boot=n_boot)
    g = gate(sources, report, rows, n_boot)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(rows, out / "c0_curves.csv")
    draw(rows, g, out)
    res = dict(inputs=dict(windows=str(windows), windows_sha256=sup.sha256(windows), report=str(report),
                           report_sha256=sup.sha256(report)), gate=g, n_boot=n_boot, curves=rows,
               analysis_sha256=sup.sha256(__file__))
    (out / "c0_figures.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args(argv)
    try:
        res = run(a.windows, a.report, a.out, a.n_boot)
    except GateFailed as e:
        print(f"GATE: FAIL - {e}")
        return 3
    r25 = res["curves"][DEPTH_MAX]
    print(f"GATE: PASS (registered C0 reproduced, max rel diff {res['gate']['check']['max_rel_diff']:.1e})")
    print(f"d25 falls: u_ctx {r25['fall']['u_ctx']:.2f}, u_imag {r25['fall']['u_imag']:.2f}, H {r25['fall']['H']:.3f}; "
          f"walking: u_ctx {r25['nonfall']['u_ctx']:.2f}, u_imag {r25['nonfall']['u_imag']:.2f}, "
          f"H {r25['nonfall']['H']:.3f}")
    print(f"wrote {a.out}/fig_c0_disagreement.pdf, fig_c0_healing.pdf, c0_curves.csv, c0_figures.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
