#!/usr/bin/env python3
"""Plan v6 §3 fix screen (4 Dec): F1 = per-member trajectory spread, on the fall paths of batches A and C.

Input: c0_battery runs (c0_windows.npz + c0_report.json) with head feedback and recorded actions, one per
model. Each imagined path follows one head with its own trunk and auxiliary hidden state (`model.reset()` per
head), which is F1's member trajectory. No new imagination is run; only the stored arrays are read.

Operational definitions (fixed 18 Nov 2026 before this script read any data; plan §3 names the criteria,
these lines make them computable):
* member spread m(d) = sum over the 45 normalized dims of the std over the K = 5 member paths of their own
  state at depth d (= c0 `u_member`, C1a's "per-member spread").
* standard penalty u(d) = sum over dims of the head disagreement along the followed path (`u_imag`), mean
  over the 5 followed paths (the RWM-U penalty, lambda = 1).
* real-context disagreement r(d) = `u_real` summed over dims (the reference of the healing index).
* scale s(d) = mean over matched non-fall windows of the realized normalized error (sum over dims of
  |followed state - true state|, mean over paths) / mean over the same windows of m(d), d = 0..25.
  F1 penalty = lambda * s(d) * m(d), lambda = 1. (In the rounds s(d) is refit from test-first errors.)
* Criterion 1, no healing: HR_F1(25) = exp(mean over fall windows of log(m(25) / r(25))) >= 1 (point
  estimate; the 95 % episode-cluster CI is reported).
* Criterion 2, ranking: Spearman over fall windows of m(25) vs the imagined base-velocity RMSE across
  paths at d = 25 (C1a's error) >= 0.5 (tie-safe; bootstrap CI reported).
* Criterion 3, size: mean over non-fall windows and d = 0..25 of s(d) m(d), divided by the same mean of u(d),
  <= 2.
* Criterion 4, cost <= 2x (else K = 3): decided at the first timed F1 training run (plan §8); a CPU timing of
  one standard step vs K member steps of the same model is reported here only as an indication.
* Ranking (descriptive here: only F1 entered the screen, every other candidate's rule did not fire): fall-
  warning AUROC (falls vs matched non-falls) at leads of 10-25 steps, i.e. depth d = 25 - lead = 0..15 of
  the imagination started 25 steps before the end of the window, for s(d) m(d), the standard u(d), the real
  body tilt at lead 25 (the first predicted row; the earliest real tilt after the imagination starts) and the
  real tilt at the same row as the imagined state (an oracle baseline: it is known only lead steps before).
Selection uses batches A and C only (plan §3). The frozen winner is scored once on the 1701 held-out falls
after 4 Dec, not here. The model of the planned F arm is D's (E1-V refit, "F = D + fix", plan §3); every
model given is screened and reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c0_battery import ci, cluster_bootstrap, json_safe, spearman  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

D = 25
LEADS = range(10, 26)
SOURCES = ("batchA", "batchC")
RULES = dict(hr25_min=1.0, spearman_min=0.5, penalty_ratio_max=2.0, cost_ratio_max=2.0)


def auroc(pos, neg):
    """Mann-Whitney AUROC (ties count one half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_run(run_dir, sources=SOURCES):
    z = np.load(Path(run_dir) / "c0_windows.npz")
    rep = json.loads((Path(run_dir) / "c0_report.json").read_text())
    meta = rep["meta"]
    if meta.get("feedback", "head") != "head" or meta.get("actions", "recorded") != "recorded":
        raise ValueError(f"{run_dir}: F1 screen needs head feedback and recorded actions")
    if meta.get("imag_context", 32) != 32:
        raise ValueError(f"{run_dir}: F1 screen needs the 32-row context")
    out = {}
    for kind in ("fall", "nonfall"):
        parts = {k: [] for k in ("m", "u", "r", "e_norm", "e_vel", "tilt_real", "clusters", "strata")}
        for s in sources:
            key = f"{s}/{kind}"
            if f"{key}/u_member" not in z.files:
                raise KeyError(f"{run_dir}: no {key} windows")
            own, target = z[f"{key}/own"], z[f"{key}/target"]                     # [P, H, W, 45], [H, W, 45]
            parts["m"].append(z[f"{key}/u_member"].sum(-1))                       # [H, W]
            parts["u"].append(z[f"{key}/u_imag"].sum(-1).mean(0))                 # [H, W]
            parts["r"].append(z[f"{key}/u_real"].sum(-1))                         # [H, W]
            parts["e_norm"].append(np.abs(own - target[None]).sum(-1).mean(0))    # [H, W]
            parts["e_vel"].append(np.sqrt(z[f"{key}/se_imag_vel"].mean(0)))      # [H, W]
            parts["tilt_real"].append(z[f"{key}/tilt_real"])                      # [H, W]
            parts["clusters"].append(z[f"{key}/clusters"])
            parts["strata"].append(np.full(z[f"{key}/clusters"].shape, s))
        out[kind] = {k: np.concatenate(v, axis=-1 if k not in ("clusters", "strata") else 0) for k, v in parts.items()}
    return out, meta


def start_tilt(run_dir, sources=SOURCES):
    """Real tilt at lead 25: tilt_real[0], the true state of the first predicted row."""
    z = np.load(Path(run_dir) / "c0_windows.npz")
    return {kind: np.concatenate([z[f"{s}/{kind}/tilt_real"][0] for s in sources]) for kind in ("fall", "nonfall")}


def scale(nonfall):
    return nonfall["e_norm"].mean(1) / nonfall["m"].mean(1)                      # [H]


def screen(run_dir, n_boot=10_000, seed=0, sources=SOURCES):
    w, meta = load_run(run_dir, sources)
    f, n = w["fall"], w["nonfall"]
    s = scale(n)
    # 1. healing of the F1 signal
    lr = np.log(f["m"][D] / f["r"][D])
    point, boot = cluster_bootstrap(lr, f["clusters"], f["strata"], n_boot, seed)
    hr = dict(HR25=float(np.exp(point)), HR25_ci=[float(np.exp(x)) for x in ci(boot)],
              standard_HR25=float(np.exp(np.log(f["u"][D] / f["r"][D]).mean())))
    # 2. ranking
    rho = spearman(f["m"][D], f["e_vel"][D])
    rng = np.random.default_rng(seed)
    rb = []
    for _ in range(2000):
        pick = []
        for st in np.unique(f["strata"]):
            idx = np.flatnonzero(f["strata"] == st)
            keys = np.unique(f["clusters"][idx])
            draw = rng.choice(keys, len(keys), replace=True)
            pick.extend(np.concatenate([idx[f["clusters"][idx] == k] for k in draw]))
        rb.append(spearman(f["m"][D][pick], f["e_vel"][D][pick]))
    rank = dict(spearman_d25=float(rho), spearman_d25_ci=[float(x) for x in np.nanpercentile(rb, [2.5, 97.5])],
                standard_spearman_d25=float(spearman(f["u"][D], f["e_vel"][D])),
                spearman_by_depth={f"d{d}": float(spearman(f["m"][d], f["e_vel"][d])) for d in (1, 8, 16, 25)})
    # 3. penalty size on nominal (matched non-fall) windows
    f1_pen, std_pen = (s[:, None] * n["m"]).mean(), n["u"].mean()
    size = dict(f1_mean_penalty=float(f1_pen), standard_mean_penalty=float(std_pen),
                ratio=float(f1_pen / std_pen), s_by_depth={f"d{d}": float(s[d]) for d in (0, 1, 8, 16, 25)})
    # ranking signal (descriptive)
    t0 = start_tilt(run_dir, sources)
    au = {}
    for lead in LEADS:
        d = D - lead
        au[f"lead{lead}"] = dict(F1=auroc(s[d] * f["m"][d], s[d] * n["m"][d]), standard=auroc(f["u"][d], n["u"][d]),
                                 tilt_at_start=auroc(t0["fall"], t0["nonfall"]),
                                 tilt_same_row=auroc(f["tilt_real"][d], n["tilt_real"][d]))
    mean_au = {k: float(np.mean([v[k] for v in au.values()])) for k in ("F1", "standard", "tilt_at_start", "tilt_same_row")}
    passed = dict(no_healing=hr["HR25"] >= RULES["hr25_min"], ranking=rho >= RULES["spearman_min"],
                  size=size["ratio"] <= RULES["penalty_ratio_max"])
    return dict(run=str(run_dir), wm=meta.get("wm"), wm_sha256=meta.get("wm_sha256"), sources=list(sources),
                windows=dict(fall=int(len(f["clusters"])), nonfall=int(len(n["clusters"]))),
                healing=hr, ranking=rank, size=size, auroc_by_lead=au, auroc_mean_lead10_25=mean_au,
                criteria=passed, passes_screen_1_to_3=bool(all(passed.values())))


@torch.no_grad()
def cost_indication(wm, batch=4096, k=5, reps=3):
    """CPU seconds of one imagination step: standard (one trunk pass, every head reads it) vs F1 (K member
    trunks: the same forward on K x batch). Indicative only (plan §8: decided at the first timed GPU run)."""
    from c0_battery import load_models
    model, _, _ = load_models(wm)
    xs, xa = torch.randn(batch, 32, 45), torch.randn(batch, 32, 12)

    def step(b):
        model.reset()
        model(xs[:b].repeat(1, 1, 1), xa[:b])                                 # warm the memory
        t = time.perf_counter()
        model(xs[:b, -1:], xa[:b, -1:])
        return time.perf_counter() - t

    def forward_k():
        model.reset()
        x2, a2 = xs.repeat(k, 1, 1), xa.repeat(k, 1, 1)
        model(x2, a2)
        t = time.perf_counter()
        model(x2[:, -1:], a2[:, -1:])
        return time.perf_counter() - t

    std = min(step(batch) for _ in range(reps))
    f1 = min(forward_k() for _ in range(reps))
    return dict(batch=batch, members=k, standard_s=std, f1_s=f1, ratio=f1 / std, device="cpu",
                note="indication only; the cost rule is decided at the first timed F1 training run (plan §8)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=C0_RUN_DIR")
    p.add_argument("--primary", required=True, help="NAME of the F arm's model (plan §3: D's model, E1-V refit)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cost", action="store_true", help="skip the CPU timing indication")
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists")
    runs = [item.partition("=")[::2] for item in a.run]
    if a.primary not in dict(runs):
        p.error("--primary must name one --run")
    a.out.mkdir(parents=True)
    torch.set_num_threads(8)
    res = {name: screen(path, a.bootstrap, a.seed) for name, path in runs}
    if not a.no_cost:
        for name, path in runs:
            wm = res[name]["wm"]
            res[name]["cost_indication"] = cost_indication(",".join(wm) if isinstance(wm, list) else wm)
    report = dict(rules=RULES, primary=a.primary, results=res, analysis_sha256=sha256(__file__),
                  verdict=("F1 PASSES criteria 1-3" if res[a.primary]["passes_screen_1_to_3"] else "F1 FAILS the screen")
                  + f" on the primary model ({a.primary}); criterion 4 (cost) at the first timed run")
    (a.out / "f1_screen.json").write_text(json.dumps(json_safe(report), indent=1, allow_nan=False))
    lines = ["Plan v6 §3 fix screen: F1 per-member spread, batches A + C fall paths (operational definitions in the script)",
             f"rules: HR_F1(25) >= {RULES['hr25_min']}, Spearman(m(25), vel error) >= {RULES['spearman_min']}, "
             f"nominal penalty ratio <= {RULES['penalty_ratio_max']}, cost <= {RULES['cost_ratio_max']}x (timed run)"]
    for name, r in res.items():
        h, k, z, c = r["healing"], r["ranking"], r["size"], r["criteria"]
        lines += [f"== {name}{' (PRIMARY: F arm model)' if name == a.primary else ''}: {r['windows']}",
                  f"  1 no healing: HR_F1(25) {h['HR25']:.3f} [{h['HR25_ci'][0]:.3f}, {h['HR25_ci'][1]:.3f}] "
                  f"(standard penalty {h['standard_HR25']:.3f}) -> {'pass' if c['no_healing'] else 'FAIL'}",
                  f"  2 ranking: Spearman d25 {k['spearman_d25']:+.2f} [{k['spearman_d25_ci'][0]:+.2f}, "
                  f"{k['spearman_d25_ci'][1]:+.2f}] (standard {k['standard_spearman_d25']:+.2f}); by depth "
                  + ", ".join(f"{d} {v:+.2f}" for d, v in k["spearman_by_depth"].items())
                  + f" -> {'pass' if c['ranking'] else 'FAIL'}",
                  f"  3 size: F1 {z['f1_mean_penalty']:.2f} vs standard {z['standard_mean_penalty']:.2f} -> ratio "
                  f"{z['ratio']:.2f} -> {'pass' if c['size'] else 'FAIL'}; s(d) "
                  + ", ".join(f"{d} {v:.3f}" for d, v in z["s_by_depth"].items()),
                  "  AUROC, mean over leads 10-25: " + ", ".join(f"{k2} {v:.3f}" for k2, v in r["auroc_mean_lead10_25"].items())]
        if "cost_indication" in r:
            ci_ = r["cost_indication"]
            lines.append(f"  4 cost (CPU indication): standard {ci_['standard_s'] * 1e3:.1f} ms, F1 "
                         f"{ci_['f1_s'] * 1e3:.1f} ms per step at {ci_['batch']} envs -> {ci_['ratio']:.2f}x")
    lines.append(report["verdict"])
    text = "\n".join(lines) + "\n"
    (a.out / "f1_screen.txt").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
