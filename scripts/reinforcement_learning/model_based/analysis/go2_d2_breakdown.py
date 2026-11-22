#!/usr/bin/env python3
"""A4 D2 option B breakdown (Charbel, 22 Nov 2026): WHICH falls does the D2 model miss, and do the five heads miss
the SAME ones? Read-only, CPU, descriptive; no rule changes. Model: the uncalibrated D2 fit (080f353d...).

Sets: the D1 probe (seed 911; already used by Q1-Q3) and the calibration set (E22 withheld D1 fitting-batch
episodes, seed 102). Teacher forced with memory from the episode start, as in Q2/Q3.
Per fall: heads that stop (p > 0.5) in the last 11 rows (0-5); first stop before the end; episode length (rows
since the reset); maximum tilt in the last 11 rows; ensemble disagreement u (variance over heads of the predicted
next state, mean over the 45 normalized dims) in the last 25 rows, ranked against all rows of surviving episodes.
Readings fixed before the run (descriptive flags, not gates):
  shared blind spot : >= 80 % of missed falls (no head stops) are missed by ALL five heads, i.e. heads agree;
  post-reset gap    : detection <= 0.1 on falls with episode length < 40 rows and >= 0.5 on the others;
  uncertainty warns : >= 50 % of missed falls have max u in the last 25 rows above the 99th percentile of u on
                      surviving episodes.
"Learn now, check later" (Charbel): predictions from this breakdown are written to the run-log before a fresh probe
(new seed) is analysed with the same code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import go2_g0_replay as rp  # noqa: E402
from go2_d2_qualify import probe_episodes, tilt_deg  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

LAST, UWIN, SHORT = 11, 25, 40


@torch.no_grad()
def episode_pass(sd, ep, device):
    """Logits [E, L] and disagreement u [L] (NaN at row 0), teacher forced with memory from the episode start."""
    E = len(sd.state_heads)
    LG, U = np.full((E, ep.L), np.nan, np.float32), np.full(ep.L, np.nan, np.float32)
    if ep.L < 2:
        return LG, U
    xs = torch.from_numpy(ep.S[:-1]).to(device)
    x = torch.cat([xs, torch.from_numpy(ep.A[1:]).to(device)], dim=-1).unsqueeze(0)
    base, _ = sd.state_base.memory.rnn(x)
    aux, _ = sd.auxiliary_base.memory.rnn(x)
    means = torch.stack([h(base[0], xs.unsqueeze(1))[0] for h in sd.state_heads])
    LG[:, 1:] = torch.stack([h(aux[0], xs.unsqueeze(1))[2] for h in sd.auxiliary_heads]).squeeze(-1).cpu().numpy()
    U[1:] = means.var(0, unbiased=False).mean(-1).cpu().numpy()
    return LG, U


def analyse(sd, eps, norm, device):
    falls, surv_u = [], []
    for ep in eps:
        LG, U = episode_pass(sd, ep, device)
        if ep.kind == "time_out":
            surv_u.append(U[1:])
            continue
        if ep.kind != "fall":
            continue
        lo = max(1, ep.L - LAST)
        stop = LG[:, lo:ep.L] > 0
        anywhere = LG[:, 1:] > 0
        first = [int(ep.L - 1 - (1 + int(np.argmax(r)))) if r.any() else None for r in anywhere]
        g = ep.S[lo:ep.L, 6:9] * norm["state_std"][6:9] + norm["state_mean"][6:9]
        falls.append(dict(env=ep.env, first_row=ep.first, length=int(ep.L), heads_stopping=int(stop.any(1).sum()),
                          per_head_caught=[bool(x) for x in stop.any(1)], first_stop_rows_before_end=first,
                          max_tilt_last11_deg=float(np.nanmax(tilt_deg(g))) if len(g) else float("nan"),
                          max_u_last25=float(np.nanmax(U[max(1, ep.L - UWIN):ep.L])) if ep.L > 1 else float("nan")))
    surv_u = np.concatenate(surv_u) if surv_u else np.zeros(0)
    p99 = float(np.nanpercentile(surv_u, 99)) if surv_u.size else float("nan")
    for f in falls:
        f["u_rank_vs_surviving"] = float((surv_u < f["max_u_last25"]).mean()) if surv_u.size else float("nan")
        f["u_above_p99"] = bool(f["max_u_last25"] > p99)
    return falls, p99, int(len(surv_u))


def summary(falls):
    hs = np.array([f["heads_stopping"] for f in falls]) if falls else np.zeros(0, int)
    missed = [f for f in falls if f["heads_stopping"] == 0]
    partial = [f for f in falls if 0 < f["heads_stopping"] < 5]
    short = [f for f in falls if f["length"] < SHORT]
    long_ = [f for f in falls if f["length"] >= SHORT]
    det = lambda fs: float(np.mean([np.mean(f["per_head_caught"]) for f in fs])) if fs else float("nan")
    tilt = lambda fs: float(np.median([f["max_tilt_last11_deg"] for f in fs])) if fs else float("nan")
    s = dict(falls=len(falls), heads_stopping_counts={str(k): int((hs == k).sum()) for k in range(6)},
             detection=det(falls), missed_by_all=len(missed), partial=len(partial),
             share_of_misses_missed_by_all=(len(missed) / (len(missed) + len(partial))) if (missed or partial) else float("nan"),
             short_falls=len(short), detection_short=det(short), long_falls=len(long_), detection_long=det(long_),
             median_tilt_missed=tilt(missed), median_tilt_caught=tilt([f for f in falls if f["heads_stopping"] == 5]),
             missed_u_above_p99=float(np.mean([f["u_above_p99"] for f in missed])) if missed else float("nan"),
             caught_u_above_p99=float(np.mean([f["u_above_p99"] for f in falls if f["heads_stopping"] > 0]))
             if any(f["heads_stopping"] > 0 for f in falls) else float("nan"))
    s["flags"] = dict(shared_blind_spot=bool(s["share_of_misses_missed_by_all"] >= 0.8),
                      post_reset_gap=bool(s["detection_short"] <= 0.1 and s["detection_long"] >= 0.5),
                      uncertainty_warns=bool(s["missed_u_above_p99"] >= 0.5))
    return s


def run(wm, norm_path, probe, shift, shift_withheld, out, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    sd, H, norm, _ = rp.load_model(wm, norm_path, device)
    rep = dict(protocol="A4 D2 option B breakdown (descriptive)", wm_sha256=sha256(wm), analysis_sha256=sha256(__file__))
    for name, path in (("probe", probe), ("calibration", shift)):
        with np.load(path, allow_pickle=False) as z:
            validate_trace(z)
            tr = {k: z[k] for k in z.files if k != "metadata"}
        if name == "probe":
            eps = probe_episodes(tr, norm)
        else:
            eps = [rp.Episode(tr, h, norm) for h in rp.heldout(tr, shift_withheld, 0.1, 101)]
        falls, p99, n_rows = analyse(sd, eps, norm, device)
        rep[name] = dict(trace_sha256=sha256(path), surviving_rows=n_rows, surviving_u_p99=p99, summary=summary(falls), falls=falls)
    (out / "d2_breakdown.json").write_text(json.dumps(rep, indent=1, default=float))
    L = ["A4 D2 breakdown (descriptive; uncalibrated D2 model)"]
    for name in ("probe", "calibration"):
        s = rep[name]["summary"]
        L.append(f"{name}: {s['falls']} falls; heads stopping (0..5): {s['heads_stopping_counts']}; detection {s['detection']:.3f}")
        L.append(f"  missed by all 5: {s['missed_by_all']}, partly missed: {s['partial']} -> share of misses missed by all "
                 f"{s['share_of_misses_missed_by_all']:.2f}")
        L.append(f"  episode < {SHORT} rows: {s['short_falls']} falls, detection {s['detection_short']:.3f}; >= {SHORT}: "
                 f"{s['long_falls']} falls, detection {s['detection_long']:.3f}")
        L.append(f"  median max tilt (last 11 rows): missed {s['median_tilt_missed']:.1f} deg, caught by all {s['median_tilt_caught']:.1f} deg")
        L.append(f"  disagreement above the surviving p99 in the last {UWIN} rows: missed {s['missed_u_above_p99']:.2f}, "
                 f"caught {s['caught_u_above_p99']:.2f}")
        L.append(f"  flags: {s['flags']}")
    text = "\n".join(L) + "\n"
    (out / "d2_breakdown.txt").write_text(text)
    print(text, flush=True)
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--shift", type=Path, required=True)
    p.add_argument("--shift-withheld", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.wm, a.norm, a.probe, a.shift, a.shift_withheld, a.out)


if __name__ == "__main__":
    main()
