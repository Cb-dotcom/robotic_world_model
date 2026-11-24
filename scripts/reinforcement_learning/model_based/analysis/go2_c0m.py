#!/usr/bin/env python3
"""C0-M on Go2 (descriptive; draft rules 24 Nov 2026 §1): the go2_c0_check windows with the memory schedule held fixed.

Same model, probes, windows and matching as go2_c0_check.py (K = 10 primary, K = 0; depth 25 imagined steps before
the target row T). go2_c0_check.py's two readings differ in the memory schedule as in ANYmal C0: u_ctx(T) restarts
memory with the 32 recorded rows before T; u_imag(h, T) carries memory from T - 25 through 25 imagined states. The
new reading u_car(T) carries memory from T - 25 exactly as u_imag, but feeds the RECORDED state of every row
(the rows go2_c0_check's recorded arm reads). Per window: log u_ctx - log u_car = memory part, log u_car - mean over
heads log u_imag = state-source part; means over windows; the registered pair bootstrap (episode components,
stratified by probe, 10,000 draws, seed 0).

No verdict: Go2 is descriptive here. The windows exclude falls closer than 57 + K rows to a reset, so this check
cannot address the startup falls (D3, D7).
Gates (the run stops, exit 3): u_car = u_imag = recorded u at the window start (depth 0; relative <= 1e-5) and
u_car(T - 25 + m) at m = 0 equals go2_c0_check.recorded_u at T - 25.
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

import go2_c0_check as g  # noqa: E402
import go2_g0_replay as rp  # noqa: E402
from c0_battery import ci, pair_bootstrap  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

PARITY_RTOL = 1e-5


@torch.no_grad()
def carried_u(sd, eps, rows, device, depth=g.DEPTH):
    """[n, depth + 1]: u on the recorded path with memory carried from T - depth (one path; all heads read it)."""
    starts = [(i, t - depth) for i, t in rows]
    xs = torch.from_numpy(np.stack([eps[i].S[j - g.H_CTX:j] for i, j in starts])).to(device)
    xa = torch.from_numpy(np.stack([eps[i].A[j - g.H_CTX + 1:j + 1] for i, j in starts])).to(device)
    U = np.empty((len(rows), depth + 1), np.float64)
    sd.reset()
    for m in range(depth + 1):
        means, _ = rp.all_heads(sd, xs, xa)
        U[:, m] = g.ppo_spread(means).cpu().numpy()
        if m < depth:
            xs = torch.from_numpy(np.stack([eps[i].S[j + m:j + m + 1] for i, j in starts])).to(device)
            xa = torch.from_numpy(np.stack([eps[i].A[j + m + 1:j + m + 2] for i, j in starts])).to(device)
    sd.reset()
    return U


def analyse(sd, eps, k, device, n_boot=10_000, seed=0):
    fw, sw, counts = g.windows(eps, k, seed=seed)
    res, vals, gates = dict(K=k, counts=counts), {}, {}
    for name, rows in (("fall", fw), ("surviving", sw)):
        uc = g.recorded_u(sd, eps, rows, device)
        ui = g.imagined_u(sd, eps, rows, device)
        ucar = carried_u(sd, eps, rows, device)
        d0 = g.recorded_u(sd, eps, [(i, T - g.DEPTH) for i, T in rows], device)
        gates[name] = float(max(np.max(np.abs(ucar[:, 0] - d0) / d0), np.max(np.abs(ui[:, :, 0] - d0[:, None]) / d0[:, None])))
        a, b, c = np.log(uc), np.log(ucar[:, -1]), np.log(ui[:, :, -1]).mean(1)
        vals[name] = (np.stack([a, b, c], 1), [eps[i].cluster for i, _ in rows], [eps[i].probe for i, _ in rows])
    res["depth0_parity_max_rel"] = gates
    if max(gates.values()) > PARITY_RTOL:
        raise RuntimeError(f"depth-0 parity failed: {gates}")
    bf, bs = pair_bootstrap(vals["fall"][0], vals["surviving"][0], vals["fall"][2], vals["fall"][1],
                            vals["surviving"][1], n_boot, seed)
    for name, b in (("fall", bf), ("surviving", bs)):
        p = vals[name][0].mean(0)
        res[name] = dict(n=int(len(vals[name][0])), H=float(p[0] - p[2]), H_ci=ci(b[:, 0] - b[:, 2]),
                         H_mem=float(p[0] - p[1]), H_mem_ci=ci(b[:, 0] - b[:, 1]),
                         H_src=float(p[1] - p[2]), H_src_ci=ci(b[:, 1] - b[:, 2]),
                         geomean=dict(u_ctx=float(np.exp(p[0])), u_car=float(np.exp(p[1])), u_imag=float(np.exp(p[2]))))
    res["dH_src"] = dict(value=res["fall"]["H_src"] - res["surviving"]["H_src"],
                         ci=ci((bf[:, 1] - bf[:, 2]) - (bs[:, 1] - bs[:, 2])))
    lo, hi = res["surviving"]["H_src_ci"]
    res["unstable_denominator"] = bool(lo <= 0 <= hi)
    return res


def run(wm, norm_path, probes, out, n_boot=10_000, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sd, _, norm, _ = rp.load_model(wm, norm_path, device)
    eps = []
    for name, path in probes:
        with np.load(path, allow_pickle=False) as z:
            validate_trace(z)
            tr = {kk: z[kk] for kk in z.files if kk != "metadata"}
        eps += g.episodes_of(tr, norm, name)[0]
    try:
        results = {str(k): analyse(sd, eps, k, device, n_boot) for k in (10, 0)}
    except RuntimeError as e:
        (out / "go2_c0m_stop.txt").write_text(str(e) + "\n")
        raise SystemExit(3)
    rep = dict(protocol="C0-M on Go2 (descriptive): go2_c0_check windows, memory schedule held fixed",
               wm_sha256=sha256(wm), norm_sha256=sha256(norm_path), probes={n: sha256(p) for n, p in probes},
               results=results, analysis_sha256=sha256(__file__))
    (out / "go2_c0m.json").write_text(json.dumps(rep, indent=1, default=float))
    f3 = lambda c: "[" + ", ".join(f"{x:.3f}" for x in c) + "]"
    L = ["C0-M on Go2 (descriptive; no verdict; falls < 57 + K rows after a reset are excluded)"]
    for k in ("10", "0"):
        r = results[k]
        L.append(f"K = {k}: {r['fall']['n']} falls, {r['surviving']['n']} matched surviving windows; depth-0 parity "
                 f"{max(r['depth0_parity_max_rel'].values()):.1e}")
        for name in ("fall", "surviving"):
            x = r[name]
            L.append(f"  {name:9s} H {x['H']:.3f} {f3(x['H_ci'])} = H_mem {x['H_mem']:.3f} {f3(x['H_mem_ci'])} + "
                     f"H_src {x['H_src']:.3f} {f3(x['H_src_ci'])}")
        L.append(f"  dH_src (fall - surviving) {r['dH_src']['value']:.3f} {f3(r['dH_src']['ci'])}"
                 + ("; UNSTABLE DENOMINATOR" if r["unstable_denominator"] else ""))
    text = "\n".join(L) + "\n"
    (out / "go2_c0m.txt").write_text(text)
    print(text, flush=True)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True, help="the D2 model 080f353d... (as go2_c0_check)")
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--probe", action="append", required=True, help="name=path (911, 912), repeatable")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args(argv)
    run(a.wm, a.norm, [tuple(s.split("=", 1)) for s in a.probe], a.out, a.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
