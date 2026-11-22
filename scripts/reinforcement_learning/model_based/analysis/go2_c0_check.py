#!/usr/bin/env python3
"""A4 Go2 C0 check (22 Nov 2026, review decision): does the ensemble warning near falls survive when the model is
fed its own predictions? Read-only, CPU. Model: the uncalibrated D2 fit (080f353d...). Data: the two friction probes
(seeds 911, 912; never fitted).

Signal: EXACTLY the one PPO penalizes (SystemDynamicsEnsemble.forward): u = std over the five heads (ddof 1) of the
head state means, summed over the 45 normalized dims.
Windows (fixed before the run):
  fall window      : target row T = terminal row - K (K = 10 rows = 0.2 s before impact, primary; K = 0, the ANYmal
                     C0 endpoint, reported too); needs 32 recorded rows before T - D in the same episode.
  surviving window : same length, from episodes that end by time-out, target row T with >= 25 rows of survival after
                     it and the same history rule; matched per fall to the nearest window (standardized commanded
                     vx, vy, wz and measured speed at the window start; measured vx, vy, wz if no command was
                     recorded), without replacement, seeded order, within the same probe.
  recorded arm     : u_ctx(T) = one-step prediction of row T from the 32 recorded rows before it.
  imagined arm     : from the 32 recorded rows before T - D, each head drives its own copy with its own mean and the
                     recorded actions; after D = 25 fed-back states the copy predicts row T: u_imag(h, T).
Statistics (registered C0 definitions): HI = mean over (window, driving head) of log(u_imag / u_ctx); H = -HI;
ratio H_fall / H_surv with an episode-cluster bootstrap stratified by probe (10,000); verdict by the registered C0 rule
(c0_battery.c0_verdict: supported if ratio >= 1.5 with CI excluding 1). Also: the warning ratio (geometric mean of u on
falls / on surviving windows) in each arm, and AUROC of u for falls vs surviving windows in each arm (advance warning
at T, with a matched false-alarm reference).
Depth-0 check: at the first imagined step both arms see the same inputs, so u_imag == u_ctx there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import go2_g0_replay as rp  # noqa: E402
from c0_battery import c0_verdict, ci  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

H_CTX, DEPTH, SURVIVE, STRIDE = 32, 25, 25, 5


def ppo_spread(means):
    """[E, B, 45] -> [B]: the epistemic uncertainty PPO penalizes."""
    return means.std(dim=0).sum(dim=-1)


@torch.no_grad()
def recorded_u(sd, eps, rows, device, batch=1024):
    out = np.empty(len(rows), np.float64)
    for b0 in range(0, len(rows), batch):
        ch = rows[b0:b0 + batch]
        xs = torch.from_numpy(np.stack([eps[i].S[t - H_CTX:t] for i, t in ch])).to(device)
        xa = torch.from_numpy(np.stack([eps[i].A[t - H_CTX + 1:t + 1] for i, t in ch])).to(device)
        sd.reset()
        means, _ = rp.all_heads(sd, xs, xa)
        out[b0:b0 + len(ch)] = ppo_spread(means).cpu().numpy()
    sd.reset()
    return out


@torch.no_grad()
def imagined_u(sd, eps, rows, device, depth=DEPTH):
    """[n, E, depth + 1]: u at each step on the copy driven by head h; the last step predicts row T."""
    E, n = len(sd.state_heads), len(rows)
    starts = [(i, t - depth) for i, t in rows]
    rep = lambda a: torch.from_numpy(np.repeat(a, E, axis=0)).to(device)
    drive = torch.arange(E, device=device).repeat(n)
    cols = torch.arange(n * E, device=device)
    xs = rep(np.stack([eps[i].S[j - H_CTX:j] for i, j in starts]))
    xa = rep(np.stack([eps[i].A[j - H_CTX + 1:j + 1] for i, j in starts]))
    U = np.empty((n, E, depth + 1), np.float64)
    sd.reset()
    for m in range(depth + 1):
        means, _ = rp.all_heads(sd, xs, xa)
        U[:, :, m] = ppo_spread(means).view(n, E).cpu().numpy()
        if m < depth:
            xs = means[drive, cols].unsqueeze(1)
            xa = rep(np.stack([eps[i].A[j + m + 1:j + m + 2] for i, j in starts]))
    sd.reset()
    return U


def episodes_of(trace, norm, probe):
    import go2_round0_prepare as prep
    eps = []
    for n, s, e, finished in prep.episodes(trace["done"]):
        if not finished:
            continue
        kind = "fall" if (trace["terminated"][e, n] and not trace["time_out"][e, n]) else "time_out"
        ep = rp.Episode(trace, dict(env=n, first=s, last=e, kind=kind), norm)
        ep.probe, ep.cluster = probe, f"{probe}/{n}/{s}"
        raw = trace["next_state"][s:e + 1, n].astype(np.float64)
        cmd = trace["command"][s:e + 1, n, :3].astype(np.float64) if "command" in trace else raw[:, [0, 1, 5]]
        ep.feat = np.column_stack([cmd, np.linalg.norm(raw[:, 0:2], axis=1)])
        eps.append(ep)
    return eps, ("command" in trace)


def windows(eps, k, depth=DEPTH, seed=0):
    """Fall rows and matched surviving rows [(episode index, T)], per probe; counts."""
    falls, excluded = [], 0
    for i, ep in enumerate(eps):
        if ep.kind == "fall":
            T = ep.L - 1 - k
            if T - depth - H_CTX >= 0:
                falls.append((i, T))
            else:
                excluded += 1
    cands = [(i, T) for i, ep in enumerate(eps) if ep.kind == "time_out"
             for T in range(depth + H_CTX, ep.L - SURVIVE, STRIDE)]
    chosen = []
    for probe in sorted({eps[i].probe for i, _ in falls}):
        fp = [w for w in falls if eps[w[0]].probe == probe]
        cp = [w for w in cands if eps[w[0]].probe == probe]
        if len(cp) < len(fp):
            raise ValueError(f"{probe}: only {len(cp)} surviving windows for {len(fp)} falls")
        f = np.array([eps[i].feat[T - depth] for i, T in fp])
        c = np.array([eps[i].feat[T - depth] for i, T in cp])
        sc = c.std(0); sc[sc == 0] = 1.0
        free = np.ones(len(cp), bool)
        pick = [None] * len(fp)
        for a in np.random.default_rng(seed).permutation(len(fp)):
            d = np.linalg.norm(c / sc - f[a] / sc, axis=1)
            d[~free] = np.inf
            b = int(np.argmin(d)); pick[a] = cp[b]; free[b] = False
        chosen += list(zip(fp, pick))
    return [w for w, _ in chosen], [s for _, s in chosen], dict(falls_used=len(chosen), falls_too_short=excluded,
                                                               surviving_candidates=len(cands))


def auroc(pos, neg):
    x = np.concatenate([pos, neg]); r = np.argsort(np.argsort(x)) + 1.0
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)       # average ranks for ties
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap(vf, vs, cf, cs, sf, ss, n_boot=10_000, seed=0):
    """Paired cluster bootstrap (episodes), stratified by probe: H_fall*, H_surv* per resample."""
    g = np.random.default_rng(seed)
    def groups(c, s):
        out = []
        for stratum in sorted(set(s)):
            cl = sorted({c[i] for i in range(len(c)) if s[i] == stratum})
            out.append([np.array([i for i in range(len(c)) if c[i] == k]) for k in cl])
        return out
    gf, gs = groups(cf, sf), groups(cs, ss)
    def draw(v, gr):
        idx = np.concatenate([grp[k] for grp in gr for k in g.choice(len(grp), len(grp))])
        return -float(np.mean(v[idx]))
    out_f, out_s = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        out_f[b], out_s[b] = draw(vf, gf), draw(vs, gs)
    return out_f, out_s


def analyse(sd, eps, k, device, n_boot, seed=0):
    fw, sw, counts = windows(eps, k, seed=seed)
    res = dict(K=k, counts=counts)
    vals = {}
    for name, rows in (("fall", fw), ("surviving", sw)):
        uc = recorded_u(sd, eps, rows, device)
        ui = imagined_u(sd, eps, rows, device)
        d0 = recorded_u(sd, eps, [(i, T - DEPTH) for i, T in rows], device)
        if (uc <= 0).any() or (ui[:, :, -1] <= 0).any():
            raise ValueError("zero head spread: the log ratio is undefined (registered C0 rule)")
        lr = np.log(ui[:, :, -1] / uc[:, None]).mean(1)                    # per window: mean over driving heads
        vals[name] = (lr, [eps[i].cluster for i, _ in rows], [eps[i].probe for i, _ in rows])
        res[name] = dict(n=len(rows), H=float(-lr.mean()), u_ctx_geomean=float(np.exp(np.log(uc).mean())),
                         u_imag_geomean=float(np.exp(np.log(ui[:, :, -1]).mean())),
                         depth0_max_rel_diff=float(np.max(np.abs(ui[:, :, 0] - d0[:, None]) / d0[:, None])),
                         u_ctx=uc.tolist(), u_imag_mean_over_heads=ui[:, :, -1].mean(1).tolist(),
                         log_u_imag_vs_depth0=[float(np.log(ui[:, :, m] / ui[:, :, 0]).mean()) for m in range(DEPTH + 1)])
    bf, bs = bootstrap(vals["fall"][0], vals["surviving"][0], vals["fall"][1], vals["surviving"][1],
                       vals["fall"][2], vals["surviving"][2], n_boot, seed)
    res["verdict"] = c0_verdict(res["fall"]["H"], bf, res["surviving"]["H"], bs)
    res["H_fall_ci"], res["H_surv_ci"] = ci(bf), ci(bs)
    uf, us = np.array(res["fall"]["u_ctx"]), np.array(res["surviving"]["u_ctx"])
    vf, vs = np.array(res["fall"]["u_imag_mean_over_heads"]), np.array(res["surviving"]["u_imag_mean_over_heads"])
    res["warning_ratio"] = dict(recorded=float(np.exp(np.log(uf).mean() - np.log(us).mean())),
                                imagined=float(np.exp(np.log(vf).mean() - np.log(vs).mean())))
    res["auroc"] = dict(recorded=auroc(uf, us), imagined=auroc(vf, vs))
    return res


def run(wm, norm_path, probes, out, n_boot=10_000, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    torch.set_num_threads(8)
    sd, H, norm, _ = rp.load_model(wm, norm_path, device)
    eps, has_cmd = [], []
    for name, path in probes:
        with np.load(path, allow_pickle=False) as z:
            validate_trace(z)
            tr = {k: z[k] for k in z.files if k != "metadata"}
        e, c = episodes_of(tr, norm, name)
        eps += e; has_cmd.append(c)
    rep = dict(protocol="A4 Go2 C0 check (exact PPO signal; endpoint before impact; matched surviving windows)",
               wm_sha256=sha256(wm), probes={n: sha256(p) for n, p in probes}, matched_on_command=all(has_cmd),
               depth=DEPTH, results={str(k): analyse(sd, eps, k, device, n_boot) for k in (10, 0)},
               analysis_sha256=sha256(__file__), wall_s=None)
    rep["wall_s"] = time.time() - t0
    (out / "go2_c0.json").write_text(json.dumps(rep, indent=1, default=float))
    L = ["A4 Go2 C0 check (read-only; exact PPO disagreement signal)"]
    for k in ("10", "0"):
        r = rep["results"][k]
        v = r["verdict"]
        L.append(f"K = {k} rows before impact ({'primary' if k == '10' else 'ANYmal C0 endpoint, includes the fall step'}): "
                 f"{r['fall']['n']} falls ({r['counts']['falls_too_short']} too short), {r['surviving']['n']} matched surviving windows")
        f3 = lambda c: "[" + ", ".join(f"{x:.3f}" for x in c) + "]"
        L.append(f"  H_fall {r['fall']['H']:.3f} {f3(r['H_fall_ci'])}; H_surv {r['surviving']['H']:.3f} {f3(r['H_surv_ci'])}; "
                 f"rule: {v.get('reading')}; ratio {v.get('ratio', float('nan')):.2f} "
                 f"{f3(v.get('ratio_ci', v.get('h_fall_ci')))} -> {v['verdict'].upper()}")
        L.append(f"  warning ratio (u falls / u surviving): recorded {r['warning_ratio']['recorded']:.2f}, imagined "
                 f"{r['warning_ratio']['imagined']:.2f}; AUROC recorded {r['auroc']['recorded']:.3f}, imagined {r['auroc']['imagined']:.3f}")
        L.append(f"  depth-0 check (max rel diff): falls {r['fall']['depth0_max_rel_diff']:.1e}, surviving {r['surviving']['depth0_max_rel_diff']:.1e}")
    text = "\n".join(L) + "\n"
    (out / "go2_c0.txt").write_text(text)
    print(text, flush=True)
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--probe", action="append", required=True, help="name=path, repeatable")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args()
    run(a.wm, a.norm, [tuple(s.split("=", 1)) for s in a.probe], a.out, a.n_boot)


if __name__ == "__main__":
    main()
