#!/usr/bin/env python3
"""Plan v6 C2: are imagined fall paths pulled toward the walking data? CPU only.

Reads c0_battery run folders (c0_windows.npz: each path's fed state `own`, the real `target`, the run's
state normalization) and a walking reference: ETH's 10k CSV (first 45 columns = state, physical units)
and, optionally, the states of pre-reset traces (subsampled). Per fall window and depth:
* kNN distance = mean distance to the k = 10 nearest reference states, in the run's normalized units,
  of the imagined state (each driving head's own path = per-member paths) and of the real state;
* tilt = angle in degrees between the physical gravity vector and straight down (any vector length;
  predicted gravity is not constrained to unit length, so its length is reported too).
Rule (plan §2): supported if imagined distance and tilt fall while real ones rise in >= 70 % of falls,
on per-member paths too. Operational reading (fixed here): per window and path, "fall" / "rise" = value
at d = 25 below / above its value at d = 0. Supported if the pooled share >= 0.70 and every head's own
paths >= 0.70; not supported if the pooled share's 95 % CI (cluster bootstrap) lies below 0.70;
otherwise inconclusive. This is "consistent with attraction toward the available walking data"; it
does not show attraction toward ETH's original training distribution (plan §11 #5). Run on a free-run
folder (c0 --actions held), it reads the same numbers under a held action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from c0_battery import DEPTHS, ci, cluster_bootstrap, json_safe, tilt_deg

THRESHOLD = 0.70
GRAV = slice(6, 9)


def knn_distance(queries, reference, k=10, chunk=4096):
    """Mean Euclidean distance to the k nearest reference rows. queries [N, D], reference [R, D]."""
    q = torch.as_tensor(np.asarray(queries, np.float32))
    r = torch.as_tensor(np.asarray(reference, np.float32))
    out = []
    for b in range(0, len(q), chunk):
        d = torch.cdist(q[b:b + chunk], r)
        out.append(d.topk(min(k, len(r)), largest=False).values.mean(1))
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def load_reference(csv_paths, trace_paths, max_rows, seed):
    rows = [np.loadtxt(p, delimiter=",", dtype=np.float32)[:, :45] for p in csv_paths]
    for p in trace_paths:
        with np.load(p, allow_pickle=False) as z:
            st = z["state"].reshape(-1, 45).astype(np.float32)
        rows.append(st)
    ref = np.concatenate(rows)
    if len(ref) > max_rows:
        ref = ref[np.sort(np.random.default_rng(seed).choice(len(ref), max_rows, replace=False))]
    return ref


def tilt(normalized, mean, std):
    """Tilt angle in degrees of the (physical) gravity vector, any vector length (c0_battery.tilt_deg)."""
    g = normalized[..., GRAV] * std[GRAV] + mean[GRAV]
    return tilt_deg(np.asarray(g, np.float32)).numpy()


def gravity_norm(normalized, mean, std):
    return np.linalg.norm(normalized[..., GRAV] * std[GRAV] + mean[GRAV], axis=-1)


def analyse_run(folder, reference_phys, sources, k, n_boot, seed):
    with np.load(Path(folder) / "c0_windows.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    mean, std = data["meta/state_mean"].astype(np.float32), data["meta/state_std"].astype(np.float32)
    ref = (reference_phys - mean) / std
    if not sources:
        sources = sorted({key.split("/")[0] for key in data if not key.startswith("meta/")})
    out = {}
    for kind in ("fall", "nonfall"):
        flags, clusters, strata, prof = [], [], [], {}
        parts = {"imag_dist": [], "real_dist": [], "imag_tilt": [], "real_tilt": [], "imag_gnorm": []}
        for s in sources:
            own, target = data[f"{s}/{kind}/own"], data[f"{s}/{kind}/target"]           # [P,H,W,45], [H,W,45]
            P, H, W, _ = own.shape
            dsel = [d for d in DEPTHS if d < H]
            di = knn_distance(own[:, dsel].reshape(-1, 45), ref, k).reshape(P, len(dsel), W)
            dr = knn_distance(target[dsel].reshape(-1, 45), ref, k).reshape(len(dsel), W)
            ti, tr = tilt(own[:, dsel], mean, std), tilt(target[dsel], mean, std)          # [P,D,W], [D,W]
            last = dsel.index(25)
            imag_dist_falls = di[:, last] < di[:, 0]
            real_dist_rises = dr[last] > dr[0]
            imag_tilt_falls = ti[:, last] < ti[:, 0]
            real_tilt_rises = tr[last] > tr[0]
            attracted = imag_dist_falls & imag_tilt_falls & (real_dist_rises & real_tilt_rises)[None]   # [P, W]
            flags.append(np.stack([attracted, imag_dist_falls, imag_tilt_falls,
                                   np.broadcast_to(real_dist_rises, attracted.shape),
                                   np.broadcast_to(real_tilt_rises, attracted.shape)], -1).astype(float))  # [P,W,5]
            clusters.append(data[f"{s}/{kind}/clusters"])
            strata.append([s] * W)
            gn = gravity_norm(own[:, dsel], mean, std).mean(0)
            for name, v in (("imag_dist", di.mean(0)), ("real_dist", dr), ("imag_tilt", ti.mean(0)), ("real_tilt", tr),
                            ("imag_gnorm", gn)):
                parts[name].append(v)
        f = np.concatenate(flags, axis=1)                                                  # [P, Wtot, 5]
        cl, st = np.concatenate(clusters), np.concatenate(strata)
        point, boot = cluster_bootstrap(f.mean(0), cl, st, n_boot, seed)                     # per-window share over paths
        per_head = f[..., 0].mean(1)
        res = dict(windows=int(f.shape[1]), paths_per_window=int(f.shape[0]),
                   share_attracted=float(point[0]), share_attracted_ci=ci(boot[:, 0]),
                   share_attracted_per_head=[float(x) for x in per_head],
                   share_imag_dist_falls=float(point[1]), share_imag_tilt_falls=float(point[2]),
                   share_real_dist_rises=float(point[3]), share_real_tilt_rises=float(point[4]),
                   profile={name: {f"d{d}": float(np.concatenate(v, axis=-1)[i].mean()) for i, d in enumerate(dsel)}
                            for name, v in parts.items()})
        if kind == "fall":
            lo, hi = res["share_attracted_ci"]
            if res["share_attracted"] >= THRESHOLD and min(per_head) >= THRESHOLD:
                verdict = "supported"
            elif hi < THRESHOLD:
                verdict = "not supported"
            else:
                verdict = "inconclusive"
            res["verdict"] = verdict
        out[kind] = res
    return out


def text(results, meta):
    lines = [f"Plan v6 C2 attraction (k = {meta['k']}, reference rows {meta['reference_rows']}, threshold {THRESHOLD:.0%})"]
    for name, r in results.items():
        f, n = r["fall"], r["nonfall"]
        lines.append(f"\n== {name}: falls {f['windows']} windows x {f['paths_per_window']} paths -> C2 {f['verdict'].upper()}")
        lines.append(f"  falls: attracted {f['share_attracted']:.3f} [{f['share_attracted_ci'][0]:.3f}, {f['share_attracted_ci'][1]:.3f}]"
                     f" (per head min {min(f['share_attracted_per_head']):.3f}); imag dist falls {f['share_imag_dist_falls']:.3f}, "
                     f"imag tilt falls {f['share_imag_tilt_falls']:.3f}, real dist rises {f['share_real_dist_rises']:.3f}, "
                     f"real tilt rises {f['share_real_tilt_rises']:.3f}")
        lines.append(f"  non-falls: attracted {n['share_attracted']:.3f}; imag dist falls {n['share_imag_dist_falls']:.3f}, "
                     f"real dist rises {n['share_real_dist_rises']:.3f}")
        for kind in ("fall", "nonfall"):
            p = r[kind]["profile"]
            lines.append(f"  {kind} profile d: " + "; ".join(
                f"{d} dist imag/real {p['imag_dist'][d]:.2f}/{p['real_dist'][d]:.2f} tilt deg {p['imag_tilt'][d]:.1f}/"
                f"{p['real_tilt'][d]:.1f} |g| imag {p['imag_gnorm'][d]:.3f}"
                for d in p["imag_dist"]))
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=DIR", help="c0_battery output folder (repeat)")
    p.add_argument("--reference-csv", action="append", default=[], help="walking CSV (first 45 columns = state)")
    p.add_argument("--reference-trace", action="append", default=[], help="pre-reset trace whose states join the reference")
    p.add_argument("--max-reference", type=int, default=60_000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--sources", default="")
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists; use a fresh one")
    if not a.reference_csv and not a.reference_trace:
        p.error("give at least one --reference-csv or --reference-trace")
    ref = load_reference(a.reference_csv, a.reference_trace, a.max_reference, a.seed)
    sources = a.sources.split(",") if a.sources else []
    results = {}
    for item in a.run:
        name, _, folder = item.partition("=")
        if not name or not folder:
            p.error(f"bad --run {item!r}")
        results[name] = analyse_run(folder, ref, sources, a.k, a.bootstrap, a.seed)
    meta = dict(runs=a.run, reference_csv=a.reference_csv, reference_trace=a.reference_trace,
                reference_rows=int(len(ref)), k=a.k, bootstrap=a.bootstrap, seed=a.seed, threshold=THRESHOLD)
    a.out.mkdir(parents=True)
    (a.out / "c2_report.json").write_text(json.dumps(json_safe(dict(meta=meta, results=results)), indent=1, allow_nan=False))
    t = text(results, meta)
    (a.out / "c2_report.txt").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
