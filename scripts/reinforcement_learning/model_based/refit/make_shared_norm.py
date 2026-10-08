#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compute ONE state/action normalizer over the union of several datasets.

Inputs: pilot-format directories (files taken in manifest order) and/or CSV globs (files with
!= 66 columns, e.g. manifest.csv, are skipped). Files reached twice (same realpath) are counted
once. ALL rows are used (nothing excluded); rows with an all-zero action (reset-like rows) are
counted and printed for information only.

Statistics are exactly those of fit_world_model_go2_pilot_segments.compute_normalizer, streamed:
values are parsed as float32 (like the fitter), accumulated in float64,
    mean = sum / n,  std = sqrt(max(sumsq / n - mean^2, 1e-12)) + 1e-6   (population std + eps).

For pilot-format inputs (manifest with steps) it also counts terminations on the last row of an
env block (should be ~0 if seam labels were cleaned, code review 1.2 Q4) and terminations with
an all-zero action (expected for every real fall under the reset-row semantics, Q1.4).

Output .npz (allow_pickle=False readable): state_mean (45,), state_std (45,), action_mean (12,),
action_std (12,) [float64], n_rows, eps, meta_json (files, rows per file, reset-like rows, ...).
A human-readable copy of the metadata goes to <out>.json.

Usage:
  python refit/make_shared_norm.py --inputs <curated_segments_dir> <plus_fail_dir> --out shared_norm.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_common as rc  # noqa: E402

EPS = 1e-6
VAR_FLOOR = 1e-12


def resolve_inputs(items):
    files = []
    for it in items:
        if os.path.isdir(it):
            if not os.path.isfile(os.path.join(it, "manifest.csv")):
                raise SystemExit(f"{it}: directory without manifest.csv")
            m = rc.read_manifest(it)
            for name, steps in zip(m["name"], m["steps"]):
                files.append((it, os.path.join(it, f"{name}.csv"), int(steps)))
        else:
            matches = sorted(glob.glob(it))
            if not matches:
                raise SystemExit(f"input matched nothing: {it}")
            for p in matches:
                if os.path.isfile(p):
                    files.append((it, p, 0))
    out, seen = [], {}
    for src, p, steps in files:
        rp = os.path.realpath(p)
        if rp in seen:
            print(f"[norm] WARNING skipping duplicate file {p} (same as {seen[rp]})")
            continue
        seen[rp] = p
        out.append((src, p, steps))
    return out


def stream_file(path: str, chunksize: int):
    """Yield float32 chunks of a 66-column CSV; None if the file is not 66 columns."""
    head = pd.read_csv(path, header=None, nrows=1)
    if head.shape[1] != rc.NUM_COLS:
        return None
    return (c.to_numpy(dtype=np.float32) for c in pd.read_csv(path, header=None, chunksize=chunksize))


def compute(files, chunksize: int = 200000):
    s_sum = np.zeros(45)
    s_sq = np.zeros(45)
    a_sum = np.zeros(12)
    a_sq = np.zeros(12)
    n = 0
    info = []
    for src, p, steps in files:
        chunks = stream_file(p, chunksize)
        if chunks is None:
            print(f"[norm] skip {p}: not {rc.NUM_COLS} columns")
            continue
        rows = zero = terms = terms_zero = terms_last = 0
        for c in chunks:
            if c.shape[1] != rc.NUM_COLS or not np.isfinite(c).all():
                raise SystemExit(f"{p}: bad chunk (columns {c.shape[1]} or non-finite values)")
            s = c[:, rc.STATE].astype(np.float64)
            a = c[:, rc.ACTION].astype(np.float64)
            s_sum += s.sum(axis=0)
            s_sq += (s * s).sum(axis=0)
            a_sum += a.sum(axis=0)
            a_sq += (a * a).sum(axis=0)
            z = np.abs(c[:, rc.ACTION]).max(axis=1) == 0
            t = c[:, rc.TERM_COL] > 0.5
            zero += int(z.sum())
            terms += int(t.sum())
            terms_zero += int((t & z).sum())
            if steps:
                last = (np.arange(rows, rows + len(c)) % steps) == steps - 1
                terms_last += int((t & last).sum())
            rows += len(c)
        n += rows
        info.append({"input": src, "path": p, "realpath": os.path.realpath(p), "rows": rows,
                     "reset_like_zero_action_rows": zero, "terms": terms, "terms_zero_action": terms_zero,
                     "block_steps": steps or None, "terms_on_block_last_row": terms_last if steps else None})
        print(f"[norm] {p}: rows={rows} zero_action_rows={zero} terms={terms} terms_with_zero_action={terms_zero}"
              + (f" terms_on_block_last_row={terms_last}" if steps else ""))
    if n == 0:
        raise SystemExit("no rows")
    s_mean, a_mean = s_sum / n, a_sum / n
    s_std = np.sqrt(np.maximum(s_sq / n - s_mean * s_mean, VAR_FLOOR)) + EPS
    a_std = np.sqrt(np.maximum(a_sq / n - a_mean * a_mean, VAR_FLOOR)) + EPS
    return {"state_mean": s_mean, "state_std": s_std, "action_mean": a_mean, "action_std": a_std}, n, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True, help="pilot-format dirs and/or CSV globs")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--chunksize", type=int, default=200000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists (use --force)")
    files = resolve_inputs(args.inputs)
    stats, n, info = compute(files, args.chunksize)
    by_input = {}
    for f in info:
        d = by_input.setdefault(f["input"], {"rows": 0, "reset_like_zero_action_rows": 0, "terms": 0,
                                             "terms_zero_action": 0, "terms_on_block_last_row": 0, "files": 0})
        for k in ("rows", "reset_like_zero_action_rows", "terms", "terms_zero_action"):
            d[k] += f[k]
        d["terms_on_block_last_row"] += f["terms_on_block_last_row"] or 0
        d["files"] += 1
    zero_total = sum(f["reset_like_zero_action_rows"] for f in info)
    meta = {"kind": "shared_norm", "formula": "population std: sqrt(max(E[x^2]-mean^2, 1e-12)) + 1e-6; float32 parse, float64 sums",
            "eps": EPS, "n_rows": n, "reset_like_zero_action_rows": zero_total, "by_input": by_input, "files": info,
            "provenance": rc.provenance(__file__, args)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fh:  # file handle: np.savez would append .npz to a path without it
        np.savez(fh, **stats, n_rows=np.int64(n), eps=np.float64(EPS), meta_json=np.array(json.dumps(meta)))
    rc.write_json(args.out + ".json", {**meta, **{k: v.tolist() for k, v in stats.items()}})
    for k, d in by_input.items():
        print(f"[norm] input {k}: files={d['files']} rows={d['rows']} reset_like_zero_action_rows={d['reset_like_zero_action_rows']} "
              f"terms={d['terms']} terms_with_zero_action={d['terms_zero_action']} terms_on_block_last_row={d['terms_on_block_last_row']}")
    print(f"[norm] total rows={n} reset_like_zero_action_rows={zero_total} ({100.0 * zero_total / n:.3f}%, kept)")
    print(f"[norm] state_std[:3]={stats['state_std'][:3].tolist()} action_std[:3]={stats['action_std'][:3].tolist()}")
    print(f"[norm] wrote {args.out} (sha256 {rc.sha256_file(args.out)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
