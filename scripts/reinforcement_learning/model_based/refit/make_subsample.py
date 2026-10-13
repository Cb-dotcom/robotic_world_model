#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build a pilot-format directory holding a random subset of WHOLE env blocks of a pilot-format
source directory (e.g. the +fail training set), with total rows close to a target.

* An env block = the ``steps`` consecutive rows of one env in an env-major file.
* Stratified per source file: file f gets ``k_f`` blocks with ``k_f * steps_f`` close to
  ``target * rows_f / total_rows``. Allocation: floor of the proportional block count, then
  one extra block per file in order of largest fractional remainder (ties: manifest order),
  each only if it brings the total closer to the target. The achieved total cannot be exact;
  it must be within ``--tol`` (relative, default 2 %) or the script fails.
* Which blocks: ``rng = np.random.default_rng(seed)``; for each file in manifest order
  ``sorted(rng.choice(num_envs, k_f, replace=False))``.
* Rows are copied VERBATIM (text lines); each output file is an env-major file with
  ``num_envs = k_f`` (same ``steps``), listed in ``manifest.csv`` (name, steps, rows, num_envs).
* ``cards.json``: totals (rows, real terms, files, blocks) and per-file kept env ids.

Usage:
  python refit/make_subsample.py --src <root>/assets/data/go2_pilot_1m/segments_plus_fail_train_flat \
      --out <dir> --target_rows 115000 --seed 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_common as rc  # noqa: E402


def allocate_blocks(rows, steps, num_envs, target):
    """Blocks per file (see module docstring). Returns a list of ints."""
    rows = np.asarray(rows, dtype=np.float64)
    total = rows.sum()
    exact = [target * r / total / s for r, s in zip(rows, steps)]
    k = [min(int(math.floor(e)), n) for e, n in zip(exact, num_envs)]
    cur = sum(ki * s for ki, s in zip(k, steps))
    order = sorted(range(len(k)), key=lambda i: (-(exact[i] - math.floor(exact[i])), i))
    for i in order:
        if k[i] < num_envs[i] and abs(cur + steps[i] - target) < abs(cur - target):
            k[i] += 1
            cur += steps[i]
    return k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="pilot-format source directory (manifest.csv + CSVs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_rows", type=int, default=115000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.02, help="max relative |achieved - target| / target")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = ap.parse_args(argv)

    m = rc.read_manifest(args.src)
    for r in m.itertuples(index=False):
        if r.rows != r.steps * r.num_envs:
            raise SystemExit(f"{args.src}/manifest.csv: {r.name}: rows {r.rows} != steps*num_envs {r.steps}*{r.num_envs}")
    k = allocate_blocks(m["rows"].tolist(), m["steps"].tolist(), m["num_envs"].tolist(), args.target_rows)
    achieved = int(sum(ki * s for ki, s in zip(k, m["steps"])))
    dev = (achieved - args.target_rows) / args.target_rows
    print(f"[subsample] target={args.target_rows} achieved={achieved} deviation={100 * dev:+.2f}% "
          f"blocks={sum(k)} files_with_blocks={sum(1 for x in k if x > 0)}/{len(k)}")
    if abs(dev) > args.tol:
        raise SystemExit(f"[subsample] achieved rows {achieved} deviate {100 * dev:+.2f}% from target "
                         f"{args.target_rows} (> {100 * args.tol:.1f}%); re-run with a larger --tol if acceptable")

    rc.prepare_out_dir(args.out, args.force)
    rng = np.random.default_rng(args.seed)
    cards, manifest = [], []
    for r, kf in zip(m.itertuples(index=False), k):
        src_path = os.path.join(args.src, f"{r.name}.csv")
        lines = rc.read_lines(src_path)
        if len(lines) != r.rows:
            raise SystemExit(f"{src_path}: {len(lines)} lines != manifest rows {r.rows}")
        src_terms = sum(1 for ln in lines if float(ln.rsplit(",", 1)[1]) > 0.5)
        envs = sorted(rng.choice(r.num_envs, size=kf, replace=False).tolist()) if kf > 0 else []
        card = {"name": r.name, "source": src_path, "source_realpath": os.path.realpath(src_path),
                "source_rows": int(r.rows), "source_terms": int(src_terms), "steps": int(r.steps),
                "source_num_envs": int(r.num_envs), "blocks_kept": int(kf), "env_ids_kept": envs}
        if kf > 0:
            out_lines = [ln for e in envs for ln in lines[e * r.steps:(e + 1) * r.steps]]
            x = rc.parse_lines(out_lines, f"{src_path} (kept blocks)")
            term = x[:, rc.TERM_COL] > 0.5
            out_path = os.path.join(args.out, f"{r.name}.csv")
            rc.write_lines(out_path, out_lines)
            back = rc.read_numeric(out_path)
            if not np.array_equal(back, x):
                raise RuntimeError(f"{out_path}: re-read data differs")
            card.update({"rows_kept": int(len(x)), "real_terms": int(term.sum()),
                         "terms_on_block_last_row": int(term[r.steps - 1::r.steps].sum()),
                         "zero_action_rows": int(rc.zero_action_mask(x).sum()),
                         "out_sha256": rc.sha256_file(out_path)})
            manifest.append(f"{r.name},{r.steps},{len(x)},{kf}")
        else:
            card.update({"rows_kept": 0, "real_terms": 0, "terms_on_block_last_row": 0, "zero_action_rows": 0})
        cards.append(card)
        print(f"[subsample] {r.name}: steps={r.steps} blocks {kf}/{r.num_envs} envs={envs} rows={card['rows_kept']} "
              f"terms={card['real_terms']} (source terms {src_terms})")
    rc.write_lines(os.path.join(args.out, "manifest.csv"), ["name,steps,rows,num_envs"] + manifest)
    totals = {"rows": sum(c["rows_kept"] for c in cards), "real_terms": sum(c["real_terms"] for c in cards),
              "files": len(manifest), "blocks": sum(k), "target_rows": args.target_rows,
              "deviation_rel": dev, "source_rows": int(m["rows"].sum()),
              "source_terms": sum(c["source_terms"] for c in cards),
              "terms_on_block_last_row": sum(c["terms_on_block_last_row"] for c in cards)}
    assert totals["rows"] == achieved
    rc.write_json(os.path.join(args.out, "cards.json"),
                  {"kind": "subsample", "totals": totals, "files": cards,
                   "source_manifest_sha256": rc.sha256_file(os.path.join(args.src, "manifest.csv")),
                   "provenance": rc.provenance(__file__, args)})
    print(f"[subsample] wrote {args.out}: rows={totals['rows']} real_terms={totals['real_terms']} "
          f"files={totals['files']} blocks={totals['blocks']} (source {totals['source_rows']} rows, "
          f"{totals['source_terms']} terms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
