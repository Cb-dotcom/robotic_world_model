#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build a pilot-format directory (manifest.csv + one CSV per segment, num_envs=1) from the
curated single-env segment files, removing forced seam labels.

For each input segment file (in the given order):
  * data rows are copied VERBATIM (text lines), so the fitter reads identical float32 values;
  * if the LAST row has termination=1 and a NON-zero action, it is a forced seam (a real fall
    row holds the spawn state and an all-zero action, code review 1.2 Q1.4): the label is set
    to 0 and counted as ``seams_removed``;
  * if the last row has termination=1 and an all-zero action, it is kept (real fall);
  * interior termination labels are never changed; interior terminations with a non-zero
    action contradict the reset-row hypothesis and are reported loudly.

Writes ``<out>/<name>.csv`` (name = input file stem), ``<out>/manifest.csv`` (name, steps=rows,
rows, num_envs=1) and ``<out>/cards.json``.

Cross-check (``--concat``): the concatenation of the input files (before label edits) should
equal the concatenated curated CSV except for termination labels. Mismatches are reported
(printed and stored in cards.json) but do not fail the script.

Usage:
  python refit/make_curated_segments.py --segs <root>/assets/data/go2_noise/seg_n{00,02,04,08,10,12}.csv \
      --concat <root>/assets/data/go2_noise/state_action_data_0.csv --out <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_common as rc  # noqa: E402


def relabel_seam_line(line: str) -> str:
    """Set the termination field (last CSV field) of a data line to 0, keeping its style."""
    head, last = line.rsplit(",", 1)
    new = "0.0" if any(c in last for c in ".eE") else "0"
    return f"{head},{new}"


def process_segment(path: str, zero_tol: float):
    lines = rc.read_lines(path)
    x = rc.parse_lines(lines, path)
    n = len(x)
    term = x[:, rc.TERM_COL] > 0.5
    zero = rc.zero_action_mask(x, zero_tol)
    out_lines = list(lines)
    x_out = x.copy()
    status = "no_term"
    seams_removed = 0
    if n and term[-1]:
        if zero[-1]:
            status = "real_fall_kept"
        else:
            status = "seam_removed"
            seams_removed = 1
            out_lines[-1] = relabel_seam_line(lines[-1])
            x_out[-1, rc.TERM_COL] = 0.0
    interior = np.where(term[:-1])[0] if n else np.zeros(0, dtype=int)
    card = {
        "source": path,
        "source_realpath": os.path.realpath(path),
        "source_sha256": rc.sha256_file(path),
        "rows": int(n),
        "terms_raw": int(term.sum()),
        "last_row_status": status,
        "last_row_max_abs_action": float(np.abs(x[-1, rc.ACTION]).max()) if n else None,
        "seams_removed": int(seams_removed),
        "real_terms": int((x_out[:, rc.TERM_COL] > 0.5).sum()),
        "interior_terms": int(len(interior)),
        "interior_terms_zero_action": int(zero[interior].sum()),
        "interior_terms_nonzero_action": int((~zero[interior]).sum()),
        "interior_terms_nonzero_action_rows": interior[~zero[interior]][:20].tolist(),
        "zero_action_rows": int(zero.sum()),
        "zero_action_nonterm_rows": int((zero & ~term).sum()),
    }
    return x, x_out, out_lines, card


def concat_check(seg_raw, names, concat_path: str):
    """Compare the concatenated raw segments with the curated CSV (all columns but termination)."""
    c = rc.read_numeric(concat_path)
    raw = np.concatenate(seg_raw, axis=0) if seg_raw else np.zeros((0, rc.NUM_COLS))
    res = {"concat": concat_path, "concat_sha256": rc.sha256_file(concat_path), "concat_rows": int(len(c)),
           "segments_rows": int(len(raw)), "warnings": []}
    if c.shape != raw.shape:
        res["warnings"].append(f"row count differs: concat {len(c)} vs segments {len(raw)}")
        res["ok"] = False
        return res
    ends = np.cumsum([len(s) for s in seg_raw]) - 1
    owner = np.repeat(np.arange(len(seg_raw)), [len(s) for s in seg_raw])
    starts = np.concatenate([[0], ends[:-1] + 1]) if len(ends) else np.zeros(0, dtype=int)
    body_c, body_r = c[:, :rc.TERM_COL], raw[:, :rc.TERM_COL]
    exact = (body_c == body_r).all(axis=1)
    close = np.isclose(body_c, body_r, rtol=1e-6, atol=1e-6).all(axis=1)
    res["rows_not_exact"] = int((~exact).sum())
    res["rows_not_close"] = int((~close).sum())
    res["max_abs_diff_non_term"] = float(np.abs(body_c - body_r).max()) if len(c) else 0.0
    if (~close).any():
        bad = np.where(~close)[0][:10]
        res["warnings"].append(f"{int((~close).sum())} rows differ in non-termination columns, first rows {bad.tolist()}")
    tc, tr = c[:, rc.TERM_COL] > 0.5, raw[:, rc.TERM_COL] > 0.5
    diff = np.where(tc != tr)[0]
    res["term_label_diffs"] = [
        {"row": int(r), "segment": names[owner[r]], "local_row": int(r - starts[owner[r]]),
         "segment_term": int(tr[r]), "concat_term": int(tc[r]), "is_segment_last_row": bool(r in set(ends.tolist()))}
        for r in diff[:50]]
    res["n_term_label_diffs"] = int(len(diff))
    res["concat_terms"] = int(tc.sum())
    res["segments_terms_raw"] = int(tr.sum())
    res["concat_term_at_segment_ends"] = [bool(tc[e]) for e in ends]
    non_end = [d for d in diff.tolist() if d not in set(ends.tolist())]
    if non_end:
        res["warnings"].append(f"{len(non_end)} termination label differences NOT at a segment end, first {non_end[:10]}")
    res["ok"] = not res["warnings"]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--segs", nargs="+", required=True, help="segment CSVs in concatenation order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concat", default=None, help="concatenated curated CSV for the cross-check")
    ap.add_argument("--zero_tol", type=float, default=0.0, help="max |action| that counts as an all-zero action")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = ap.parse_args(argv)

    names = [os.path.splitext(os.path.basename(p))[0] for p in args.segs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate segment names: {names}")
    rc.prepare_out_dir(args.out, args.force)

    cards, seg_raw, manifest = [], [], []
    for p, name in zip(args.segs, names):
        x, x_out, out_lines, card = process_segment(p, args.zero_tol)
        if len(x) == 0:
            raise SystemExit(f"{p}: empty")
        out_path = os.path.join(args.out, f"{name}.csv")
        rc.write_lines(out_path, out_lines)
        back = rc.read_numeric(out_path)
        if back.shape != x_out.shape or not np.array_equal(back, x_out):
            raise RuntimeError(f"{out_path}: re-read data differs from the intended output")
        card["name"] = name
        card["out_sha256"] = rc.sha256_file(out_path)
        cards.append(card)
        seg_raw.append(x)
        manifest.append(f"{name},{len(x)},{len(x)},1")
        print(f"[curated] {name}: rows={card['rows']} terms_raw={card['terms_raw']} last_row={card['last_row_status']} "
              f"seams_removed={card['seams_removed']} real_terms={card['real_terms']} "
              f"interior_terms={card['interior_terms']} (zero-action {card['interior_terms_zero_action']}) "
              f"zero_action_nonterm_rows={card['zero_action_nonterm_rows']}")
        if card["interior_terms_nonzero_action"]:
            print(f"[curated] WARNING {name}: {card['interior_terms_nonzero_action']} interior termination rows have a "
                  f"NON-zero action -> the reset-row hypothesis (fall row = zero action) does not hold for this file; "
                  f"the seam rule on the last row may then misclassify a real fall. Rows: "
                  f"{card['interior_terms_nonzero_action_rows']}")
    rc.write_lines(os.path.join(args.out, "manifest.csv"), ["name,steps,rows,num_envs"] + manifest)

    totals = {"files": len(cards), "rows": sum(c["rows"] for c in cards),
              "real_terms": sum(c["real_terms"] for c in cards),
              "seams_removed": sum(c["seams_removed"] for c in cards),
              "terms_raw": sum(c["terms_raw"] for c in cards),
              "interior_terms_nonzero_action": sum(c["interior_terms_nonzero_action"] for c in cards),
              "zero_action_nonterm_rows": sum(c["zero_action_nonterm_rows"] for c in cards)}
    out = {"kind": "curated_segments", "totals": totals, "files": cards, "provenance": rc.provenance(__file__, args)}
    if args.concat:
        chk = concat_check(seg_raw, names, args.concat)
        out["concat_check"] = chk
        print(f"[curated] concat check vs {args.concat}: concat_rows={chk['concat_rows']} segments_rows={chk['segments_rows']} "
              f"rows_not_exact={chk.get('rows_not_exact')} max_abs_diff={chk.get('max_abs_diff_non_term')} "
              f"term_label_diffs={chk.get('n_term_label_diffs')} concat_terms={chk.get('concat_terms')} "
              f"segments_terms_raw={chk.get('segments_terms_raw')}")
        for d in chk.get("term_label_diffs", [])[:20]:
            print(f"[curated]   term diff row {d['row']} ({d['segment']} local {d['local_row']}, last_row={d['is_segment_last_row']}): "
                  f"segment={d['segment_term']} concat={d['concat_term']}")
        for w in chk["warnings"]:
            print(f"[curated] WARNING concat check: {w}")
        if chk["ok"]:
            print("[curated] concat check OK (non-termination columns identical up to 1e-6; label diffs only at segment ends)")
    rc.write_json(os.path.join(args.out, "cards.json"), out)
    print(f"[curated] wrote {args.out}: files={totals['files']} rows={totals['rows']} real_terms={totals['real_terms']} "
          f"seams_removed={totals['seams_removed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
