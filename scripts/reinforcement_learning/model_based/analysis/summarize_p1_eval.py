#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compact comparison table over eval_signals.py result directories.

Usage:  python analysis/summarize_p1_eval.py RESULTS_DIR [--signals epi,term,knn10] [--sets pre5,fall]

Reads every RESULTS_DIR/*/summary.json (one per WM x trace run). For each negative set
(scorer = the scorer's sample, clean = clean negatives) prints one table:
rows = run, columns = tie-aware AUC [95% env-bootstrap CI] for each signal x positive
set, plus the scorer-AUC of epi(pre5) (the number to compare with the scorer headline).
Writes the same text to RESULTS_DIR/comparison.txt and a long-format comparison.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import List, Sequence


def _cell(m) -> str:
    if not m or m.get("auc") is None:
        return "n/a"
    lo, hi = m.get("ci_lo"), m.get("ci_hi")
    ci = f"[{lo:.3f},{hi:.3f}]" if lo is not None and hi is not None else "[n/a]"
    return f"{m['auc']:.3f} {ci}"


def load_runs(root: str):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        with open(p) as fh:
            runs.append((os.path.basename(os.path.dirname(p)), json.load(fh)))
    return runs


def summarize(root: str, signals: Sequence[str] = ("epi", "term_logit", "knn10"),
              sets: Sequence[str] = ("pre5", "pre1", "lead10", "fall")) -> str:
    runs = load_runs(root)
    if not runs:
        return f"no */summary.json under {root}"
    out: List[str] = []
    cols = [(s, p) for p in sets for s in signals]
    w = 21
    namew = max(12, max(len(n) for n, _ in runs))
    for neg in ("scorer", "clean"):
        out.append(f"\n=== negatives = {neg}: tie-aware AUC [95% env-bootstrap CI] ===")
        hdr = f"{'run':{namew}s} {'n_fall':>6s} {'n_pre5':>6s} {'n_neg':>6s}"
        hdr += "".join(f" | {s + ' ' + p:{w}s}" for s, p in cols) + " | epi pre5 aucS"
        out.append(hdr)
        for name, js in runs:
            c = js.get("counts", {})
            n_neg = c.get(f"n_neg_{neg}", "")
            line = f"{name:{namew}s} {c.get('n_fall', ''):>6} {c.get('n_pre5', ''):>6} {n_neg:>6}"
            mets = js.get("metrics", {})
            for s, p in cols:
                line += f" | {_cell(mets.get(s, {}).get(p, {}).get(neg)):{w}s}"
            aucs = mets.get("epi", {}).get("pre5", {}).get(neg, {}).get("auc_scorer")
            line += f" | {aucs:.3f}" if aucs is not None else " | n/a"
            out.append(line)
    out.append("\nboot_skipped (pre5, scorer negs): " + ", ".join(
        f"{n}:{js['metrics'][s]['pre5']['scorer']['boot_skipped']}"
        for n, js in runs for s in signals[:1] if s in js.get("metrics", {})))
    out.append("note: 'fall' is contaminated by the reset row (spawn state + zero action); use pre5/pre1/lead10.")
    return "\n".join(out)


def write_csv(root: str, path: str) -> None:
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["run", "signal", "positives", "negatives", "auc", "ci_lo", "ci_hi", "auc_scorer", "ratio",
                     "n_pos", "n_neg", "boot_skipped"])
        for name, js in load_runs(root):
            for s, by_p in js.get("metrics", {}).items():
                for p, by_n in by_p.items():
                    for n, m in by_n.items():
                        wr.writerow([name, s, p, n, m.get("auc"), m.get("ci_lo"), m.get("ci_hi"), m.get("auc_scorer"),
                                     m.get("ratio"), m.get("n_pos"), m.get("n_neg"), m.get("boot_skipped")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--signals", default="epi,term_logit,knn10")
    ap.add_argument("--sets", default="pre5,pre1,lead10,fall")
    a = ap.parse_args()
    txt = summarize(a.root, a.signals.split(","), a.sets.split(","))
    print(txt)
    with open(os.path.join(a.root, "comparison.txt"), "w") as fh:
        fh.write(txt + "\n")
    write_csv(a.root, os.path.join(a.root, "comparison.csv"))


if __name__ == "__main__":
    main()
