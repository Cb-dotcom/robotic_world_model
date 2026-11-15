#!/usr/bin/env python3
"""Plan v6 C3(2), fall-data dose-response: training-data folders that keep 0/25/50/100 % of the fall episodes.

Source: an e1_prepare_data.py segments folder (one CSV per episode, manifest.csv). A segment is a fall
episode if its last row has termination > 0.5 (e1_prepare_data writes the true terminal state there).
Only fall episodes whose name starts with --fall_tag (default baseline1701) are subsampled. Every other
segment is kept in every dose: non-fall episodes of the falling policy and all segments of other tags
("reference kept in the mix"; a fall episode of another tag is kept and counted).

Kept falls are nested: one permutation of the sorted fall names with numpy default_rng(--seed); dose d
keeps the first round(d / 100 * n) of it, so 25 % is inside 50 % is inside 100 %. A dropped fall episode
is dropped whole (its walking steps too); the cards record the rows of every dose.

Output (fresh folder): dose<ddd>/manifest.csv + one symlink per kept segment (absolute path of the
source CSV; the fitter records realpath and sha256), dose<ddd>/cards.json, and dose_cards.json.
CPU only, no torch.

  make_fall_dose.py --src <e1v>/data/segments --out <dir> --doses 0,25,50,100 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def last_termination(path):
    """Termination value (last column) of the last non-empty line of a segment CSV."""
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        block = min(size, 65536)
        stream.seek(size - block)
        lines = [line for line in stream.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty file")
    return float(lines[-1].split(b",")[-1])


def read_manifest(src):
    with open(Path(src) / "manifest.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) < {"name", "steps", "rows"}:
        raise ValueError(f"{src}/manifest.csv needs columns name,steps,rows[,num_envs]")
    names = [row["name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("duplicate segment names in manifest.csv")
    return rows


def plan_doses(rows, src, fall_tag, doses, seed):
    """Return (falls_in_tag, other_falls, kept_by_dose) with kept_by_dose[d] = set of kept names."""
    falls, other_falls = [], []
    for row in rows:
        if last_termination(Path(src) / f"{row['name']}.csv") > 0.5:
            (falls if row["name"].startswith(fall_tag) else other_falls).append(row["name"])
    order = [str(n) for n in np.random.default_rng(seed).permutation(sorted(falls))]
    dropped_all = set(falls)
    kept = {}
    for d in doses:
        keep_falls = set(order[: int(round(d / 100.0 * len(order)))])
        kept[d] = {row["name"] for row in rows if row["name"] not in dropped_all or row["name"] in keep_falls}
    return falls, other_falls, order, kept


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="e1_prepare_data.py segments folder")
    p.add_argument("--out", type=Path, required=True, help="fresh output folder")
    p.add_argument("--doses", default="0,25,50,100", help="percent of fall episodes kept, comma separated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fall_tag", default="baseline1701", help="name prefix of the subsampled fall episodes")
    args = p.parse_args(argv)
    if args.out.exists():
        p.error("--out must not exist")
    doses = [int(x) for x in args.doses.split(",")]
    if any(d < 0 or d > 100 for d in doses) or len(set(doses)) != len(doses):
        p.error("--doses must be distinct integers in 0..100")
    src = args.src.resolve()
    rows = read_manifest(src)
    falls, other_falls, order, kept = plan_doses(rows, src, args.fall_tag, doses, args.seed)
    if not falls:
        raise SystemExit(f"no fall episode with prefix {args.fall_tag!r} in {src}")
    manifest_sha = hashlib.sha256((src / "manifest.csv").read_bytes()).hexdigest()
    header = list(rows[0].keys())
    cards = {"src": str(src), "src_manifest_sha256": manifest_sha, "fall_tag": args.fall_tag, "seed": args.seed,
             "falls_in_tag": len(falls), "falls_other_tags_kept_always": other_falls,
             "fall_order": order, "segments_total": len(rows), "doses": {}}
    args.out.mkdir(parents=True)
    for d in doses:
        folder = args.out / f"dose{d:03d}"
        folder.mkdir()
        kept_rows = [row for row in rows if row["name"] in kept[d]]
        with open(folder / "manifest.csv", "x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(kept_rows)
        for row in kept_rows:
            os.symlink(src / f"{row['name']}.csv", folder / f"{row['name']}.csv")
        kept_falls = [n for n in order if n in kept[d]]
        card = {"dose_percent": d, "segments": len(kept_rows), "rows": int(sum(int(r["rows"]) for r in kept_rows)),
                "falls_kept": len(kept_falls), "falls_dropped": len(falls) - len(kept_falls),
                "real_terms_expected": len(kept_falls) + len(other_falls), "kept_fall_names": kept_falls}
        (folder / "cards.json").write_text(json.dumps(card, indent=2) + "\n")
        cards["doses"][str(d)] = {k: v for k, v in card.items() if k != "kept_fall_names"}
        print(f"[dose] {d:3d} %: {card['segments']} segments, {card['rows']} rows, "
              f"falls kept {card['falls_kept']}/{len(falls)}, real terms expected {card['real_terms_expected']}")
    (args.out / "dose_cards.json").write_text(json.dumps(cards, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
