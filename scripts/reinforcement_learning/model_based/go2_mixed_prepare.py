#!/usr/bin/env python3
"""A4 D2 (21 Nov 2026): the mixed-data manifest for one Go2 world model.

Sources: the nominal s2 round-0 recording and the D1 fitting batch (s2 under 0.30 x ground friction). Each is split
by the E22 rule first (10 % of whole episodes withheld, seed 101; go2_round0_prepare.prepare), so the round-0 split
is the same as A3's (checked against A3's withheld.json when given).

Output (fresh folder):
  r0/, shift/    the go2_round0_prepare outputs of each source (their own normalizers are NOT used)
  segments/      both training-segment sets, names prefixed r0_ / sh_, one manifest.csv (fitter layout)
  go2_norm.npz   the fitter's compute_normalizer on the combined training segments
  start_data/    both sources' start files (episode ends flagged), start_data.json (C: pooled imagination starts)
  manifest.json  counts of both sources, falls reaching fitting, hashes, the rule
The D1 probe (seed 911) and both withheld sets never enter segments, the normalizer or the start data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

MB = Path(__file__).resolve().parent
for p in (str(MB), str(MB / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import go2_round0_prepare as prep  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402


def _index(p):
    return int(p.stem.rsplit("_", 1)[1])


def prepare_mixed(r0_trace, shift_trace, out, frac=0.1, seed=101, a3_withheld=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.mkdir(parents=True)
    r0 = prep.prepare(r0_trace, out / "r0", frac, seed)
    sh = prep.prepare(shift_trace, out / "shift", frac, seed)
    if a3_withheld is not None and (out / "r0" / "withheld.json").read_text() != Path(a3_withheld).read_text():
        raise ValueError("the round-0 split differs from A3's withheld.json")
    seg = out / "segments"
    seg.mkdir()
    lines = ["name,steps,rows,num_envs"]
    for tag, src in (("r0", out / "r0" / "segments"), ("sh", out / "shift" / "segments")):
        for line in (src / "manifest.csv").read_text().splitlines()[1:]:
            name, steps, rows, envs = line.split(",")
            shutil.copyfile(src / f"{name}.csv", seg / f"{tag}_{name}.csv")
            lines.append(f"{tag}_{name},{steps},{rows},{envs}")
    (seg / "manifest.csv").write_text("\n".join(lines) + "\n")
    start = out / "start_data"
    start.mkdir()
    files, total, min_rows, episodes = 0, 0, None, 0
    for src in (out / "r0" / "start_data", out / "shift" / "start_data"):
        episodes += json.loads((src / "start_data.json").read_text())["episodes"]
        for f in sorted(src.glob("state_action_data_*.csv"), key=_index):
            shutil.copyfile(f, start / f"state_action_data_{files}.csv")
            n = sum(1 for _ in f.open())
            files, total = files + 1, total + n
            min_rows = n if min_rows is None else min(min_rows, n)
    (start / "start_data.json").write_text(json.dumps(dict(
        rows=total, files=files, min_file_rows=int(min_rows), episodes=episodes,
        layout="next_state 45, action 12, contact 8, episode-end flag 1 (start sampler only)",
        sources="round-0 nominal + D1 shift (0.30 friction), pooled"), indent=1))
    from fit_world_model_go2_pilot_segments import compute_normalizer, read_segments
    segs, _, _ = read_segments(seg, prep.STATE, prep.ACTION, 0, prep.CONTACT, prep.TERM)
    sm, ss, am, as_ = (t.reshape(-1).numpy() for t in compute_normalizer(segs, prep.STATE, prep.ACTION))
    with (out / "go2_norm.npz").open("xb") as f:
        np.savez(f, state_mean=sm, state_std=ss, action_mean=am, action_std=as_)
    c0, c1 = r0["counts"], sh["counts"]
    result = dict(
        rule="A4 D2: E22 split (fraction %s, seed %s) of each source before fitting; segments and start data pooled; "
             "normalizer from the pooled training segments" % (frac, seed),
        sources=dict(round0=dict(path=str(r0_trace), sha256=r0["source_sha256"], counts=c0),
                     shift=dict(path=str(shift_trace), sha256=sh["source_sha256"], counts=c1)),
        segments=len(lines) - 1, fit_rows=c0["fit_rows"] + c1["fit_rows"],
        falls_reaching_fitting=c0["falls_reaching_fitting"] + c1["falls_reaching_fitting"],
        terminal_positive_rows_reaching_fitting=(c0["terminal_positive_rows_reaching_fitting"]
                                                 + c1["terminal_positive_rows_reaching_fitting"]),
        start_rows=total, start_files=files,
        outputs=dict(segments_manifest_sha256=sha256(seg / "manifest.csv"), go2_norm_sha256=sha256(out / "go2_norm.npz"),
                     start_data_json_sha256=sha256(start / "start_data.json"),
                     r0_withheld_sha256=sha256(out / "r0" / "withheld.json"),
                     shift_withheld_sha256=sha256(out / "shift" / "withheld.json")),
        code_sha256=sha256(__file__))
    (out / "manifest.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round0", type=Path, required=True)
    p.add_argument("--shift", type=Path, required=True, help="the D1 fitting batch")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--a3-withheld", type=Path, default=None, help="A3's data/withheld.json (the split must be equal)")
    a = p.parse_args(argv)
    r = prepare_mixed(a.round0, a.shift, a.out, a3_withheld=a.a3_withheld)
    s0, s1 = r["sources"]["round0"]["counts"], r["sources"]["shift"]["counts"]
    print(f"[d2] round-0: {s0['training']} training episodes ({s0['training_falls']} falls); shift: {s1['training']} "
          f"training episodes ({s1['training_falls']} falls), withheld {s1['withheld']} ({s1['withheld_falls']} falls)")
    print(f"[d2] {r['segments']} segments, {r['fit_rows']} rows; falls reaching fitting {r['falls_reaching_fitting']} "
          f"({r['terminal_positive_rows_reaching_fitting']} terminal positive rows); start rows {r['start_rows']}")
    print(f"[d2] normalizer sha256 {r['outputs']['go2_norm_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
