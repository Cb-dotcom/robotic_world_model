#!/usr/bin/env python3
"""A4 D8: verify the saved reference evaluations (unchanged D7 and s2 at 0.30 friction) before they are used. Read-only.

For each schedule asked (--cmd, repeatable; 10 is the documented one) and each policy (D7, S2) and seed (42, 43):
  cell       go2_d8_score.validate_cell: task, envs, steps, seed, std, command schedule, pushes, verified friction
             factor, complete per-env episode lengths, exposure, falls in the CSV = falls in the summary
  checkpoint the summary's checkpoint path exists and its sha256 starts with the expected prefix (D7 policy, s2)
  readback   eval.log next to the cell holds the simulator friction read-back line
  totals     the pooled falls equal the documented numbers (--expect-falls D7=306 --expect-falls S2=219 at 10 s)
Every file's sha256 is written into references.json. A schedule whose cells do not all exist is listed as MISSING
cell by cell (it is not an error: it tells which references a D8 evaluation must produce itself). Any other failure
exits 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import go2_d8_score as sc  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

READBACK = "[friction] verified in the simulator"
TAGS = ("D7", "S2")


def verify(ref, friction, cmds, expect_ckpt, expect_falls, envs=sc.ENVS, steps=sc.STEPS):
    ref = Path(ref)
    res = dict(ref=str(ref), friction=friction, schedules={}, problems=[])
    for cmd in cmds:
        cells = {t: [ref / f"{t}_cmd{cmd}_s{s}" for s in sc.SEEDS] for t in TAGS}
        missing = [str(c) for v in cells.values() for c in v if not (c / "episodes_summary.json").is_file()]
        if missing:
            res["schedules"][str(cmd)] = dict(status="MISSING", missing=missing)
            continue
        sched = dict(status="OK", policies={})
        for t, folders in cells.items():
            rows, falls = [], 0
            for s, folder in zip(sc.SEEDS, folders):
                try:
                    c = sc.validate_cell(folder, cmd, s, friction, envs, steps)
                except sc.CellInvalid as e:
                    res["problems"].append(str(e))
                    continue
                summ = json.loads((folder / "episodes_summary.json").read_text())
                ck = Path(summ.get("checkpoint", ""))
                ck_sha = sha256(ck) if ck.is_file() else None
                if ck_sha is None:
                    res["problems"].append(f"{folder}: checkpoint {ck} not found")
                elif t in expect_ckpt and not ck_sha.startswith(expect_ckpt[t]):
                    res["problems"].append(f"{folder}: checkpoint sha256 {ck_sha[:12]} != expected {expect_ckpt[t]}")
                log = folder / "eval.log"
                readback = log.is_file() and READBACK in log.read_text(errors="replace")
                if not readback:
                    res["problems"].append(f"{folder}: no friction read-back line in eval.log")
                falls += c["falls"]
                rows.append(dict(seed=s, falls=c["falls"], exposure_s=c["exposure_s"], exposure_source=c["exposure_source"],
                                 speed_ratio=c["speed_ratio"], track_per_step=c["track_per_step"],
                                 checkpoint=str(ck), checkpoint_sha256=ck_sha, friction_readback=bool(readback),
                                 episodes_summary_sha256=c["episodes_summary_sha256"],
                                 episodes_csv_sha256=c["episodes_csv_sha256"],
                                 eval_log_sha256=sha256(log) if log.is_file() else None))
            want = expect_falls.get((t, cmd))
            if want is not None and falls != want:
                res["problems"].append(f"{t} {cmd} s: {falls} falls, documented {want}")
            sched["policies"][t] = dict(falls=falls, per_seed=rows, documented_falls=want)
        res["schedules"][str(cmd)] = sched
    res["ok"] = not res["problems"]
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", type=Path, required=True, help="the saved D7 evaluation folder (a4_d7_1123b/eval)")
    p.add_argument("--friction", type=float, default=0.30)
    p.add_argument("--cmd", type=int, action="append", default=None, help="schedules to check (default 10 and 30)")
    p.add_argument("--expect-ckpt", action="append", default=[], metavar="TAG=SHA_PREFIX")
    p.add_argument("--expect-falls", action="append", default=[], metavar="TAG:CMD=N")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    ck = dict(s.split("=", 1) for s in a.expect_ckpt)
    falls = {}
    for s in a.expect_falls:
        k, v = s.split("=", 1)
        tag, cmd = k.split(":")
        falls[(tag, int(cmd))] = int(v)
    res = verify(a.ref, a.friction, a.cmd or [10, 30], ck, falls)
    with a.out.open("x") as f:
        json.dump(res, f, indent=1)
    for cmd, s in res["schedules"].items():
        if s["status"] == "MISSING":
            print(f"[d8-refs] {cmd} s commands: MISSING {len(s['missing'])} cells: " + ", ".join(Path(m).name for m in s["missing"]))
        else:
            print(f"[d8-refs] {cmd} s commands: " + "; ".join(f"{t} {v['falls']} falls" for t, v in s["policies"].items()))
    for pr in res["problems"]:
        print(f"[d8-refs] PROBLEM: {pr}")
    print(f"[d8-refs] {'OK' if res['ok'] else 'FAILED'}")
    return 0 if res["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
