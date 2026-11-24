#!/usr/bin/env python3
"""A4 D8 scorer (draft 24 Nov 2026; rules open until Charbel freezes them in A4 §3): frozen-model arm F8 vs
updated-model arm U8, both from D7, with unchanged D7 and s2 as references. Read-only, CPU.

Cells: <dir>/<TAG>_cmd<C>_s<seed>/{episodes_summary.json, episodes.csv} as eval_episode_stats.py writes them.
F8 and U8 come from the D8 run's eval folder; D7 and S2 from their saved evaluations (D7's run, a4_d7_1123b).

Every cell is validated before any number is used (a failure stops the scorer; exit 3):
  protocol   task, num_envs, num_steps, seed (= the folder's), std, cmd_resample_s (= the folder's), push enabled,
             friction shift verified at the expected factor; equal across all policies of a schedule;
  complete   per env, the episode lengths in episodes.csv add up to num_steps (every step of every env is in exactly
             one episode, censored ones included), so the cell is not a partial run;
  exposure   expected = num_envs x num_steps x step_dt, and the expected budget (--envs, --steps) must match it; the
             summary's exposure_s, when present, must equal it (rel 1e-9); a cell without exposure_s uses the validated
             value and is flagged "validated metadata";
  falls      falls counted in episodes.csv (fall and not censored) = the summary's falls.
The historical scorers (go2_g0_gate.py, go2_d3_gate.py with its fixed 6,400 s per seed) are not changed.

Per policy (pooled over the seeds): falls, exposure, falls per 1000 s with the exact Poisson 95 % CI;
speed_ratio and track_per_step as the D3 gate reads them (mean of the per-seed summary values); falls per km
(measured distance, go2_d3_fall_timing.stats); startup: falls within 40 rows of a reset (draft window) and within the
first second (round(1 / step_dt) rows, the window of the D7 report), per reset and per second at risk
(go2_startup_hazard.hazards).

Readout components (always written): movement bars (D3's gate: speed_ratio >= 0.80, track_per_step >= s2's - 0.030);
A: U8 falls <= 0.80 x F8, F8 falls <= 0.80 x U8, movement kept per arm; B1: U8 falls <= 0.80 x D7 falls;
B2: D3's gate on U8 against s2 (falls <= 1.25 x s2 + 3 and both movement bars).
Verdicts only under a named rule set and the dated record that froze it (--rules NAME --rule-record TEXT):
  draft-2026-11-24     the draft text as written (A: an arm that alone keeps movement can be the better arm;
                       progression mentions B2 only). Note: with zero falls in both arms both A conditions hold;
                       this set reports that case as "AMBIGUOUS (both conditions hold)".
  proposed-2026-11-25  Charbel's proposed reading: a fall comparison needs BOTH arms to keep movement, otherwise A =
                       "MOVEMENT NOT KEPT (<arm>)"; equal falls (zero in both included) = NO CLEAR DIFFERENCE;
                       progression to a scored-study proposal needs A = UPDATED BETTER, B1 and B2.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fall_rate import falls_per_1000s  # noqa: E402
from go2_d3_fall_timing import read_cell, stats as timing_stats  # noqa: E402
from go2_startup_hazard import hazards  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

SEEDS = (42, 43)
TASK = "Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0"
ENVS, STEPS, STD = 32, 10000, 0.5
SPEED_MIN, TRACK_MARGIN, FALL_FACTOR, FALL_SLACK = 0.80, 0.030, 1.25, 3
A_FACTOR, B1_FACTOR = 0.80, 0.80
STARTUP_ROWS = 40
RULE_SETS = ("draft-2026-11-24", "proposed-2026-11-25")
POLICIES = ("F8", "U8", "D7", "S2")


class CellInvalid(ValueError):
    pass


def validate_cell(folder, cmd, seed, friction, envs=ENVS, steps=STEPS, std=STD, task=TASK):
    folder = Path(folder)
    if not (folder / "episodes_summary.json").is_file() or not (folder / "episodes.csv").is_file():
        raise CellInvalid(f"{folder}: missing episodes_summary.json or episodes.csv")
    summ, ep = read_cell(folder)
    want = dict(task=task, num_envs=envs, num_steps=steps, seed=seed, std=std, cmd_resample_s=cmd)
    for k, v in want.items():
        got = summ.get(k)
        same = (got is not None and float(got) == float(v)) if k != "task" else got == v
        if not same:
            raise CellInvalid(f"{folder}: {k} = {got!r}, expected {v!r}")
    if not (summ.get("push_robot") or {}).get("enabled", False):
        raise CellInvalid(f"{folder}: push_robot not enabled (the G6 protocol keeps pushes)")
    fs = summ.get("friction_shift") or {}
    if not fs.get("verified") or abs(float(fs.get("factor", math.nan)) - friction) > 1e-9:
        raise CellInvalid(f"{folder}: friction shift {fs} is not a verified x{friction}")
    with open(folder / "episodes.csv", newline="") as f:
        envcol = np.array([int(r["env"]) for r in csv.DictReader(f)])
    if envcol.size == 0 or set(np.unique(envcol)) != set(range(envs)):
        raise CellInvalid(f"{folder}: episodes.csv does not cover envs 0..{envs - 1}")
    per_env = np.bincount(envcol, weights=ep["length"], minlength=envs)
    if not np.all(per_env == steps):
        bad = np.flatnonzero(per_env != steps)[:5].tolist()
        raise CellInvalid(f"{folder}: incomplete cell (env step totals differ from {steps} for envs {bad})")
    dt = float(summ["step_dt"])
    expected = envs * steps * dt
    inner = summ["summary"]
    if "exposure_s" in inner:
        if abs(float(inner["exposure_s"]) - expected) > 1e-9 * expected:
            raise CellInvalid(f"{folder}: summary exposure_s {inner['exposure_s']} != {envs} x {steps} x {dt}")
        source = "summary.exposure_s (checked)"
    else:
        source = "validated metadata (num_envs x num_steps x step_dt; episode lengths complete)"
    falls = int((ep["fall"] & ~ep["censored"]).sum())
    if falls != int(inner["falls"]):
        raise CellInvalid(f"{folder}: {falls} falls in episodes.csv vs {inner['falls']} in the summary")
    inner = dict(inner, exposure_s=expected)
    t = timing_stats(dict(summ, summary=inner), ep)
    return dict(folder=str(folder), seed=seed, cmd=cmd, falls=falls, exposure_s=expected, exposure_source=source,
                step_dt=dt, speed_ratio=float(inner["speed_ratio"]), track_per_step=float(inner["track_per_step"]),
                distance_m=t["distance_m"], friction=float(fs["factor"]),
                episodes_summary_sha256=sha256(folder / "episodes_summary.json"),
                episodes_csv_sha256=sha256(folder / "episodes.csv"),
                lengths=ep["length"], fall_mask=ep["fall"] & ~ep["censored"])


def policy(cells):
    dts = {c["step_dt"] for c in cells}
    if len(dts) != 1:
        raise CellInvalid("step_dt differs between seeds")
    dt = dts.pop()
    falls, expo = sum(c["falls"] for c in cells), sum(c["exposure_s"] for c in cells)
    rate, lo, hi = falls_per_1000s(falls, expo)
    L = np.concatenate([c["lengths"] for c in cells]).astype(np.int64)
    F = np.concatenate([c["fall_mask"] for c in cells])
    hz = hazards(L, F, dt)
    first_s = int(round(1.0 / dt))
    resets = len(L)
    dist = sum(c["distance_m"] for c in cells)
    startup = {}
    for name, rows in (("within_40_rows", STARTUP_ROWS), ("first_second", first_s)):
        n = int((F & (L <= rows)).sum())
        at_risk_s = float(np.minimum(L, rows).sum() * dt)
        r, rlo, rhi = falls_per_1000s(n, at_risk_s)
        startup[name] = dict(rows=rows, falls=n, per_reset=n / resets, hazard_per_s=[r / 1000, rlo / 1000, rhi / 1000],
                             exposure_at_risk_s=at_risk_s)
    return dict(falls=falls, exposure_s=expo, rate_per_1000s=rate, ci95=[lo, hi], resets=resets,
                speed_ratio=float(np.mean([c["speed_ratio"] for c in cells])),
                track_per_step=float(np.mean([c["track_per_step"] for c in cells])),
                distance_m=dist, falls_per_km=1000.0 * falls / dist if dist > 0 else float("nan"),
                startup=startup, hazard_bins=hz["bins"],
                per_seed=[{k: v for k, v in c.items() if k not in ("lengths", "fall_mask")} for c in cells])


def components(p):
    s2_track = p["S2"]["track_per_step"]
    move = {t: dict(speed=p[t]["speed_ratio"] >= SPEED_MIN, tracking=p[t]["track_per_step"] >= s2_track - TRACK_MARGIN)
            for t in ("F8", "U8", "D7")}
    for t in move:
        move[t]["kept"] = move[t]["speed"] and move[t]["tracking"]
    u, f, d7, s2 = p["U8"]["falls"], p["F8"]["falls"], p["D7"]["falls"], p["S2"]["falls"]
    return dict(bars=dict(speed_min=SPEED_MIN, tracking_bar=s2_track - TRACK_MARGIN, a_factor=A_FACTOR,
                          b1_bar=B1_FACTOR * d7, b2_fall_bar=FALL_FACTOR * s2 + FALL_SLACK),
                movement=move,
                A=dict(u_le_08f=u <= A_FACTOR * f, f_le_08u=f <= A_FACTOR * u, ratio_u_over_f=(u / f) if f else float("nan"),
                       equal_falls=u == f),
                B1=u <= B1_FACTOR * d7, B2=dict(falls=u <= FALL_FACTOR * s2 + FALL_SLACK, movement=move["U8"]["kept"]),
                falls=dict(F8=f, U8=u, D7=d7, S2=s2))


def readout(c, rules):
    if rules not in RULE_SETS:
        raise ValueError(f"rules must be one of {RULE_SETS}")
    A, mv = c["A"], c["movement"]
    b2 = c["B2"]["falls"] and c["B2"]["movement"]
    if rules == "draft-2026-11-24":
        better = A["u_le_08f"] and mv["U8"]["kept"]
        worse = A["f_le_08u"] and mv["F8"]["kept"]
        a = ("AMBIGUOUS (both conditions hold)" if better and worse else
             "UPDATED BETTER" if better else "UPDATED WORSE" if worse else "NO CLEAR DIFFERENCE")
        follows = ("credible update; scored study needs A5" if a == "UPDATED BETTER" and b2 else
                   "development result: adaptation helps, pipeline not ready" if a == "UPDATED BETTER" else
                   "Go2 stays development evidence")
    else:
        if not (mv["U8"]["kept"] and mv["F8"]["kept"]):
            missing = [t for t in ("F8", "U8") if not mv[t]["kept"]]
            a = f"MOVEMENT NOT KEPT ({', '.join(missing)})"
        elif A["equal_falls"]:
            a = "NO CLEAR DIFFERENCE"
        elif A["u_le_08f"]:
            a = "UPDATED BETTER"
        elif A["f_le_08u"]:
            a = "UPDATED WORSE"
        else:
            a = "NO CLEAR DIFFERENCE"
        follows = ("propose a scored Go2 study (A5) for review" if a == "UPDATED BETTER" and c["B1"] and b2 else
                   "development result; no scored-study proposal" if a == "UPDATED BETTER" else
                   "Go2 stays development evidence")
    return dict(rules=rules, A=a, B1=bool(c["B1"]), B2=bool(b2), follows=follows)


def score(new_dir, ref_dir, friction, cmd=10, envs=ENVS, steps=STEPS, ref_dirs=None):
    """new_dir holds F8 and U8; ref_dir holds D7 and S2 (ref_dirs={tag: dir} overrides per tag)."""
    where = {"F8": new_dir, "U8": new_dir, "D7": ref_dir, "S2": ref_dir}
    where.update(ref_dirs or {})
    cells = {t: [validate_cell(Path(where[t]) / f"{t}_cmd{cmd}_s{s}", cmd, s, friction, envs, steps) for s in SEEDS]
             for t in POLICIES}
    p = {t: policy(v) for t, v in cells.items()}
    if len({round(p[t]["exposure_s"], 6) for t in p}) != 1:
        raise CellInvalid("exposure differs between policies")
    return p


def missing_cells(new_dir, ref_dir, cmd, ref_dirs=None):
    where = {"F8": new_dir, "U8": new_dir, "D7": ref_dir, "S2": ref_dir}
    where.update(ref_dirs or {})
    return [str(Path(where[t]) / f"{t}_cmd{cmd}_s{s}") for t in POLICIES for s in SEEDS
            if not (Path(where[t]) / f"{t}_cmd{cmd}_s{s}" / "episodes_summary.json").is_file()]


def text(p, c, r, cmd):
    L = [f"A4 D8 (development evidence; one PPO seed): {cmd} s commands, seeds 42 + 43, "
         f"{p['S2']['exposure_s']:.0f} s per policy"]
    for t in POLICIES:
        x = p[t]
        st = x["startup"]
        L.append(f"  {t}: falls {x['falls']} ({x['rate_per_1000s']:.2f} per 1000 s [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]); "
                 f"speed_ratio {x['speed_ratio']:.3f}; track_per_step {x['track_per_step']:.4f}; "
                 f"{x['falls_per_km']:.1f} falls per km; within 40 rows {st['within_40_rows']['falls']} "
                 f"({st['within_40_rows']['per_reset']:.4f} per reset); first second {st['first_second']['falls']} "
                 f"({st['first_second']['per_reset']:.4f} per reset)")
    b = c["bars"]
    L.append(f"  movement bars: speed >= {b['speed_min']}, track >= {b['tracking_bar']:.4f}: "
             + ", ".join(f"{t} {'kept' if v['kept'] else 'NOT kept'}" for t, v in c["movement"].items()))
    L.append(f"  A: U8/F8 falls {c['A']['ratio_u_over_f']:.3f}; U8 <= 0.80 F8 {c['A']['u_le_08f']}; F8 <= 0.80 U8 "
             f"{c['A']['f_le_08u']}")
    L.append(f"  B1: U8 <= 0.80 x D7 ({b['b1_bar']:.1f}) {c['B1']}; B2: falls <= {b['b2_fall_bar']:.2f} "
             f"{c['B2']['falls']}, movement {c['B2']['movement']}")
    if r:
        L.append(f"READOUT ({r['rules']}; record {r.get('record')}): A {r['A']}; B1 {r['B1']}; B2 {r['B2']} -> {r['follows']}")
    else:
        L.append("READOUT: not made (no frozen rule record); components only")
    return "\n".join(L) + "\n"


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--new", type=Path, required=True, help="the D8 eval folder (F8_*, U8_*)")
    pa.add_argument("--ref", type=Path, required=True, help="folder with the saved D7_* and S2_* cells")
    pa.add_argument("--friction", type=float, required=True)
    pa.add_argument("--cmd", type=int, default=10, help="primary: 10")
    pa.add_argument("--also-cmd", type=int, action="append", default=[], help="reported only (e.g. 30)")
    pa.add_argument("--envs", type=int, default=ENVS)
    pa.add_argument("--steps", type=int, default=STEPS)
    pa.add_argument("--rules", choices=RULE_SETS, default=None)
    pa.add_argument("--rule-record", default=None, help="the dated A4 §3 / decisions-log entry that froze the rules")
    pa.add_argument("--out", type=Path, required=True)
    a = pa.parse_args(argv)
    if (a.rules is None) != (a.rule_record is None):
        pa.error("--rules and --rule-record go together")
    try:
        p = score(a.new, a.ref, a.friction, a.cmd, a.envs, a.steps)
    except CellInvalid as e:
        print(f"[d8-score] INVALID CELL: {e}", flush=True)
        return 3
    c = components(p)
    r = dict(readout(c, a.rules), record=a.rule_record) if a.rules else None
    extra = {}
    for cmd in a.also_cmd:
        miss = missing_cells(a.new, a.ref, cmd)
        if miss:
            extra[str(cmd)] = dict(reported=False, missing=miss)
            continue
        try:
            extra[str(cmd)] = dict(reported=True, policies=score(a.new, a.ref, a.friction, cmd, a.envs, a.steps))
        except CellInvalid as e:
            extra[str(cmd)] = dict(reported=False, invalid=str(e))
    out = dict(protocol="A4 D8 scorer (draft 24 Nov 2026)", cmd=a.cmd, friction=a.friction, envs=a.envs, steps=a.steps,
               policies=p, components=c, readout=r, also=extra, analysis_sha256=sha256(__file__))
    with a.out.open("x") as f:
        json.dump(out, f, indent=1, default=float)
    t = text(p, c, r, a.cmd)
    for cmd, v in extra.items():
        t += (f"  {cmd} s commands: reported\n" if v["reported"] else
              f"  {cmd} s commands: NOT reported ({'missing ' + str(len(v['missing'])) + ' cells' if 'missing' in v else v['invalid']})\n")
    a.out.with_suffix(".txt").write_text(t)
    print(t, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
