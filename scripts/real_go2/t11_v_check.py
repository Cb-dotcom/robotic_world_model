# SPDX-License-Identifier: BSD-3-Clause
"""T11 velocity check (V gate input): taped straight walks vs the robot's speed sources.

For each run (one bag + tape readings of one robot point at start and end; 18 Nov rule:
the front-left foot):
  1. Find the walk: the longest segment where the /sportmodestate speed is above
     --move-thresh (short dips below it are merged).
  2. Take the robot pose while it stands still before and after the walk
     (median over --still-s seconds).
  3. Compare the displacement of each source with the tape displacement:
       sport_pos  /sportmodestate.position (world, mcf odometry)
       sport_int  /sportmodestate.velocity (BASE frame) turned by yaw and integrated
       <odom>     every nav_msgs/Odometry topic with messages (robot_odom, Point-LIO, ...)
     ratio = source displacement / tape displacement. 1.000 = exact.
  4. Report the commanded speed (/api/sport/request Move x) and the yaw drift.

The ratio does not depend on the walk timing. Speeds (m/s) use the walk duration
found in step 1 and are for orientation only.

Official V-gate verdict (decisions-log 18 Nov 2026, both V rows), from the Point-LIO source only.
One run = one listed row = one walk (a whole bag, or one leg of it) with its own tape readings and
its own commanded speed (median /api/sport/request Move x over the walk; a backward leg commands -x).
  PASS / FAIL    the listed rows are exactly the prescribed set: 3 runs at commanded 0.30 and 3 at
                 0.45 m/s, each row in exactly one speed group, each bag in one speed group, no walk
                 listed twice, every tape distance on the 5 m line, Point-LIO in every run, bar 5 %;
                 PASS = every run has |ratio - 1| <= 5 % (inclusive).
  INVALID        a listed run is missing or errored, Point-LIO has no finite ratio in a run, or a run
                 has no readable commanded speed (data of a listed run are broken).
  DIAGNOSTIC ONLY: <reasons>   the data are readable but the set or the settings are not the rule
                 (too few or extra runs, another speed, ambiguous rows, no Point-LIO in any bag,
                 another line length, --max-rel-err, --speed-tol-mps or --line-tol-m changed).
                 Ratios are printed, never PASS.
The speed and line tolerances are PROPOSED (not in the 18 Nov rule) until a dated decisions-log row.
/sportmodestate and /utlidar/robot_odom are one Unitree estimate: reported, never the gate.
Exit 0 = official verdict (PASS or FAIL, read the V GATE line), 1 = INVALID (as before),
2 = no official verdict (DIAGNOSTIC ONLY).

Runs are given as a CSV (header: bag,tape_start_m,tape_end_m[,note,lat_start_m,lat_end_m,leg])
or with repeated --run BAG START_M END_M. lat_* = signed sideways offset (m, left +) of the
reading point from the tape line; tape displacement = hypot(along, lat_end - lat_start).
leg = which walk in the bag (1 = first, 2 = second, ...; empty = the longest walk). Use it
for out-and-back bags: one CSV row per leg, the robot still >= 1 s between legs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import go2_bag  # noqa: E402

SPORT = "/sportmodestate"
REQUEST = "/api/sport/request"

# V gate, decisions-log 18 Nov 2026: Point-LIO vs a measured 5 m line, keyboard teleop 0.30 and
# 0.45 m/s, 3 runs each; pass = every run within 5 %.
POINT_LIO = "/state_estimation_at_scan"  # 14 Hz Point-LIO odometry (lab 25 Nov); /odom relays Unitree
V_SPEEDS_MPS = (0.30, 0.45)
V_RUNS_PER_SPEED = 3  # one run = one listed row (one walk): an out-and-back bag listed by leg is two runs
V_LINE_M = 5.0
V_BAR = 0.05
# Not in the 18 Nov rule: proposed, to be frozen by a dated decisions-log row before collection.
V_SPEED_TOL_MPS = 0.05
V_LINE_TOL_M = 0.5
V_TOL_STATUS = "PROPOSED, to be frozen by a dated decisions-log row before collection"
EPS = 1e-9  # float slack only, so inclusive bars hold: 5.25 / 5.0 - 1 = 0.0500000000000000444


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_walk(t: np.ndarray, speed: np.ndarray, move_thresh: float, merge_s: float,
              leg: int = 0, min_walk_s: float = 1.5) -> tuple[int, int, int]:
    """(start, end, number of walks >= min_walk_s). leg = 0: the longest walk. leg = k >= 1: the
    k-th walk in time order (walks shorter than min_walk_s are ignored; a stop longer than merge_s splits walks)."""
    segs = _segments(speed > move_thresh)
    if not segs:
        raise ValueError(f"no motion above {move_thresh} m/s")
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if t[s] - t[merged[-1][1] - 1] <= merge_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    walks = [se for se in merged if t[se[1] - 1] - t[se[0]] >= min_walk_s]
    if leg <= 0:
        s, e = max(merged, key=lambda se: t[se[1] - 1] - t[se[0]])
        return int(s), int(e), len(walks)
    if leg > len(walks):
        raise ValueError(f"leg {leg} asked, bag has {len(walks)} walks of >= {min_walk_s} s")
    s, e = walks[leg - 1]
    return int(s), int(e), len(walks)


def still_pose(t, pos, t_edge, still_s, before: bool):
    sel = (t >= t_edge - still_s) & (t < t_edge) if before else (t > t_edge) & (t <= t_edge + still_s)
    if sel.sum() < 3:
        return None, int(sel.sum())
    return np.median(pos[sel], axis=0), int(sel.sum())


def disp_xy(p0, p1) -> float:
    return float(np.hypot(*(np.asarray(p1)[:2] - np.asarray(p0)[:2])))


def analyse_run(bag: str, tape_start: float, tape_end: float, args, lat_start: float = 0.0,
                lat_end: float = 0.0, leg: int = 0) -> dict:
    out: dict = {"bag": str(bag), "leg": leg, "tape_start_m": tape_start, "tape_end_m": tape_end,
                 "lat_start_m": lat_start, "lat_end_m": lat_end}
    along, lateral = abs(tape_end - tape_start), lat_end - lat_start
    tape = float(np.hypot(along, lateral))  # true straight-line displacement of the reading point
    out.update(tape_along_m=along, tape_lateral_m=lateral, tape_m=tape)
    if not tape > 0:
        raise ValueError(f"tape displacement {tape} m: no run")
    sp = go2_bag.read_topic(bag, SPORT, stride=args.stride)
    t = sp.t
    vel_b = sp.cols["velocity"]
    speed = np.hypot(vel_b[:, 0], vel_b[:, 1])
    s, e, out["walks_in_bag"] = find_walk(t, speed, args.move_thresh, args.merge_s, leg, args.min_walk_s)
    t0, t1 = t[s], t[e - 1]
    if t1 - t0 < args.min_walk_s:
        raise ValueError(f"walk {t1 - t0:.2f} s is shorter than --min-walk-s {args.min_walk_s} s")
    out.update(walk_t0_s=float(t0 - t[0]), walk_s=float(t1 - t0))
    out["tape_speed_mps"] = tape / (t1 - t0) if t1 > t0 else float("nan")

    srcs: dict[str, dict] = {}
    # sport_pos
    p0, n0 = still_pose(t, sp.cols["position"], t0, args.still_s, True)
    p1, n1 = still_pose(t, sp.cols["position"], t1, args.still_s, False)
    if p0 is not None and p1 is not None:
        srcs["sport_pos"] = {"disp_m": disp_xy(p0, p1)}
    else:
        srcs["sport_pos"] = {"disp_m": float("nan"), "why": f"no still pose (before {n0}, after {n1} rows)"}
    # sport_int: body velocity -> world by yaw, integrated over the walk
    yaw = sp.cols["rpy"][:, 2]
    c, s_ = np.cos(yaw), np.sin(yaw)
    vw = np.stack([c * vel_b[:, 0] - s_ * vel_b[:, 1], s_ * vel_b[:, 0] + c * vel_b[:, 1]], 1)
    lo = np.searchsorted(t, t0 - args.still_s)
    hi = np.searchsorted(t, t1 + args.still_s, side="right")
    dt = np.diff(t[lo:hi])
    integ = (0.5 * (vw[lo + 1:hi] + vw[lo:hi - 1]) * dt[:, None]).sum(0)
    srcs["sport_int"] = {"disp_m": float(np.hypot(*integ))}
    out["mean_vx_body_mps"] = float(vel_b[s:e, 0].mean())
    out["yaw_drift_deg"] = float(np.degrees(np.angle(np.exp(1j * (yaw[e - 1] - yaw[s])))))
    out["error_codes_in_walk"] = sorted({int(x) for x in sp.cols["error_code"][s:e]})

    # odometry topics
    for topic in go2_bag.odometry_topics(bag):
        od = go2_bag.read_topic(bag, topic, stride=args.odom_stride)
        q0, m0 = still_pose(od.t, od.cols["position"], t0, args.still_s, True)
        q1, m1 = still_pose(od.t, od.cols["position"], t1, args.still_s, False)
        frames = f"{od.cols['frame_id'][0]} -> {od.cols['child_frame_id'][0]}" if len(od) else ""
        if q0 is None or q1 is None:
            srcs[topic] = {"disp_m": float("nan"), "frames": frames,
                           "why": f"no still pose (before {m0}, after {m1} rows)"}
        else:
            srcs[topic] = {"disp_m": disp_xy(q0, q1), "frames": frames}

    for v in srcs.values():
        v["ratio"] = v["disp_m"] / tape if tape > 0 else float("nan")
    out["sources"] = srcs

    # commanded speed
    info = go2_bag.topic_info(bag)
    if REQUEST in info and info[REQUEST][1] > 0:
        rq = go2_bag.read_topic(bag, REQUEST)
        sel = (rq.t >= t0) & (rq.t <= t1) & (rq.cols["api_id"] == go2_bag.SPORT_API_MOVE)
        xs = rq.cols["x"][sel]
        out["cmd_vx_median"] = float(np.nanmedian(xs)) if sel.any() else float("nan")
        out["cmd_move_msgs"] = int(sel.sum())
    return out


def _f(r, k):
    v = (r.get(k) or "").strip()
    return float(v) if v else 0.0


def load_runs(args) -> list[tuple]:
    runs = []
    if args.runs_csv:
        base = Path(args.runs_csv).resolve().parent
        with open(args.runs_csv, newline="") as f:
            for r in csv.DictReader(f):
                b = Path(r["bag"])
                runs.append((str(b if b.is_absolute() else base / b), float(r["tape_start_m"]),
                             float(r["tape_end_m"]), r.get("note", ""), _f(r, "lat_start_m"), _f(r, "lat_end_m"),
                             int(_f(r, "leg"))))
    for b, a, z in args.run or []:
        runs.append((b, float(a), float(z), "", 0.0, 0.0, 0))
    if not runs:
        raise SystemExit("give --runs-csv or at least one --run BAG START_M END_M")
    return runs


def summarize(rows: list[dict]) -> tuple[dict, list[str]]:
    """Ratio statistics per source. rows = every expected run, errored ones included."""
    n_exp = len(rows)
    names = sorted({k for r in rows for k in r.get("sources", {})})
    summ, lines = {}, []
    lines.append(f"{'source':28s} {'n/exp':>5s} {'ratio mean':>10s} {'min':>7s} {'max':>7s} {'max |err| %':>11s}")
    for n in names:
        rs = np.array([r.get("sources", {}).get(n, {}).get("ratio", np.nan) for r in rows], dtype=float)
        rs = rs[np.isfinite(rs)]  # errored run, source absent, or NaN ratio: not counted in n
        tag = "  (gate source)" if n == POINT_LIO else ""
        if rs.size == 0:
            summ[n] = {"n": 0, "n_expected": n_exp}
            lines.append(f"{n:28s} {'0/' + str(n_exp):>5s}  (no usable runs){tag}")
            continue
        worst = float(np.max(np.abs(rs - 1.0)) * 100)
        summ[n] = {"n": int(rs.size), "n_expected": n_exp, "ratio_mean": float(rs.mean()),
                   "ratio_min": float(rs.min()), "ratio_max": float(rs.max()), "max_abs_err_pct": worst}
        lines.append(f"{n:28s} {f'{rs.size}/{n_exp}':>5s} {rs.mean():10.3f} {rs.min():7.3f} {rs.max():7.3f} "
                     f"{worst:11.1f}{tag}")
    return summ, lines


def _name(r: dict) -> str:
    return Path(r["bag"]).name + (f" leg {r['leg']}" if r.get("leg") else "")


def v_gate(rows: list[dict], summ: dict, bar: float, speed_tol: float = V_SPEED_TOL_MPS,
           line_tol: float = V_LINE_TOL_M) -> tuple[str, list[str]]:
    """Official V-gate status (PASS, FAIL, INVALID, DIAGNOSTIC ONLY) and the reasons.
    Every listed row is one run and must be accounted for (see the module docstring)."""
    pl = summ.get(POINT_LIO)
    n_err = sum("error" in r for r in rows)
    bad = [f"{n_err} of {len(rows)} runs not analysed (ERROR above)"] if n_err else []
    if pl and pl["n"] < pl["n_expected"]:
        bad.append(f"Point-LIO ratio missing in {pl['n_expected'] - pl['n']} of {pl['n_expected']} runs")
    no_cmd = [_name(r) for r in rows if "error" not in r and not np.isfinite(r.get("cmd_vx_median", np.nan))]
    if no_cmd:
        bad.append(f"no readable commanded speed ({REQUEST} Move x) in {', '.join(no_cmd)}")
    if bad:
        return "INVALID", bad
    why = []
    for val, ref, what in ((bar, V_BAR, f"bar {bar * 100:g} % (--max-rel-err) is not the decided {V_BAR * 100:g} %"),
                           (speed_tol, V_SPEED_TOL_MPS, f"speed tolerance {speed_tol:g} m/s (--speed-tol-mps) "
                                                        f"is not the proposed {V_SPEED_TOL_MPS:g} m/s"),
                           (line_tol, V_LINE_TOL_M, f"line tolerance {line_tol:g} m (--line-tol-m) "
                                                    f"is not the proposed {V_LINE_TOL_M:g} m")):
        if not np.isclose(val, ref):
            why.append(what)
    if pl is None:
        why.append(f"no Point-LIO source ({POINT_LIO}) in any run; /sportmodestate and "
                   "/utlidar/robot_odom are one Unitree estimate and cannot decide the gate")
    runs: dict[float, int] = {v: 0 for v in V_SPEEDS_MPS}
    groups: dict[str, set] = {}  # bag -> speed groups of its runs
    legs: dict[str, list] = {}  # bag -> listed legs
    for r in rows:
        name, bag, cmd = _name(r), str(Path(r["bag"]).resolve()), r["cmd_vx_median"]
        legs.setdefault(bag, []).append(r.get("leg", 0))
        if not r.get("leg") and r.get("walks_in_bag", 1) > 1:
            why.append(f"{name}: the bag has {r['walks_in_bag']} walks; the row must name its leg")
        hit = [v for v in V_SPEEDS_MPS if abs(abs(cmd) - v) <= speed_tol + EPS]  # a backward leg commands -x
        if len(hit) == 1:
            runs[hit[0]] += 1
            groups.setdefault(bag, set()).add(hit[0])
        else:
            why.append(f"{name}: commanded {cmd:+.2f} m/s is " + ("in two speed groups" if hit else
                       f"not one of {', '.join(f'{v:.2f}' for v in V_SPEEDS_MPS)} (± {speed_tol:g})"))
        if not abs(r["tape_m"] - V_LINE_M) <= line_tol + EPS:
            why.append(f"{name}: tape {r['tape_m']:.2f} m, protocol line {V_LINE_M:g} m")
    for bag, ls in legs.items():
        if len(ls) > 1 and (0 in ls or len(set(ls)) < len(ls)):
            why.append(f"{Path(bag).name}: listed {len(ls)} times with legs {ls}; each row must name a distinct leg")
        if len(groups.get(bag, ())) > 1:
            why.append(f"{Path(bag).name}: runs at both 0.30 and 0.45 m/s; one bag holds runs of one speed")
    for v, n in runs.items():
        if n != V_RUNS_PER_SPEED:
            why.append(f"{n} runs at commanded {v:.2f} m/s; the rule prescribes exactly {V_RUNS_PER_SPEED}")
    if why:
        return "DIAGNOSTIC ONLY", why
    worst = max(abs(r["sources"][POINT_LIO]["ratio"] - 1.0) for r in rows)
    return ("PASS" if worst <= bar + EPS else "FAIL"), []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-csv")
    ap.add_argument("--run", nargs=3, action="append", metavar=("BAG", "TAPE_START_M", "TAPE_END_M"))
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--move-thresh", type=float, default=0.10, help="m/s; walk detection")
    ap.add_argument("--merge-s", type=float, default=0.5, help="merge walk pieces closer than this")
    ap.add_argument("--still-s", type=float, default=0.5, help="still window before/after the walk")
    ap.add_argument("--min-walk-s", type=float, default=1.5, help="shorter walks are ignored when a leg is chosen")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th /sportmodestate message")
    ap.add_argument("--odom-stride", type=int, default=1)
    ap.add_argument("--max-rel-err", type=float, default=V_BAR,
                    help="per-run bar (decided 18 Nov: 0.05 = 5 %%). Another value gives DIAGNOSTIC ONLY.")
    ap.add_argument("--speed-tol-mps", type=float, default=V_SPEED_TOL_MPS,
                    help="commanded-speed match (PROPOSED). Another value gives DIAGNOSTIC ONLY.")
    ap.add_argument("--line-tol-m", type=float, default=V_LINE_TOL_M,
                    help="tape distance vs the 5 m line (PROPOSED). Another value gives DIAGNOSTIC ONLY.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bag, a, z, note, la, lz, leg in load_runs(args):
        try:
            r = analyse_run(bag, a, z, args, la, lz, leg)
        except Exception as ex:  # one bad run must not hide the others
            r = {"bag": bag, "error": f"{type(ex).__name__}: {ex}"}
        r["note"] = note
        rows.append(r)

    lines = ["T11 velocity check", ""]
    for r in rows:
        if "error" in r:
            lines.append(f"{Path(r['bag']).name}: ERROR {r['error']}")
            continue
        src = "  ".join(f"{k}={v['ratio']:.3f}" for k, v in r["sources"].items())
        lines.append(
            f"{Path(r['bag']).name}{' leg ' + str(r['leg']) if r.get('leg') else ''}: tape {r['tape_m']:.3f} m (along {r['tape_along_m']:.3f}, "
            f"lateral {r['tape_lateral_m']:+.3f}) in {r['walk_s']:.2f} s "
            f"({r['tape_speed_mps']:.3f} m/s), cmd vx {r.get('cmd_vx_median', float('nan')):.2f}, "
            f"est vx {r['mean_vx_body_mps']:.3f}, yaw drift {r['yaw_drift_deg']:+.1f} deg | ratio {src}"
        )
    summ, slines = summarize(rows)
    status, reasons = v_gate(rows, summ, args.max_rel_err, args.speed_tol_mps, args.line_tol_m)
    tol = {"speed_tol_mps": args.speed_tol_mps, "line_m": V_LINE_M, "line_tol_m": args.line_tol_m,
           "status": V_TOL_STATUS}
    lines += ["", *slines, "",
              f"V GATE (Point-LIO vs tape, every run within {args.max_rel_err * 100:g} %; "
              f"{V_RUNS_PER_SPEED} runs each at {' and '.join(f'{v:.2f}' for v in V_SPEEDS_MPS)} m/s; "
              f"speed ± {args.speed_tol_mps:g} m/s, line {V_LINE_M:g} ± {args.line_tol_m:g} m PROPOSED): "
              + (f"{status}: " + "; ".join(reasons) if reasons else status),
              "Unitree sources (sport_pos, sport_int, /utlidar/robot_odom, /odom) are reported, never the gate."]
    text = "\n".join(lines) + "\n"
    (out / "v_check.txt").write_text(text)
    (out / "v_check.json").write_text(json.dumps({"runs": rows, "summary": summ,
                                                   "gate": {"status": status, "reasons": reasons},
                                                   "max_rel_err": args.max_rel_err, "tolerances": tol,
                                                   "settings": {k: v for k, v in vars(args).items()
                                                                if k not in ("run",)}},
                                                  indent=2, default=float))
    print(text, end="")
    return {"PASS": 0, "FAIL": 0, "INVALID": 1}.get(status, 2)

if __name__ == "__main__":
    raise SystemExit(main())
