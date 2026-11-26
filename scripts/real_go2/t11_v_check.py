# SPDX-License-Identifier: BSD-3-Clause
"""T11 velocity check (V gate input): taped straight walks vs the robot's speed sources.

For each run (one bag + tape readings of the head front edge at start and end):
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

This script measures. It does not decide. Without --max-rel-err it prints no
PASS/FAIL: the V-gate pass bar is Charbel's decision (CLAUDE.md rule 8).
A source gets PASS or FAIL only when every listed run gave it a finite ratio; a run
that is missing, errored (no walk, walk shorter than --min-walk-s, zero tape) or NaN
makes it INVALID. Exit 1 when any run errored or any verdict is INVALID.

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


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_walk(t: np.ndarray, speed: np.ndarray, move_thresh: float, merge_s: float,
              leg: int = 0, min_walk_s: float = 1.5) -> tuple[int, int]:
    """leg = 0: the longest walk. leg = k >= 1: the k-th walk in time order
    (walks shorter than min_walk_s are ignored; a stop longer than merge_s splits walks)."""
    segs = _segments(speed > move_thresh)
    if not segs:
        raise ValueError(f"no motion above {move_thresh} m/s")
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if t[s] - t[merged[-1][1] - 1] <= merge_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    if leg <= 0:
        s, e = max(merged, key=lambda se: t[se[1] - 1] - t[se[0]])
        return int(s), int(e)
    walks = [se for se in merged if t[se[1] - 1] - t[se[0]] >= min_walk_s]
    if leg > len(walks):
        raise ValueError(f"leg {leg} asked, bag has {len(walks)} walks of >= {min_walk_s} s")
    s, e = walks[leg - 1]
    return int(s), int(e)


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
    s, e = find_walk(t, speed, args.move_thresh, args.merge_s, leg, args.min_walk_s)
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


def summarize(rows: list[dict], max_rel_err: float | None) -> tuple[dict, list[str]]:
    """rows = every expected run, errored ones included (they count as not measured)."""
    n_exp = len(rows)
    names = sorted({k for r in rows for k in r.get("sources", {})})
    summ, lines = {}, []
    lines.append(f"{'source':28s} {'n/exp':>5s} {'ratio mean':>10s} {'min':>7s} {'max':>7s} {'max |err| %':>11s}")
    for n in names:
        rs = np.array([r.get("sources", {}).get(n, {}).get("ratio", np.nan) for r in rows], dtype=float)
        n_bad = int((~np.isfinite(rs)).sum())  # errored run, source absent, or NaN ratio
        rs = rs[np.isfinite(rs)]
        if rs.size == 0:
            summ[n] = {"n": 0, "n_expected": n_exp}
            if max_rel_err is not None:
                summ[n]["verdict"] = "INVALID"
            lines.append(f"{n:28s} {'0/' + str(n_exp):>5s}  (no usable runs)"
                         + ("  INVALID" if max_rel_err is not None else ""))
            continue
        worst = float(np.max(np.abs(rs - 1.0)) * 100)
        summ[n] = {"n": int(rs.size), "n_expected": n_exp, "ratio_mean": float(rs.mean()),
                   "ratio_min": float(rs.min()), "ratio_max": float(rs.max()), "max_abs_err_pct": worst}
        if max_rel_err is not None:
            summ[n]["verdict"] = "INVALID" if n_bad else ("PASS" if worst <= max_rel_err * 100 else "FAIL")
        lines.append(f"{n:28s} {f'{rs.size}/{n_exp}':>5s} {rs.mean():10.3f} {rs.min():7.3f} {rs.max():7.3f} "
                     f"{worst:11.1f}" + (f"  {summ[n]['verdict']}" if max_rel_err is not None else ""))
    n_err = sum("error" in r for r in rows)
    if n_err:
        lines.append(f"{n_err} of {n_exp} runs not analysed (ERROR above): no source can PASS.")
    if max_rel_err is None:
        lines.append("No pass bar given (--max-rel-err): descriptive only. The V-gate bar is Charbel's decision.")
    return summ, lines


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
    ap.add_argument("--max-rel-err", type=float, default=None,
                    help="optional pass bar, e.g. 0.05 = 5 %%. Only with a dated approval row.")
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
    summ, slines = summarize(rows, args.max_rel_err)
    lines += ["", *slines]
    text = "\n".join(lines) + "\n"
    (out / "v_check.txt").write_text(text)
    (out / "v_check.json").write_text(json.dumps({"runs": rows, "summary": summ,
                                                   "max_rel_err": args.max_rel_err,
                                                   "settings": {k: v for k, v in vars(args).items()
                                                                if k not in ("run",)}},
                                                  indent=2, default=float))
    print(text, end="")
    bad = any("error" in r for r in rows) or any(v.get("verdict") == "INVALID" for v in summ.values())
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
