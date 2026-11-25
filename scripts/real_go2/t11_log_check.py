# SPDX-License-Identifier: BSD-3-Clause
"""T11 logger check: is a Go2 bag usable as world-model data? (recording contract §1, §7)

Per topic: message count, rate, and receive-time gaps (> 20, > 30, > 100 ms).
/lowstate:  `tick` steps. A receive gap with continuous ticks = recorder/transport
            delay (messages arrive late, none lost). A tick jump = messages lost.
            Max motor temperature per joint (input for the temperature limit).
/lowcmd:    rate, mode, kp / kd values seen (H2 needs 25 / 0.5; mcf uses 40 / 1).
Joint topics: gaps where ALL topics stop together (H0 saw up to 92 ms).

50 Hz emulation (contract §1): a policy loop at 20 ms reads the newest /lowstate.
The interval between the states it gets is the step interval:
  > 30 ms  -> step flagged;  > 100 ms -> segment ends.
Usable windows = 58 consecutive steps (32 context + 26 rows) with no flagged step,
counted both as non-overlapping windows and as sliding start positions.

This measures the bag. It does not decide. Thresholds are the contract's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import go2_bag  # noqa: E402

GAP_LEVELS_MS = (20.0, 30.0, 100.0)
STEP_S = 0.020
FLAG_S = 0.030
END_S = 0.100
WINDOW = 58


def gap_stats(t_ns: np.ndarray) -> dict:
    if len(t_ns) < 2:
        return {"count": int(len(t_ns))}
    d = np.diff(t_ns.astype(np.float64)) * 1e-6  # ms
    dur = (t_ns[-1] - t_ns[0]) * 1e-9
    out = {
        "count": int(len(t_ns)),
        "duration_s": float(dur),
        "rate_hz": float((len(t_ns) - 1) / dur) if dur > 0 else float("nan"),
        "dt_median_ms": float(np.median(d)),
        "dt_p99_ms": float(np.percentile(d, 99)),
        "dt_max_ms": float(d.max()),
    }
    for g in GAP_LEVELS_MS:
        out[f"gaps_gt_{int(g)}ms"] = int((d > g).sum())
    return out


def emulate_policy_steps(t_state: np.ndarray) -> dict:
    """50 Hz loop reading the newest state. t_state in seconds, sorted."""
    t0, t1 = t_state[0], t_state[-1]
    grid = np.arange(t0 + STEP_S, t1 + 1e-9, STEP_S)
    idx = np.searchsorted(t_state, grid, side="right") - 1
    used = t_state[idx]
    interval = np.diff(np.r_[t_state[0], used])
    flagged = interval > FLAG_S
    seg_end = interval > END_S
    # windows: runs of unflagged steps (a flagged step breaks the run)
    ok = ~flagged
    runs = []
    d = np.diff(np.r_[0, ok.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        runs.append(e - s)
    runs = np.asarray(runs, dtype=np.int64)
    return {
        "steps": int(len(grid)),
        "flagged_steps_gt_30ms": int(flagged.sum()),
        "segment_ends_gt_100ms": int(seg_end.sum()),
        "segments": int(seg_end.sum()) + 1,
        "repeated_state_steps": int((interval == 0).sum()),
        "max_step_interval_ms": float(interval.max() * 1e3) if len(interval) else float("nan"),
        "usable_windows_nonoverlap": int((runs // WINDOW).sum()) if runs.size else 0,
        "usable_window_starts_sliding": int(np.clip(runs - WINDOW + 1, 0, None).sum()) if runs.size else 0,
        "usable_fraction_of_steps": float(runs[runs >= WINDOW].sum() / max(len(grid), 1)) if runs.size else 0.0,
    }


def common_stops(times: dict[str, np.ndarray], min_ms: float = 20.0) -> dict:
    """Gaps > min_ms in the merged timeline of all busy topics (all topics silent together)."""
    busy = [v for v in times.values() if len(v) > 100]
    if not busy:
        return {}
    allt = np.sort(np.concatenate(busy)).astype(np.float64)
    d = np.diff(allt) * 1e-6
    return {"all_topics_silent_gt_20ms": int((d > min_ms).sum()),
            "all_topics_silent_max_ms": float(d.max()) if d.size else 0.0}


def analyse(bag: str, stride_temp: int) -> dict:
    info = go2_bag.topic_info(bag)
    times = go2_bag.receive_times(bag, [t for t, (_, n) in info.items() if n > 0])
    res: dict = {"bag": str(bag), "topics": {}}
    for topic, (mt, n) in sorted(info.items()):
        res["topics"][topic] = {"type": mt, **gap_stats(times.get(topic, np.zeros(0, np.int64)))}
    res["common"] = common_stops(times)

    if info.get("/lowstate", (None, 0))[1] > 1:
        lo = go2_bag.read_topic(bag, "/lowstate")
        tick = lo.cols["tick"].astype(np.int64)
        dt = np.diff(tick)
        mode = int(np.bincount(dt[(dt > 0) & (dt < 1000)]).argmax()) if (dt > 0).any() else 0
        lost = int(np.clip(dt // max(mode, 1) - 1, 0, None)[dt > 0].sum()) if mode else 0
        rx = np.diff(lo.t_ns.astype(np.float64)) * 1e-6
        late_not_lost = int(((rx > 20) & (dt == mode)).sum())
        res["lowstate_tick"] = {
            "tick_step_mode": mode,
            "tick_steps_not_mode": int((dt != mode).sum()),
            "tick_backwards": int((dt < 0).sum()),
            "messages_lost_by_tick": lost,
            "rx_gaps_gt_20ms_with_continuous_tick": late_not_lost,
        }
        res["policy_50hz"] = emulate_policy_steps(lo.t)
        temp = lo.cols["temperature"][::max(stride_temp, 1)]
        res["motor_temp_max_c_sim_order"] = dict(
            zip(go2_bag.SIM_JOINT_NAMES, temp[:, go2_bag.SIM_FROM_SDK].max(0).tolist()))
        res["battery_soc_first_last"] = [int(lo.cols["soc"][0]), int(lo.cols["soc"][-1])]

    if info.get("/lowcmd", (None, 0))[1] > 0:
        lc = go2_bag.read_topic(bag, "/lowcmd")
        pairs, counts = np.unique(np.round(np.c_[lc.cols["kp"].ravel(), lc.cols["kd"].ravel()], 3),
                                  axis=0, return_counts=True)
        order = np.argsort(-counts)
        tot = counts.sum()
        res["lowcmd_gains"] = [{"kp": float(pairs[i, 0]), "kd": float(pairs[i, 1]),
                                "share": float(counts[i] / tot)} for i in order[:6]]
        res["lowcmd_modes"] = sorted({int(x) for x in np.unique(lc.cols["mode"])})
    return res


def render(res: dict) -> str:
    L = [f"T11 logger check: {res['bag']}", "", f"{'topic':34s} {'n':>8s} {'Hz':>7s} {'med ms':>7s} "
         f"{'p99 ms':>7s} {'max ms':>8s} {'>20':>5s} {'>30':>5s} {'>100':>5s}"]
    for tp, s in res["topics"].items():
        if s["count"] < 2:
            L.append(f"{tp:34s} {s['count']:8d}  (empty)")
            continue
        L.append(f"{tp:34s} {s['count']:8d} {s['rate_hz']:7.1f} {s['dt_median_ms']:7.2f} {s['dt_p99_ms']:7.2f} "
                 f"{s['dt_max_ms']:8.1f} {s['gaps_gt_20ms']:5d} {s['gaps_gt_30ms']:5d} {s['gaps_gt_100ms']:5d}")
    if res.get("common"):
        c = res["common"]
        L.append(f"\nAll topics silent together: {c['all_topics_silent_gt_20ms']} times > 20 ms, "
                 f"max {c['all_topics_silent_max_ms']:.1f} ms")
    if "lowstate_tick" in res:
        k = res["lowstate_tick"]
        L.append(f"/lowstate tick: step {k['tick_step_mode']}, {k['tick_steps_not_mode']} other steps, "
                 f"{k['messages_lost_by_tick']} lost by tick, {k['tick_backwards']} backwards, "
                 f"{k['rx_gaps_gt_20ms_with_continuous_tick']} receive gaps > 20 ms with continuous tick (late, not lost)")
        p = res["policy_50hz"]
        L.append(f"50 Hz emulation: {p['steps']} steps, {p['flagged_steps_gt_30ms']} flagged (> 30 ms), "
                 f"{p['segment_ends_gt_100ms']} segment ends (> 100 ms), max interval {p['max_step_interval_ms']:.1f} ms, "
                 f"{p['repeated_state_steps']} steps reused the previous state")
        L.append(f"  usable 58-step windows: {p['usable_windows_nonoverlap']} non-overlapping, "
                 f"{p['usable_window_starts_sliding']} sliding starts, "
                 f"{100 * p['usable_fraction_of_steps']:.1f} % of steps inside a usable run")
        t = res["motor_temp_max_c_sim_order"]
        L.append("Max motor temperature [C]: " + ", ".join(f"{k} {v:.0f}" for k, v in t.items()))
        L.append(f"Battery SOC first/last: {res['battery_soc_first_last']}")
    if "lowcmd_gains" in res:
        L.append("/lowcmd kp/kd (share of joint-commands): " + ", ".join(
            f"{g['kp']:g}/{g['kd']:g} {100 * g['share']:.1f}%" for g in res["lowcmd_gains"])
            + f"; modes {res['lowcmd_modes']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride-temp", type=int, default=10)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allres, text = [], ""
    for b in args.bags:
        try:
            r = analyse(b, args.stride_temp)
            text += render(r) + "\n"
        except Exception as ex:
            r = {"bag": b, "error": f"{type(ex).__name__}: {ex}"}
            text += f"{b}: ERROR {r['error']}\n\n"
        allres.append(r)
    (out / "log_check.txt").write_text(text)
    (out / "log_check.json").write_text(json.dumps(allres, indent=2, default=float))
    print(text, end="")
    return 0 if all("error" not in r for r in allres) else 1


if __name__ == "__main__":
    raise SystemExit(main())
