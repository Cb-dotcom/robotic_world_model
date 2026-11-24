#!/usr/bin/env python3
"""A4 D8 (draft 24 Nov 2026): the updated model's data = D6b's training data + the D7 fitting batch's training split.

Inputs:
  --d6b-data   D6b's prepared folder (go2_startup_prepare.py with reset_torque "zero"): segments/, startup.json,
               start_data/, reset_states.npz, manifest.json, r0/ and shift/ (their withheld.json)
  --batch      the D7 fitting batch (0.30 friction, seed 104, 64 x 5000)
  --probe      the D7 held-out probe (seed 916): read ONLY to prove it is absent from every output
The batch is split by the E22 rule (10 % of whole episodes withheld, seed 101) and prepared exactly as D6b's sources
(go2_startup_prepare.prepare_source: a reset row with 0 N m torque in front of every episode that began at a physical
reset; startup windows; start blocks with episode-end flags). D2's normalizer stays frozen: this script writes none.

Output (fresh folder):
  d7/            prepare_source output of the batch (segments, start_data, withheld.json)
  segments/      D6b's segments + the batch's (names d7_...), one manifest.csv
  startup.json   D6b's startup segments + the batch's
  start_data/    D6b's start files, then the batch's (the imagination start bank; identical for both PPO arms)
  reset_states.npz  D6b's reset states (sources 0, 1) + the batch's (source 2)
  manifest.json  counts, the window census (standard and startup windows per source; the realized D7 share under
                 uniform sampling over all windows), the startup reset share q (rule chosen by --q-rule), hashes
  split_check.json  verify_split on the written files: no row of the probe and no row of any withheld episode (D6b's
                 two sources and the batch) occurs in segments/, start_data/ or reset_states.npz
The q rule is a launch choice fixed before recording: "d6b" keeps D6b's value; "recompute" applies D6's rule
q = min(0.5, 256 / mean training-episode rows) to the pooled training episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

MB = Path(__file__).resolve().parent
for p in (str(MB), str(MB / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import go2_round0_prepare as prep  # noqa: E402
import go2_startup_prepare as sp  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

Q_RULES = ("d6b", "recompute")
KEY_DIMS = prep.STATE + prep.ACTION          # a row's identity: state (45) + action (12)
STARTUP_ROWS = 40


# ---------------------------------------------------------------- probe minimum counts (before any fit)
def probe_counts(trace):
    """Completed episodes that start at a physical reset (episode_step == 1 at the first row), split by outcome, and
    censored segments (unfinished at the end of the recording, or without a physical reset in the recording)."""
    step = trace["episode_step"]
    c = dict(episodes=0, from_reset=0, startup_falls=0, startup_falls_le25=0, startup_falls_26_40=0, later_falls=0,
             time_outs=0, censored_unfinished=0, censored_no_reset=0)
    for n, s, e, finished in prep.episodes(trace["done"]):
        c["episodes"] += 1
        if not finished:
            c["censored_unfinished"] += 1
            continue
        if int(step[s, n]) != 1:
            c["censored_no_reset"] += 1
            continue
        c["from_reset"] += 1
        fall = bool(trace["terminated"][e, n] and not trace["time_out"][e, n])
        L = e - s + 1
        if fall and L <= STARTUP_ROWS:
            c["startup_falls"] += 1
            c["startup_falls_le25" if L <= 25 else "startup_falls_26_40"] += 1
        elif fall:
            c["later_falls"] += 1
        else:
            c["time_outs"] += 1
    return c


def probe_minimum(c, min_startup_falls=20, min_time_outs=30):
    ok = c["startup_falls"] >= min_startup_falls and c["time_outs"] >= min_time_outs
    return dict(ok=bool(ok), startup_falls=c["startup_falls"], time_outs=c["time_outs"],
                bars=dict(startup_falls=min_startup_falls, time_outs=min_time_outs))


# ---------------------------------------------------------------- window census (the fitter's uniform sampling)
def window_census(manifest_rows, startup_names, seq_len, history_horizon):
    """Standard windows per segment = rows - seq_len + 1 (one episode per segment, the terminal row last, so no
    window crosses a reset); startup windows per startup segment = min(history_horizon - 1, rows - 1)
    (fit_world_model_go2_pilot_segments.build_startup_groups)."""
    startup = set(startup_names)
    std = sum(max(0, n - seq_len + 1) for _, n in manifest_rows)
    st = sum(min(history_horizon - 1, n - 1) for name, n in manifest_rows if name in startup)
    return dict(standard=int(std), startup=int(st), total=int(std + st))


def _manifest(seg_dir):
    lines = (Path(seg_dir) / "manifest.csv").read_text().splitlines()[1:]
    return [(ln.split(",")[0], int(ln.split(",")[2])) for ln in lines if ln]


def _horizons():
    from configs.go2_flat_cfg import Go2FlatConfig
    mac = Go2FlatConfig().model_architecture_config
    return int(mac.history_horizon), int(mac.history_horizon) + int(mac.forecast_horizon)


# ---------------------------------------------------------------- split membership on the written files
def _keys(rows):
    a = np.ascontiguousarray(np.asarray(rows, np.float32)[:, :KEY_DIMS])
    return {hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in a}


def excluded_keys(trace, episodes, reset_torque="zero"):
    """Row keys of the given episodes [(env, first, last)]: every recorded row (next_state, action) and the reset
    row (pre-action state with the torque convention, previous action) the preparation would put in front."""
    keys = set()
    for n, s, e in episodes:
        keys |= _keys(prep.rows_of(trace, n, s, e))
        keys |= _keys(sp.reset_row(trace, n, s, reset_torque)[None])
    return keys


def output_keys(folder):
    """Row keys of every file the fitter, the normalizer or the start sampler can read in a D8 data folder."""
    folder = Path(folder)
    keys, rows = set(), 0
    for f in sorted((folder / "segments").glob("*.csv")) + sorted((folder / "start_data").glob("state_action_data_*.csv")):
        if f.name == "manifest.csv":
            continue
        a = np.loadtxt(f, delimiter=",", dtype=np.float32, ndmin=2)
        keys |= _keys(a)
        rows += len(a)
    with np.load(folder / "reset_states.npz") as z:
        keys |= _keys(np.concatenate([z["state"], z["previous_action"]], axis=1))
        rows += len(z["state"])
    return keys, rows


def verify_split(folder, held_out):
    """held_out: [(label, trace dict, episodes [(env, first, last)])]. Returns per label the number of its rows found
    in the outputs (must be 0) and the number of distinct rows checked."""
    out_keys, rows = output_keys(folder)
    res = dict(output_rows=rows, output_distinct=len(out_keys), groups={})
    for label, trace, eps in held_out:
        k = excluded_keys(trace, eps)
        res["groups"][label] = dict(episodes=len(eps), rows_checked=len(k), found_in_outputs=len(k & out_keys))
    res["ok"] = all(g["found_in_outputs"] == 0 for g in res["groups"].values())
    return res


def all_episodes(trace):
    return [(n, s, e) for n, s, e, _ in prep.episodes(trace["done"])]


def withheld_episodes(path):
    return [(d["env"], d["first"], d["last"]) for d in json.loads(Path(path).read_text())]


def load(path):
    with np.load(path, allow_pickle=False) as z:
        validate_trace(z)
        return {k: z[k] for k in z.files if k != "metadata"}


# ---------------------------------------------------------------- preparation
def prepare_d8(d6b_data, batch, out, probe, q_rule, frac=0.1, seed=101, d6b_sources=None):
    """d6b_sources: {"r0": path, "shift": path} to verify D6b's withheld episodes too (default: the paths in D6b's
    manifest)."""
    if q_rule not in Q_RULES:
        raise ValueError(f"q rule must be one of {Q_RULES}")
    d6b, out = Path(d6b_data), Path(out)
    man6 = json.loads((d6b / "manifest.json").read_text())
    if man6.get("reset_torque") != "zero":
        raise ValueError("the D6b data must use the 0 N m reset-row torque convention")
    if sha256(probe) == sha256(batch) or sha256(probe) in {s["sha256"] for s in man6["sources"].values()}:
        raise ValueError("the probe is one of the fitting sources")
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.mkdir(parents=True)
    src = sp.prepare_source(batch, out / "d7", "d7", frac, seed, "zero")
    seg = out / "segments"
    seg.mkdir()
    lines = ["name,steps,rows,num_envs"]
    rows6 = _manifest(d6b / "segments")
    for name, n in rows6:
        (seg / f"{name}.csv").write_bytes((d6b / "segments" / f"{name}.csv").read_bytes())
        lines.append(f"{name},{n},{n},1")
    for name, n in src["manifest"]:
        if name in dict(rows6):
            raise ValueError(f"segment name collision: {name}")
        (seg / f"{name}.csv").write_bytes((out / "d7" / "segments" / f"{name}.csv").read_bytes())
        lines.append(f"{name},{n},{n},1")
    (seg / "manifest.csv").write_text("\n".join(lines) + "\n")
    startup6 = json.loads((d6b / "startup.json").read_text())["segments"]
    (out / "startup.json").write_text(json.dumps(dict(segments=startup6 + src["startup"],
                                                      rule="first row = physical reset row"), indent=1))
    start = out / "start_data"
    start.mkdir()
    files, nrows, min_rows = 0, 0, None
    for d in (d6b / "start_data", out / "d7" / "start_data"):
        for f in sorted(d.glob("state_action_data_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            data = f.read_bytes()
            (start / f"state_action_data_{files}.csv").write_bytes(data)
            n = data.count(b"\n")
            nrows += n
            min_rows = n if min_rows is None else min(min_rows, n)
            files += 1
    with np.load(d6b / "reset_states.npz") as z:
        r6 = {k: z[k] for k in z.files}
    rb = src["resets"]
    r = {k: np.concatenate([r6[k], rb[k]]) for k in ("state", "previous_action", "env", "row")}
    r["source"] = np.concatenate([r6["source"], np.full(len(rb["env"]), 2, np.int8)])
    with (out / "reset_states.npz").open("xb") as f:
        np.savez(f, **r)
    # q and the window census
    ep6 = sum(s["counts"]["training"] for s in man6["sources"].values())
    rows6_ep = sum(s["counts"]["training_episode_rows"] for s in man6["sources"].values())
    c7 = src["counts"]
    pooled_mean = (rows6_ep + c7["training_episode_rows"]) / max(ep6 + c7["training"], 1)
    q_recompute = min(sp.Q_CAP, sp.IMAGINED_EPISODE_STEPS / pooled_mean)
    q = float(man6["startup_reset_share"]) if q_rule == "d6b" else q_recompute
    H, seq = _horizons()
    cen6 = window_census(rows6, startup6, seq, H)
    cen7 = window_census(src["manifest"], src["startup"], seq, H)
    share = cen7["total"] / max(cen6["total"] + cen7["total"], 1)
    (start / "start_data.json").write_text(json.dumps(dict(
        rows=nrows, files=files, min_file_rows=int(min_rows), episodes=ep6 + c7["training"],
        layout="next_state 45 (a reset row holds the pre-action reset state), action 12, contact 8, episode-end flag 1",
        sources="D6b's pooled start data (round-0 nominal + D1 shift) + the D7 batch's training split (A4 D8)"), indent=1))
    # split membership, on the files just written
    srcs = d6b_sources or {"r0": man6["sources"]["r0"]["path"], "shift": man6["sources"]["sh"]["path"]}
    held = [("probe_916_all_rows", load(probe), None), ("d7_batch_withheld", None, out / "d7" / "withheld.json")]
    for tag, sub in (("r0", "r0"), ("shift", "shift")):
        held.append((f"d6b_{tag}_withheld", srcs[tag], d6b / sub / "withheld.json"))
    groups, batch_tr = [], load(batch)
    for label, tr, wpath in held:
        if label.startswith("probe"):
            groups.append((label, tr, all_episodes(tr)))
        elif label == "d7_batch_withheld":
            groups.append((label, batch_tr, withheld_episodes(wpath)))
        else:
            groups.append((label, load(tr), withheld_episodes(wpath)))
    check = verify_split(out, groups)
    check["sources_sha256"] = dict(probe=sha256(probe), batch=sha256(batch),
                                   r0=sha256(srcs["r0"]), shift=sha256(srcs["shift"]))
    if check["sources_sha256"]["r0"] != man6["sources"]["r0"]["sha256"] or \
            check["sources_sha256"]["shift"] != man6["sources"]["sh"]["sha256"]:
        raise ValueError("D6b's source recordings differ from its manifest")
    (out / "split_check.json").write_text(json.dumps(check, indent=1))
    result = dict(
        rule=("A4 D8 draft: D6b's training data (unchanged) + the D7 batch's E22 training split (fraction %s, seed %s), "
              "reset rows at 0 N m; D2's normalizer frozen; the probe never enters" % (frac, seed)),
        d6b=dict(path=str(d6b), manifest_sha256=sha256(d6b / "manifest.json")),
        batch=dict(path=str(batch), sha256=src["source_sha256"], checkpoint_sha256=src["checkpoint_sha256"], counts=c7),
        probe=dict(path=str(probe), sha256=sha256(probe)),
        segments=len(lines) - 1, startup_segments=len(startup6) + len(src["startup"]), reset_states=int(len(r["env"])),
        window_census=dict(d6b=cen6, d7=cen7, d7_share_uniform=share, seq_len=seq, history_horizon=H),
        startup_reset_share=q, q_rule=q_rule, q_d6b=float(man6["startup_reset_share"]), q_recompute=q_recompute,
        pooled_mean_training_episode_rows=pooled_mean, split_check_ok=check["ok"],
        outputs=dict(segments_manifest_sha256=sha256(seg / "manifest.csv"), startup_json_sha256=sha256(out / "startup.json"),
                     reset_states_sha256=sha256(out / "reset_states.npz"),
                     start_data_json_sha256=sha256(start / "start_data.json"),
                     d7_withheld_sha256=sha256(out / "d7" / "withheld.json"),
                     split_check_sha256=sha256(out / "split_check.json")),
        code_sha256=sha256(__file__))
    (out / "manifest.json").write_text(json.dumps(result, indent=1))
    if not check["ok"]:
        raise ValueError(f"held-out rows found in the outputs: {check['groups']}")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--d6b-data", type=Path, required=True)
    a.add_argument("--batch", type=Path, required=True)
    a.add_argument("--probe", type=Path, required=True)
    a.add_argument("--q-rule", choices=Q_RULES, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("probe-counts")
    b.add_argument("--probe", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--min-startup-falls", type=int, default=20)
    b.add_argument("--min-time-outs", type=int, default=30)
    x = p.parse_args(argv)
    if x.cmd == "probe-counts":
        c = probe_counts(load(x.probe))
        m = probe_minimum(c, x.min_startup_falls, x.min_time_outs)
        with x.out.open("x") as f:
            json.dump(dict(counts=c, minimum=m, probe_sha256=sha256(x.probe)), f, indent=1)
        print(f"[d8-probe] {c}; minimum {'met' if m['ok'] else 'NOT met'} ({m['bars']})")
        return 0 if m["ok"] else 4
    r = prepare_d8(x.d6b_data, x.batch, x.out, x.probe, x.q_rule)
    w = r["window_census"]
    print(f"[d8-prepare] {r['segments']} segments ({r['startup_segments']} startup); D7 windows {w['d7']['total']} of "
          f"{w['d6b']['total'] + w['d7']['total']} -> realized D7 share {w['d7_share_uniform']:.4f}; q = "
          f"{r['startup_reset_share']:.4f} ({r['q_rule']}; d6b {r['q_d6b']:.4f}, recompute {r['q_recompute']:.4f}); "
          f"split check {'ok' if r['split_check_ok'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
