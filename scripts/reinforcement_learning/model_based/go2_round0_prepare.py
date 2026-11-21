#!/usr/bin/env python3
"""Plan v6 amendment A3 §2-3 (21 Nov 2026): the frozen Go2 round-0 data manifest.

Input: the nominal s2 recording (rsl_rl/record_pre_reset_transitions.py, schema pre_reset_transitions_v1).
Rule (A3, E22): 10 % of WHOLE episodes are withheld before any fit, drawn with a fixed seed (101) from all
episodes of all envs (an env's last, unfinished episode counts as an episode). Withheld episodes never enter
fitting or imagination starts.

Output (fresh folder):
  segments/        manifest.csv + one CSV per training episode, fitter layout (as e1_prepare_data.py):
                   row t = [next_state[t] (45), action[t] (12), contact[t] (8), termination_observation[t] (1)];
                   a fall episode ends with its true terminal state and termination 1; a time-out ends with 0;
                   episodes shorter than the fitter window (history + forecast) are skipped and counted
  start_data/      state_action_data_{i}.csv, one per env (its training episodes, same row layout, all lengths),
                   with the LAST column set to 1 at every episode end (fall, time-out or recording end): the
                   trainer's start sampler never takes a window across it. start_data.json: rows, files, episodes
  go2_norm.npz     the fitter's own normalizer (compute_normalizer) on the training segments
  withheld.json    the withheld episodes (env, first row, last row, kind)
  manifest.json    counts (episodes, falls, time-outs; withheld, skipped, kept; falls and terminal positives
                   that reach fitting), hashes of every output, the rule
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

MB = Path(__file__).resolve().parent
for p in (str(MB), str(MB / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

STATE, ACTION, CONTACT, TERM = 45, 12, 8, 1


def seq_len_from_config():
    from configs.go2_flat_cfg import Go2FlatConfig
    mac = Go2FlatConfig().model_architecture_config
    return int(mac.history_horizon) + int(mac.forecast_horizon)


def episodes(done):
    """[(env, first, last, finished)] for every episode of every env, env-major, in time order."""
    out = []
    T, N = done.shape
    for n in range(N):
        start = 0
        for end in np.flatnonzero(done[:, n]):
            out.append((n, start, int(end), True))
            start = int(end) + 1
        if start <= T - 1:
            out.append((n, start, T - 1, False))
    return out


def withheld_set(eps, frac, seed):
    k = int(round(frac * len(eps)))
    pick = np.random.default_rng(seed).choice(len(eps), size=k, replace=False) if k else np.zeros(0, int)
    return set(int(i) for i in pick)


def rows_of(trace, n, s, e):
    return np.concatenate([trace["next_state"][s:e + 1, n], trace["action"][s:e + 1, n],
                           trace["contact"][s:e + 1, n],
                           trace["termination_observation"][s:e + 1, n].astype(np.float32)], axis=1).astype(np.float32)


def prepare(trace_path, out, frac=0.1, seed=101):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    with np.load(trace_path, allow_pickle=False) as z:
        meta = validate_trace(z)
        trace = {k: z[k] for k in z.files if k != "metadata"}
    if trace["contact"].shape[-1] != CONTACT or trace["termination_observation"].shape[-1] != TERM:
        raise ValueError("unexpected contact or termination width")
    seq_len = seq_len_from_config()
    term, tout, tob = trace["terminated"], trace["time_out"], trace["termination_observation"]
    eps = episodes(trace["done"])
    held = withheld_set(eps, frac, seed)
    seg_dir, start_dir = out / "segments", out / "start_data"
    seg_dir.mkdir(parents=True)
    start_dir.mkdir()
    kinds = {}
    c = dict(episodes=len(eps), falls=0, time_outs=0, unfinished=0, withheld=0, withheld_falls=0,
             training=0, training_falls=0, skipped_short=0, skipped_short_falls=0, kept_segments=0,
             falls_reaching_fitting=0, terminal_positive_rows_reaching_fitting=0, fit_rows=0, start_rows=0)
    manifest, held_list, per_env = [], [], {n: [] for n in range(trace["done"].shape[1])}
    for i, (n, s, e, finished) in enumerate(eps):
        fall = bool(finished and term[e, n] and not tout[e, n])
        kind = "fall" if fall else ("time_out" if finished else "unfinished")
        kinds[kind] = kinds.get(kind, 0) + 1
        c["falls"] += fall
        c["time_outs"] += kind == "time_out"
        c["unfinished"] += kind == "unfinished"
        if tob[s:e, n].sum() != 0:
            raise ValueError(f"env {n}: termination label before the episode end")
        if finished and bool(tob[e, n, 0] > 0.5) != bool(term[e, n]):
            raise ValueError(f"env {n}: termination label differs from physical termination")
        if i in held:
            c["withheld"] += 1
            c["withheld_falls"] += fall
            held_list.append(dict(env=n, first=s, last=e, kind=kind))
            continue
        c["training"] += 1
        c["training_falls"] += fall
        rows = rows_of(trace, n, s, e)
        start_rows = rows.copy()
        start_rows[:, -1] = 0.0
        start_rows[-1, -1] = 1.0                     # episode end: the start sampler's boundary
        per_env[n].append(start_rows)
        if len(rows) < seq_len:
            c["skipped_short"] += 1
            c["skipped_short_falls"] += fall
            continue
        name = f"s2_env{n:03d}_r{s:05d}"
        np.savetxt(seg_dir / f"{name}.csv", rows, delimiter=",", fmt="%.9g")
        manifest.append((name, len(rows), len(rows), 1))
        c["kept_segments"] += 1
        c["fit_rows"] += len(rows)
        c["falls_reaching_fitting"] += fall
        c["terminal_positive_rows_reaching_fitting"] += int(rows[:, -1].sum())
    if not manifest:
        raise RuntimeError("no training episode is long enough for the fitter window")
    with (seg_dir / "manifest.csv").open("x") as f:
        f.write("name,steps,rows,num_envs\n")
        f.writelines(f"{a},{b},{r},{m}\n" for a, b, r, m in manifest)
    files, min_rows = 0, None
    for n in sorted(per_env):
        if not per_env[n]:
            continue
        block = np.concatenate(per_env[n], axis=0)
        np.savetxt(start_dir / f"state_action_data_{files}.csv", block, delimiter=",", fmt="%.9g")
        files += 1
        c["start_rows"] += len(block)
        min_rows = len(block) if min_rows is None else min(min_rows, len(block))
    start_info = dict(rows=c["start_rows"], files=files, min_file_rows=int(min_rows), episodes=c["training"],
                      layout="next_state 45, action 12, contact 8, episode-end flag 1 (start sampler only)")
    (start_dir / "start_data.json").write_text(json.dumps(start_info, indent=1))
    from fit_world_model_go2_pilot_segments import compute_normalizer, read_segments
    segs, _, _ = read_segments(seg_dir, STATE, ACTION, 0, CONTACT, TERM)
    sm, ss, am, as_ = (t.reshape(-1).numpy() for t in compute_normalizer(segs, STATE, ACTION))
    with (out / "go2_norm.npz").open("xb") as f:
        np.savez(f, state_mean=sm, state_std=ss, action_mean=am, action_std=as_)
    (out / "withheld.json").write_text(json.dumps(held_list, indent=1))
    result = dict(
        rule="plan v6 amendment A3 §2: whole episodes withheld before fitting (E22), "
             f"fraction {frac}, seed {seed}; withheld episodes never enter fitting or imagination starts",
        source=str(trace_path), source_sha256=sha256(trace_path), task=meta.get("task"),
        checkpoint=meta.get("checkpoint"), checkpoint_sha256=meta.get("checkpoint_sha256"),
        record_seed=meta.get("seed"), num_envs=int(trace["done"].shape[1]), num_steps=int(trace["done"].shape[0]),
        seq_len=seq_len, counts=c, kinds=kinds,
        sparse_failure_note=(f"{c['falls']} fall(s) in the recording; {c['falls_reaching_fitting']} reach fitting "
                             f"({c['terminal_positive_rows_reaching_fitting']} terminal positive row(s)); "
                             f"{c['withheld_falls']} withheld, {c['skipped_short_falls']} in skipped short episodes"),
        outputs=dict(segments_manifest_sha256=sha256(seg_dir / "manifest.csv"),
                     start_data_json_sha256=sha256(start_dir / "start_data.json"),
                     go2_norm_sha256=sha256(out / "go2_norm.npz"), withheld_sha256=sha256(out / "withheld.json")),
        code_sha256=sha256(__file__))
    (out / "manifest.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True, help="the nominal s2 round-0 recording")
    p.add_argument("--out", type=Path, required=True, help="fresh output folder")
    p.add_argument("--withhold-fraction", type=float, default=0.1)
    p.add_argument("--withhold-seed", type=int, default=101)
    a = p.parse_args(argv)
    r = prepare(a.trace, a.out, a.withhold_fraction, a.withhold_seed)
    c = r["counts"]
    print(f"[round0] {r['source']} ({r['source_sha256'][:12]}): {c['episodes']} episodes ({c['falls']} falls, "
          f"{c['time_outs']} time-outs, {c['unfinished']} unfinished)", flush=True)
    print(f"[round0] withheld {c['withheld']} ({c['withheld_falls']} falls); training {c['training']}; segments "
          f"{c['kept_segments']} ({c['fit_rows']} rows), skipped short {c['skipped_short']}; start rows {c['start_rows']}")
    print(f"[round0] {r['sparse_failure_note']}")
    print(f"[round0] normalizer sha256 {r['outputs']['go2_norm_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
