#!/usr/bin/env python3
"""E1 (plan v5.1): convert pre-reset recordings into fitter segments and held-out test traces.

Input: two recordings from rsl_rl/record_pre_reset_transitions.py (schema pre_reset_transitions_v1).
Output (fresh directory):
  segments/manifest.csv + one CSV per episode   training data for fit_world_model_go2_pilot_segments.py
  test_<tag>.npz                               held-out envs, same schema, for e1_model_check.py
  eth_norm.npz                                 AnymalDFlatConfig constants (the ETH model's z-space)
  summary.json                                 counts, split and hashes

Row layout per episode (the post-step observation layout the fitter expects; state row k is
paired with the action of row k+1, see SystemDynamicsEnsemble.compute_state_loss):
  row t = [next_state[t] (45), action[t] (12), contact[t] (8), termination_observation[t] (1)]
The true terminal state is the last row of a fall episode, with termination 1. A time-out
ends with termination 0. A window never crosses an episode because each episode is its own
segment. Episodes shorter than the fitter window are skipped and counted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

MB = Path(__file__).resolve().parent
sys.path.insert(0, str(MB))
sys.path.insert(0, str(MB / "analysis"))
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

STATE, ACTION, CONTACT, TERM = 45, 12, 8, 1


def eth_norm_arrays():
    from configs.anymal_d_flat_cfg import AnymalDFlatConfig
    data = AnymalDFlatConfig().data_config
    out = {
        "state_mean": np.asarray(data.state_data_mean, dtype=np.float32),
        "state_std": np.asarray(data.state_data_std, dtype=np.float32),
        "action_mean": np.asarray(data.action_data_mean, dtype=np.float32),
        "action_std": np.asarray(data.action_data_std, dtype=np.float32),
    }
    assert out["state_mean"].shape == (STATE,) and out["action_std"].shape == (ACTION,)
    return out


def seq_len_from_config():
    from configs.anymal_d_flat_cfg import AnymalDFlatConfig
    mac = AnymalDFlatConfig().model_architecture_config
    return int(mac.history_horizon) + int(mac.forecast_horizon)


def episode_bounds(done_column):
    """(start, end) inclusive per episode; the last one may be truncated by the recording end."""
    start, T = 0, done_column.shape[0]
    for end in [*np.flatnonzero(done_column), T - 1]:
        end = int(end)
        if end < start:
            continue
        yield start, end
        start = end + 1


def convert(trace, tag, train_envs, seg_dir, seq_len, manifest_rows):
    nxt, act = trace["next_state"], trace["action"]
    con, tob = trace["contact"], trace["termination_observation"]
    term, tout = trace["terminated"], trace["time_out"]
    if con.shape[-1] != CONTACT or tob.shape[-1] != TERM:
        raise ValueError("Unexpected contact/termination width")
    stats = {"episodes": 0, "kept": 0, "skipped_short": 0, "falls_kept": 0, "falls_skipped_short": 0,
             "timeouts_kept": 0, "truncated_kept": 0, "rows": 0}
    for n in train_envs:
        for k, (s, e) in enumerate(episode_bounds(trace["done"][:, n])):
            stats["episodes"] += 1
            length = e - s + 1
            is_fall = bool(term[e, n] and not tout[e, n])
            if tob[s:e, n].sum() != 0:
                raise ValueError(f"{tag} env {n}: termination label before the episode end")
            if bool(tob[e, n, 0] > 0.5) != bool(term[e, n]):
                raise ValueError(f"{tag} env {n}: termination label differs from physical termination")
            if length < seq_len:
                stats["skipped_short"] += 1
                stats["falls_skipped_short"] += int(is_fall)
                continue
            rows = np.concatenate([nxt[s:e + 1, n], act[s:e + 1, n], con[s:e + 1, n],
                                   tob[s:e + 1, n].astype(np.float32)], axis=1).astype(np.float32)
            name = f"{tag}_env{n:03d}_ep{k:03d}"
            np.savetxt(seg_dir / f"{name}.csv", rows, delimiter=",", fmt="%.9g")
            manifest_rows.append((name, length, length, 1))
            stats["kept"] += 1
            stats["rows"] += length
            stats["falls_kept"] += int(is_fall)
            stats["timeouts_kept"] += int(bool(tout[e, n]))
            stats["truncated_kept"] += int(not trace["done"][e, n])
    return stats


def write_test_slice(trace, test_envs, path, source_sha):
    meta = json.loads(str(trace["metadata"]))
    arrays = {key: np.ascontiguousarray(trace[key][:, test_envs]) for key in trace.files if key != "metadata"}
    meta.update(num_envs=len(test_envs), verified_terminal_rows=int(arrays["done"].sum()),
                e1_test_slice_of_sha256=source_sha, e1_test_envs=[int(n) for n in test_envs])
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata=json.dumps(meta, allow_nan=False))
    with np.load(path, allow_pickle=False) as check:
        validate_trace(check)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True, help="recording of the policy that falls (1601)")
    p.add_argument("--reference", type=Path, required=True, help="recording of the reference policy")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--test-fraction", type=float, default=0.1875, help="last envs held out (12 of 64)")
    args = p.parse_args(argv)
    if args.out.exists():
        p.error("--out must not exist")
    seq_len = seq_len_from_config()
    seg_dir = args.out / "segments"
    seg_dir.mkdir(parents=True)
    manifest_rows, summary = [], {"seq_len": seq_len, "inputs": {}}
    for tag, path in (("baseline1601", args.baseline), ("reference", args.reference)):
        with np.load(path, allow_pickle=False) as trace:
            meta = validate_trace(trace)
            N = trace["done"].shape[1]
            n_test = max(1, int(round(N * args.test_fraction)))
            train_envs, test_envs = list(range(N - n_test)), list(range(N - n_test, N))
            source_sha = sha256(path)
            stats = convert(trace, tag, train_envs, seg_dir, seq_len, manifest_rows)
            write_test_slice(trace, test_envs, args.out / f"test_{tag}.npz", source_sha)
            summary["inputs"][tag] = {"path": str(path), "sha256": source_sha, "num_envs": N,
                                      "num_steps": int(trace["done"].shape[0]), "seed": meta.get("seed"),
                                      "checkpoint_sha256": meta.get("checkpoint_sha256"),
                                      "train_envs": [train_envs[0], train_envs[-1]],
                                      "test_envs": [test_envs[0], test_envs[-1]], "train": stats}
    if not manifest_rows:
        raise RuntimeError("No episode is long enough for the fitter window")
    with (seg_dir / "manifest.csv").open("x") as f:
        f.write("name,steps,rows,num_envs\n")
        f.writelines(f"{n},{s},{r},{e}\n" for n, s, r, e in manifest_rows)
    norm = eth_norm_arrays()
    with (args.out / "eth_norm.npz").open("xb") as f:
        np.savez(f, **norm)
    summary["eth_norm_sha256"] = sha256(args.out / "eth_norm.npz")
    summary["segments"] = len(manifest_rows)
    summary["train_rows"] = int(sum(r[2] for r in manifest_rows))
    with (args.out / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
