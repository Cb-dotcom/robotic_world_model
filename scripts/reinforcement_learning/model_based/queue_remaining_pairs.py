#!/usr/bin/env python3
"""Finish the three-seed ANYmal study after the existing seed-1 queue.

Download this next to queue_paired_policy.py in ignored logs/. No checkout
switch is needed. All GPU jobs are sequential. No extra hyperparameter sweep
or Go2 training is launched. Missing optional reference checkpoints are logged.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import queue_paired_policy as pair


def alive_queue(pid):
    try:
        proc = Path(f"/proc/{pid}")
        if proc.joinpath("stat").read_text().split(") ", 1)[1].startswith("Z"):
            return False
        words = proc.joinpath("cmdline").read_bytes().decode().split("\0")
        return any(word.endswith("queue_paired_policy.py") for word in words)
    except (FileNotFoundError, ProcessLookupError):
        return False


def wait_for_seed1(args, status):
    deadline = time.monotonic() + args.wait_hours * 3600
    while True:
        state = json.loads((args.seed1_dir / "status.json").read_text())
        if state["state"] == "complete":
            result = json.loads((args.seed1_dir / "comparison.json").read_text())
            validate_result(result, 1, args.num_steps)
            return result
        if state["state"] == "failed":
            raise RuntimeError(f"Seed-1 queue failed: {state.get('error')}; inspect {args.seed1_dir}")
        pid = int(args.seed1_pid_file.read_text().strip())
        if not alive_queue(pid):
            # The child can finish between the status read and the PID check.
            fresh = json.loads((args.seed1_dir / "status.json").read_text())
            if fresh["state"] == "complete":
                continue
            if fresh["state"] == "failed":
                raise RuntimeError(f"Seed-1 queue failed: {fresh.get('error')}; inspect {args.seed1_dir}")
            raise RuntimeError("Seed-1 queue is not running and has no complete result")
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for seed-1 queue")
        time.sleep(args.poll_seconds)


def validate_result(result, seed, steps):
    if (result["training_seed"] != seed or result["evaluation_seeds"] != [42, 43]
            or result["num_steps"] != steps):
        raise RuntimeError(f"Unexpected result identity/exposure for training seed {seed}")


def json_safe(value):
    """Undefined fall-only diagnostics are normal when a reference never falls."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def report(out, results):
    fall_wins = sum(r["fall_win"] for r in results)
    track_wins = sum(r["tracking_win"] for r in results)
    speed_ok = all(r["all_warm_speed_ratios_ge_0_8"] for r in results)
    verdict = (fall_wins == 3 and track_wins >= 2 and speed_ok) if len(results) == 3 else None
    summary = {"pairs_completed": len(results), "pairs": results,
               "fall_wins": fall_wins, "tracking_wins": track_wins,
               "all_warm_speed_ok": speed_ok, "step2_pass": verdict,
               "rule": "fall wins 3/3; tracking wins at least 2/3; warm speed ratio >= .8 in every evaluation"}
    pair.write_json(out / "all_pairs.json", summary)
    lines = [f"Step 2: {len(results)}/3 training-seed pairs completed"]
    for r in results:
        lines.append(f"seed {r['training_seed']}: cold falls={r['cold']['falls']}, "
                     f"warm falls={r['warm']['falls']}; fall_win={r['fall_win']}, "
                     f"tracking_gain={r['tracking_gain_with_warm']:.6f}, tracking_win={r['tracking_win']}")
    lines += [f"fall wins={fall_wins}/3; tracking wins={track_wins}/3; warm_speed_ok={speed_ok}",
              f"Step 2 verdict: {'PENDING' if verdict is None else 'PASS' if verdict else 'FAIL'}"]
    (out / "all_pairs.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def references(args, status):
    root = Path("logs/rsl_rl/unitree_go2_flat")
    jobs = [
        ("ceiling_43", pair.TASK, Path("assets/models/pretrain_rnn_ens.pt"), 43),
        ("go2_jun12", "Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0",
         root / "2026-06-12_13-39-03_pretrain_ens5/model_2000.pt", 42),
        ("go2_jun17", "Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0",
         root / "2026-06-17_22-09-04_pretrain_ens5/model_2000.pt", 42),
    ]
    records = []
    for name, task, relative, seed in jobs:
        checkpoint = args.repo_root / relative
        item = {"name": name, "checkpoint": str(checkpoint)}
        if not checkpoint.is_file():
            item["state"] = "missing"
        else:
            status("reference_evaluation", reference=name)
            directory = args.out / name
            directory.mkdir()
            try:
                item["checkpoint_sha256"] = pair.sha256(checkpoint)
                pair.execute([args.python, str(pair.EVALUATOR), "--task", task,
                              "--checkpoint", str(checkpoint), "--num_envs", "32",
                              "--num_steps", str(args.num_steps), "--seed", str(seed),
                              "--std", "0.5", "--cmd_resample_s", "10", "--headless",
                              "--out_csv", str(directory / "episodes.csv")],
                             directory / "eval.log", args.repo_root)
                d = json.loads((directory / "episodes_summary.json").read_text())
                if (d["task"] != task or d["seed"] != seed or d["num_steps"] != args.num_steps
                        or d["num_envs"] != 32 or d["std"] != .5
                        or d["command_resampling_time_range_s"] != [10., 10.]
                        or not d["push_robot"]["enabled"]
                        or Path(d["checkpoint"]).resolve() != checkpoint.resolve()):
                    raise RuntimeError("Unexpected reference evaluation settings")
                if not all(math.isfinite(d["summary"][key]) for key in ("track_per_step", "speed_ratio")):
                    raise RuntimeError("Non-finite primary reference metrics")
                item.update(state="complete", summary=json_safe(d["summary"]))
            except Exception as exc:
                item.update(state="failed", error=str(exc), log=str(directory / "eval.log"))
        records.append(item)
        pair.write_json(args.out / "reference_status.json", records)
        print(f"[references] {name}: {item['state']}", flush=True)
    return records


def run(args, status):
    base = args.repo_root / "logs/offline/anymal_d_flat"
    for run_id in (1603, 1604, 1605, 1606):
        if list(base.glob(f"*_{run_id}")):
            raise RuntimeError(f"Run {run_id} already exists; refusing duplicate training")
    status("waiting_for_seed1", source=str(args.seed1_dir))
    results = [wait_for_seed1(args, status)]
    report(args.out, results)
    for seed, cold_run, warm_run in ((2, 1603, 1604), (3, 1605, 1606)):
        for run_id in (cold_run, warm_run):
            if list(base.glob(f"*_{run_id}")):
                raise RuntimeError(f"Run {run_id} appeared while waiting; refusing duplicate training")
        cold_log = args.out / f"cold_{cold_run}_seed{seed}.log"
        with cold_log.open("x") as stream:
            env = dict(os.environ, WANDB_MODE="offline", PYTHONUNBUFFERED="1")
            command = [args.python, str(pair.MB / "p15_train.py"), "paper", str(cold_run),
                       "anymal_d_flat", "--seed", str(seed)]
            with subprocess.Popen(command, cwd=args.repo_root, env=env,
                                  stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT) as child:
                status("training_cold", seed=seed, run=cold_run, pid=child.pid, log=str(cold_log))
                returncode = child.wait()
                if returncode:
                    raise RuntimeError(f"Cold run {cold_run} exited with code {returncode}; inspect {cold_log}")
                cold_pid = child.pid
        output = args.out / f"seed{seed}_pair"
        status("paired_queue", seed=seed, pair_status=str(output / "status.json"))
        rc = pair.main(["--repo-root", str(args.repo_root), "--out", str(output),
                        "--cold-pid", str(cold_pid), "--cold-log", str(cold_log),
                        "--cold-run", str(cold_run), "--warm-run", str(warm_run),
                        "--seed", str(seed), "--num-steps", str(args.num_steps), "--python", args.python])
        if rc:
            raise RuntimeError(f"Seed-{seed} pair stopped; inspect {output / 'status.json'}")
        result = json.loads((output / "comparison.json").read_text())
        validate_result(result, seed, args.num_steps)
        results.append(result)
        report(args.out, results)
    if args.skip_references:
        status("complete", pairs=3, references="not_requested")
    else:
        refs = references(args, status)
        state = "complete" if all(d["state"] == "complete" for d in refs) else "complete_with_reference_issues"
        status(state, pairs=3, report=str(args.out / "all_pairs.txt"),
               references=str(args.out / "reference_status.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed1-dir", type=Path, required=True)
    parser.add_argument("--seed1-pid-file", type=Path, default=Path("logs/p2/seed1_queue.pid"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default="/isaac-sim/python.sh")
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--wait-hours", type=float, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--skip-references", action="store_true")
    args = parser.parse_args(argv)
    if args.num_steps <= 0 or args.wait_hours <= 0 or args.poll_seconds <= 0:
        parser.error("Horizon and wait durations must be positive")
    for name in ("repo_root", "seed1_dir", "seed1_pid_file", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    if any(p.name != "queue.log" for p in args.out.iterdir()):
        parser.error("Output directory is not fresh")

    def status(state, **details):
        pair.write_json(args.out / "status.json", {
            "state": state, "utc": datetime.now(timezone.utc).isoformat(), **details})
        print(f"[remaining] {state}: {details}", flush=True)

    lock_path = args.repo_root / "logs/p2/remaining_pairs.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args, status)
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[remaining] FAILED: {exc}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
