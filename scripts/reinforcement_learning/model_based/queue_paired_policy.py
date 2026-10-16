#!/usr/bin/env python3
"""Wait for a cold ANYmal run, train its warm pair, then score seeds 42 and 43.

Standalone, standard-library-only entry point. It can be extracted with git show
into logs/ without switching the checkout used by an active training process.
Outputs are fresh, a per-pair file lock prevents duplicate queues, and errors
stop the queue. A final checkpoint alone is not proof of completed training.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


TASK = "Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0"
MB = Path("scripts/reinforcement_learning/model_based")
EVALUATOR = Path("scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py")
NORM_HASH = "3d21c884971bf67492920a600e972462fcc6a5293e2d68e5e3ca8027ee37fc56"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_directory(repo, run):
    paths = sorted((repo / "logs/offline/anymal_d_flat").glob(f"*_{run}"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one directory for run {run}; found {len(paths)}")
    return paths[0]


def provenance(directory, run, mode, seed):
    record = json.loads((directory / "run_args.json").read_text())
    for key, expected in (("run_num", run), ("mode", mode), ("seed", seed),
                          ("task", "anymal_d_flat")):
        if record.get(key) != expected:
            raise RuntimeError(f"Run {run}: expected {key}={expected!r}, got {record.get(key)!r}")
    if record["normalizer_contract"]["norm_stats_sha256"] != NORM_HASH:
        raise RuntimeError(f"Run {run}: unexpected ANYmal normalizer")
    return record


def trainer_alive(pid, run):
    """Check identity as well as PID, so a reused PID cannot block this queue."""
    proc = Path(f"/proc/{pid}")
    try:
        if proc.joinpath("stat").read_text().split(") ", 1)[1].startswith("Z"):
            return False
        words = proc.joinpath("cmdline").read_bytes().decode().split("\0")
    except FileNotFoundError:
        return False
    if not any(word.endswith("p15_train.py") for word in words):
        return False
    return any(words[i:i + 2] == ["paper", str(run)] for i in range(len(words) - 1))


def completed(directory, log):
    checkpoint = directory / "policy_2499.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"Missing final checkpoint: {checkpoint}")
    if "Training completed. Policy saved to" not in log.read_text(errors="replace"):
        raise RuntimeError(f"No training-completion marker in {log}; refusing partial/failed run")
    return checkpoint


def wait_for_cold(args, directory):
    deadline = time.monotonic() + args.wait_hours * 3600
    while trainer_alive(args.cold_pid, args.cold_run):
        if time.monotonic() >= deadline:
            raise RuntimeError("Cold training wait timed out; no warm job was started")
        time.sleep(args.poll_seconds)
    return completed(directory, args.cold_log)


def execute(command, log, repo):
    env = dict(os.environ, WANDB_MODE="offline", PYTHONUNBUFFERED="1")
    with log.open("x") as stream:
        subprocess.run(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def pool(records):
    steps = sum(d["summary"]["counted_steps"] for d in records)
    if steps <= 0:
        raise RuntimeError("No counted evaluation steps")
    cmd = sum(d["summary"]["speed_cmd_per_step"] * d["summary"]["counted_steps"] for d in records)
    if cmd <= 0:
        raise RuntimeError("No commanded evaluation speed")
    result = {
        "counted_steps": steps,
        "falls": sum(d["summary"]["falls"] for d in records),
        "episodes_completed": sum(d["summary"]["episodes_completed"] for d in records),
        "track_per_step": sum(d["summary"]["track_per_step"] * d["summary"]["counted_steps"]
                              for d in records) / steps,
        "speed_ratio": sum(d["summary"]["speed_act_per_step"] * d["summary"]["counted_steps"]
                           for d in records) / cmd,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Non-finite pooled evaluation result")
    return result


def compare(evaluations, args):
    condition_keys = ("task", "num_envs", "num_steps", "std", "step_dt", "episode_length_s",
                      "command_resampling_time_range_s", "push_robot")
    reference = {key: evaluations["cold"][0][key] for key in condition_keys}
    for arm, records in evaluations.items():
        for expected_seed, record in zip((42, 43), records):
            if record["seed"] != expected_seed or {key: record[key] for key in condition_keys} != reference:
                raise RuntimeError(f"Mismatched evaluation settings in {arm}, seed {expected_seed}")
            expected = (record["task"] == TASK and record["num_envs"] == 32
                        and record["num_steps"] == args.num_steps and record["std"] == 0.5
                        and record["command_resampling_time_range_s"] == [10.0, 10.0]
                        and record["push_robot"]["enabled"])
            if not expected or not math.isfinite(record["summary"]["speed_ratio"]):
                raise RuntimeError(f"Invalid evaluation conditions/result in {arm}")
    cold, warm = pool(evaluations["cold"]), pool(evaluations["warm"])
    return {
        "training_seed": args.seed, "evaluation_seeds": [42, 43], "num_steps": args.num_steps,
        "cold": cold, "warm": warm,
        "fewer_falls_with_warm": cold["falls"] - warm["falls"],
        "tracking_gain_with_warm": warm["track_per_step"] - cold["track_per_step"],
        "fall_win": cold["falls"] - warm["falls"] >= 3,
        "tracking_win": warm["track_per_step"] - cold["track_per_step"] >= 0.02,
        "all_warm_speed_ratios_ge_0_8": all(d["summary"]["speed_ratio"] >= 0.8
                                           for d in evaluations["warm"]),
        "scope": "One training-seed pair. Step 2 requires all three pairs; this is not a whole-study verdict.",
    }


def queue(args, status):
    repo, out = args.repo_root, args.out
    cold_dir = run_directory(repo, args.cold_run)
    cold_record = provenance(cold_dir, args.cold_run, "paper", args.seed)
    if list((repo / "logs/offline/anymal_d_flat").glob(f"*_{args.warm_run}")):
        raise RuntimeError(f"Warm run {args.warm_run} already exists; refusing duplicate training")
    wm_path = Path(cold_record["world_model_checkpoint"]["path"])
    if sha256(wm_path) != cold_record["world_model_checkpoint"]["sha256"]:
        raise RuntimeError("World-model bytes changed since cold training started")
    status("waiting_for_cold", cold_run=args.cold_run, cold_pid=args.cold_pid)
    cold_checkpoint = wait_for_cold(args, cold_dir)
    cold_record = provenance(cold_dir, args.cold_run, "paper", args.seed)
    if list((repo / "logs/offline/anymal_d_flat").glob(f"*_{args.warm_run}")):
        raise RuntimeError(f"Warm run {args.warm_run} appeared while waiting; refusing duplicate training")
    if sha256(wm_path) != cold_record["world_model_checkpoint"]["sha256"]:
        raise RuntimeError("World-model bytes changed while waiting for cold training")
    status("training_warm", warm_run=args.warm_run)
    warm_log = out / "warm_train.log"
    execute([args.python, str(MB / "p15_train.py"), "paper_warm", str(args.warm_run),
             "anymal_d_flat", "--seed", str(args.seed)], warm_log, repo)
    warm_dir = run_directory(repo, args.warm_run)
    warm_checkpoint = completed(warm_dir, warm_log)
    warm_record = provenance(warm_dir, args.warm_run, "paper_warm", args.seed)
    for key in ("resolved_config", "normalizer_contract", "world_model_checkpoint", "initial_policy_checkpoint"):
        if cold_record[key] != warm_record[key]:
            raise RuntimeError(f"Cold/warm training mismatch: {key}")
    manifest = {"training_seed": args.seed, "queue_sha256": sha256(__file__),
                "cold": {"run": args.cold_run, "checkpoint": str(cold_checkpoint),
                         "checkpoint_sha256": sha256(cold_checkpoint), "run_args": cold_record},
                "warm": {"run": args.warm_run, "checkpoint": str(warm_checkpoint),
                         "checkpoint_sha256": sha256(warm_checkpoint), "run_args": warm_record}}
    write_json(out / "manifest.json", manifest)
    evaluations = {"cold": [], "warm": []}
    for arm, checkpoint in (("cold", cold_checkpoint), ("warm", warm_checkpoint)):
        for seed in (42, 43):
            status("evaluating", arm=arm, evaluation_seed=seed)
            directory = out / f"{arm}_eval_{seed}"
            directory.mkdir()
            execute([args.python, str(EVALUATOR), "--task", TASK, "--checkpoint", str(checkpoint),
                     "--num_envs", "32", "--num_steps", str(args.num_steps), "--seed", str(seed),
                     "--std", "0.5", "--cmd_resample_s", "10", "--headless",
                     "--out_csv", str(directory / "episodes.csv")], directory / "eval.log", repo)
            record = json.loads((directory / "episodes_summary.json").read_text())
            if Path(record["checkpoint"]).resolve() != checkpoint.resolve():
                raise RuntimeError("Evaluator used an unexpected checkpoint")
            evaluations[arm].append(record)
    result = compare(evaluations, args)
    write_json(out / "comparison.json", result)
    lines = ["Paired ANYmal policy test", f"Training seed: {args.seed}; evaluation seeds: 42, 43"]
    for arm in ("cold", "warm"):
        s = result[arm]
        lines.append(f"{arm}: tracking={s['track_per_step']:.6f}, falls={s['falls']}/{s['episodes_completed']}, "
                     f"speed_ratio={s['speed_ratio']:.6f}")
    lines.extend([f"fall_win={result['fall_win']}; tracking_win={result['tracking_win']}; "
                  f"warm_speed_ok={result['all_warm_speed_ratios_ge_0_8']}", result["scope"]])
    (out / "comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    status("complete", comparison=str(out / "comparison.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True, help="Fresh directory; no outputs are reused")
    parser.add_argument("--cold-pid", type=int, required=True)
    parser.add_argument("--cold-log", type=Path, default=Path("logs/p2/paper_1601_seed1.log"))
    parser.add_argument("--cold-run", type=int, default=1601)
    parser.add_argument("--warm-run", type=int, default=1602)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--python", default="/isaac-sim/python.sh")
    parser.add_argument("--wait-hours", type=float, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if (args.cold_pid <= 0 or args.cold_run == args.warm_run or not 0 <= args.seed <= 2**32 - 1
            or args.num_steps <= 0 or args.wait_hours <= 0 or args.poll_seconds <= 0):
        parser.error("Invalid PID, run IDs, seed, horizon, or wait duration")
    args.repo_root = args.repo_root.resolve()
    args.out = args.out.resolve()
    args.cold_log = args.cold_log.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    # The caller may create the directory for the nohup console log only.
    if any(p.name != "queue.log" for p in args.out.iterdir()):
        parser.error("Output directory is not fresh; use a new one")

    def status(state, **details):
        write_json(args.out / "status.json", {
            "state": state, "utc": datetime.now(timezone.utc).isoformat(), **details})
        print(f"[queue] {state}: {details}", flush=True)

    lock_path = args.repo_root / f"logs/p2/warm_run_{args.warm_run}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue(args, status)
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[queue] FAILED: {exc}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
