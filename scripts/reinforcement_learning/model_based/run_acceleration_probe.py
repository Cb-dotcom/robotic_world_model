#!/usr/bin/env python3
"""One cold ANYmal seed-1 training, then two evaluations against saved run 1601.

This measures only the acceleration-slice correction. It is not a full reward
fidelity certificate. The parent uses the standard library only. No existing
training, checkpoint, evaluation, or process is modified.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import queue_paired_policy as pair
from queue_remaining_pairs import validate_episode_csv, validate_reference

MB = pair.MB
RUN = 1701
BASELINE_COMMIT = "0cebd87d4386b42ba23e6449d7ce3402b792c2a5"


def review_baseline_git(git):
    if git.get("commit") != BASELINE_COMMIT:
        raise RuntimeError("Unexpected baseline source commit")
    status = git.get("status", [])
    if status == ["?? assets/models/vel_trace.npz"]:
        # Reviewed workstation evidence: this diagnostic file is not read by
        # p15_train/train, their configs, reward envs or normalization loader.
        return "Reviewed untracked diagnostic vel_trace.npz; training inputs unchanged"
    if status == [] and git.get("dirty") is False:
        return "Recorded baseline working tree clean"
    raise RuntimeError(f"Unreviewed baseline working-tree changes: {status!r}")


def gpu_free_gib():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip()) / 1024


def check_training_match(old, new):
    for key in ("seed", "task", "mode", "resolved_config", "normalizer_contract",
                "world_model_checkpoint", "initial_policy_checkpoint"):
        if old[key] != new[key]:
            raise RuntimeError(f"Baseline/corrected training mismatch: {key}")


def result(old_records, new_records):
    keys = ("task", "num_envs", "num_steps", "std", "step_dt", "episode_length_s",
            "command_resampling_time_range_s", "push_robot")
    for old, new in zip(old_records, new_records):
        if any(old[k] != new[k] for k in keys) or old["seed"] != new["seed"]:
            raise RuntimeError("Baseline/corrected evaluation conditions differ")
    old, new = pair.pool(old_records), pair.pool(new_records)
    screen = (new["falls"] <= .8 * old["falls"]
              and new["track_per_step"] >= old["track_per_step"] - .01
              and all(r["summary"]["speed_ratio"] >= .8 for r in new_records))
    return dict(baseline=old, corrected=new, fewer_falls=old["falls"] - new["falls"],
                tracking_gain=new["track_per_step"] - old["track_per_step"],
                engineering_screen_pass=screen,
                rule="At least 20% fewer falls, tracking loss <= .01, speed >= .8 in both evals.",
                scope="One training seed. This selects replication, not a walker or causal proof.")


def read_eval(directory, checkpoint, seed):
    record = json.loads((directory / "episodes_summary.json").read_text())
    validate_reference(record, pair.TASK, checkpoint, seed, 10000, True)
    validate_episode_csv(directory / "episodes.csv", record["summary"])
    return record


def check_source(repo):
    # Only the acceleration fix may differ along the training execution path.
    paths = [str(MB / p) for p in ("p15_train.py", "train.py", "policy_training.py",
             "envs", "configs", "normalization_contract.py", "run_provenance.py")]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=repo, check=True)
    changed = subprocess.run(["git", "diff", "--name-only", "0cebd87", "HEAD", "--", *paths],
                             cwd=repo, check=True, capture_output=True, text=True).stdout.splitlines()
    expected = str(MB / "envs/anymal_d_flat.py")
    if changed != [expected]:
        raise RuntimeError(f"Unexpected training source changes: {changed}")
    subprocess.run(["git", "diff", "--exit-code", "d9d2a66", "HEAD", "--", expected],
                   cwd=repo, check=True)
    subprocess.run(["git", "diff", "--exit-code", "0cebd87", "--", "assets/data"],
                   cwd=repo, check=True)


def run(args, status):
    repo, out = Path.cwd(), args.out
    check_source(repo)
    baseline_dir = pair.run_directory(repo, 1601)
    baseline = pair.provenance(baseline_dir, 1601, "paper", 1)
    baseline_source_review = review_baseline_git(baseline["git"])
    if list((repo / "logs/offline/anymal_d_flat").glob(f"*_{RUN}")):
        raise RuntimeError(f"Run {RUN} already exists; refusing duplicate training")
    wm = Path(baseline["world_model_checkpoint"]["path"])
    if pair.sha256(wm) != baseline["world_model_checkpoint"]["sha256"]:
        raise RuntimeError("World model changed since baseline training")
    old_checkpoint = baseline_dir / "policy_2499.pt"
    manifest = json.loads((args.baseline_pair / "manifest.json").read_text())
    if pair.sha256(old_checkpoint) != manifest["cold"]["checkpoint_sha256"]:
        raise RuntimeError("Baseline policy changed since saved evaluation")
    old_records = [read_eval(args.baseline_pair / f"cold_eval_{s}", old_checkpoint, s) for s in (42, 43)]
    if gpu_free_gib() < 12:
        raise RuntimeError("Less than 12 GiB free on GPU 0; no job started")
    status("training", run=RUN, training_seed=1, allocator_cap_gib=22)
    train_log = out / "train.log"
    pair.execute([args.python, str(Path(__file__).resolve()), "--training-child"], train_log, repo)
    new_dir = pair.run_directory(repo, RUN)
    new_checkpoint = pair.completed(new_dir, train_log)
    corrected = pair.provenance(new_dir, RUN, "paper", 1)
    check_training_match(baseline, corrected)
    pair.write_json(out / "manifest.json", {
        "baseline": baseline, "corrected": corrected,
        "baseline_source_review": baseline_source_review,
        "policy_sha256": pair.sha256(new_checkpoint),
        "data_sha256": pair.sha256(repo / "assets/data/state_action_data_0.csv"),
        "baseline_policy_sha256": pair.sha256(old_checkpoint),
        "note": "Original baseline data bytes were not hashed at training time; current data match tracked baseline source.",
        "resource_limit": "One job at a time; 22 GiB PyTorch allocator cap on training only. Not a compute quota or simulator-memory cap.",
    })
    new_records = []
    for seed in (42, 43):
        if gpu_free_gib() < 12:
            raise RuntimeError("Less than 12 GiB free before evaluation; training is saved, no evaluation started")
        status("evaluating", evaluation_seed=seed)
        directory = out / f"corrected_eval_{seed}"
        directory.mkdir()
        pair.execute([args.python, str(pair.EVALUATOR), "--task", pair.TASK,
                      "--checkpoint", str(new_checkpoint), "--num_envs", "32", "--num_steps", "10000",
                      "--seed", str(seed), "--std", ".5", "--cmd_resample_s", "10", "--headless",
                      "--out_csv", str(directory / "episodes.csv")], directory / "eval.log", repo)
        new_records.append(read_eval(directory, new_checkpoint, seed))
    report = result(old_records, new_records)
    pair.write_json(out / "comparison.json", report)
    lines = []
    for name in ("baseline", "corrected"):
        r = report[name]
        lines.append(f"{name}: tracking={r['track_per_step']:.6f}, falls={r['falls']}/{r['episodes_completed']}, speed_ratio={r['speed_ratio']:.6f}")
    lines += [f"engineering_screen_pass={report['engineering_screen_pass']}", report["rule"], report["scope"]]
    (out / "comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    status("complete", comparison=str(out / "comparison.json"))


def training_child():
    import runpy
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.5, 22 * 1024**3 / total), 0)
    script = str(MB / "p15_train.py")
    sys.path.insert(0, str(MB.resolve()))
    sys.argv = [script, "paper", str(RUN), "anymal_d_flat", "--seed", "1"]
    runpy.run_path(script, run_name="__main__")


def main():
    if sys.argv[1:] == ["--training-child"]:
        training_child()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default="/isaac-sim/python.sh")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.baseline_pair = args.baseline_pair.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    # The nohup launcher creates these two files before this process starts.
    if any(p.name not in {"queue.log", "queue.pid"} or not p.is_file() for p in args.out.iterdir()):
        parser.error("Output directory is not fresh")
    def status(state, **details):
        pair.write_json(args.out / "status.json", dict(state=state, utc=datetime.now(timezone.utc).isoformat(), **details))
        print(f"[acceleration-probe] {state}: {details}", flush=True)
    lock_path = Path("logs/p2/acceleration_probe.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args, status)
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
