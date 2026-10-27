#!/usr/bin/env python3
"""Two short saved-policy recordings, then CPU fidelity analysis. No training."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import queue_paired_policy as pair
from run_acceleration_probe import gpu_free_gib, review_baseline_git

RECORDER = Path("scripts/reinforcement_learning/rsl_rl/record_pre_reset_transitions.py")
ANALYSIS = pair.MB / "analysis/analyse_transition_fidelity.py"
EXPECTED_WM = "2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a"


def command(python, checkpoint, out, *, smoke=False):
    return [python, str(RECORDER), "--checkpoint", str(checkpoint), "--seed", "42",
            "--num_envs", "1" if smoke else "16", "--num_steps", "2" if smoke else "5000",
            "--headless", "--out", str(out)]


def run(args, status):
    repo, out = Path.cwd(), args.out
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "scripts", "source"], cwd=repo, check=True)
    wm = repo / "assets/models/pretrain_rnn_ens.pt"
    if pair.sha256(wm) != EXPECTED_WM:
        raise RuntimeError("Unexpected supplied WM/reference checkpoint")
    baseline = pair.run_directory(repo, 1601)
    record = pair.provenance(baseline, 1601, "paper", 1)
    review = review_baseline_git(record["git"])
    checkpoint = baseline / "policy_2499.pt"
    saved = json.loads((args.baseline_pair / "manifest.json").read_text())
    if pair.sha256(checkpoint) != saved["cold"]["checkpoint_sha256"]:
        raise RuntimeError("Baseline policy differs from the saved paired study")
    if record["world_model_checkpoint"]["sha256"] != EXPECTED_WM:
        raise RuntimeError("Baseline was trained with another WM")
    pair.write_json(out / "manifest.json", {
        "baseline_checkpoint": str(checkpoint), "baseline_sha256": pair.sha256(checkpoint),
        "reference_and_wm": str(wm), "reference_and_wm_sha256": EXPECTED_WM,
        "baseline_source_review": review,
        "protocol": "seed42, 16env x 5000steps each, no pushes, commands10s; pre-reset recorder hooks",
        "resource_limit": "One simulator subprocess at a time; >=12GiB free before each. CPU replay uses 2 threads. No process signalled.",
        "new_training": False,
    })
    for arm, ckpt, smoke in (("smoke", checkpoint, True), ("baseline", checkpoint, False), ("reference", wm, False)):
        if gpu_free_gib() < 12:
            raise RuntimeError(f"Less than 12GiB free before {arm}; no new simulator process started")
        status("recording", arm=arm)
        pair.execute(command(args.python, ckpt, out / f"{arm}.npz", smoke=smoke), out / f"{arm}.log", repo)
        # Validate every recording before spending time on the next subprocess.
        sys.path.insert(0, str(pair.MB / "analysis"))
        import numpy as np
        from pre_reset_transitions import validate_trace
        with np.load(out / f"{arm}.npz", allow_pickle=False) as trace:
            validate_trace(trace)
    status("analysing_on_cpu")
    pair.execute([args.python, str(ANALYSIS), "--baseline", str(out / "baseline.npz"),
                  "--reference", str(out / "reference.npz"), "--wm", str(wm), "--out", str(out)],
                 out / "analysis.log", repo)
    if not (out / "report.json").is_file() or not (out / "report.txt").is_file():
        raise RuntimeError("Analysis returned without both report artifacts")
    status("complete", report=str(out / "report.txt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default="/isaac-sim/python.sh")
    args = parser.parse_args()
    args.out, args.baseline_pair = args.out.resolve(), args.baseline_pair.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {"queue.log", "queue.pid"} or not p.is_file() for p in args.out.iterdir()):
        parser.error("Output directory is not fresh; preserve it and use a new directory")
    def status(state, **details):
        pair.write_json(args.out / "status.json", dict(state=state, utc=datetime.now(timezone.utc).isoformat(), **details))
        print(f"[transition-diagnostic] {state}: {details}", flush=True)
    lock_path = Path("logs/p2/transition_diagnostic.lock")
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
