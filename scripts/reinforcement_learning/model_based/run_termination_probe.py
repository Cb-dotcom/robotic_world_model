#!/usr/bin/env python3
"""Calibrate the imagined termination rule, train ONE ANYmal seed with it, evaluate.

Evidence (27 Oct, logs/p2/transition_fidelity.*): on real falls of the offline policy,
ETH's termination rule (selected head p > 0.5) fires in 6-56% of cases per head, while
the head probabilities separate falls from survival by ~100x. Imagination therefore
almost never ends in a fall (P1.5 log: 2007/2498 iterations at the 256-step cap).

Steps (stop on any error; fresh output folder; one simulator/training job at a time):
 1. analysis/calibrate_termination.py on the diagnostic's baseline.npz / reference.npz
    (selection rule fixed in that file; infeasible -> stop, no training).
 2. Train run 1801: paper recipe, cold, training seed 1, current env (acceleration fix
    d9d2a66 included, as in run 1701), ONLY the termination rule changed.
 3. Evaluate seeds 42/43, 32 envs x 10000 steps, std .5, commands 10 s, pushes on.
 4. Pool and apply the rules fixed below.

Rules fixed 27 Oct before training (control = run 1701, same code path except the rule):
  screen ("promising": train seeds 2 and 3): pooled falls <= 89 (= 0.5 x 178),
          pooled tracking >= .874513 (1701 - .02), speed ratio >= .8 in both evaluations.
  walker bar (plan): pooled falls <= 6, pooled tracking >= .859 (0.9 x pooled ceiling
          .9546), speed ratio >= .8 in both evaluations.
One training seed: an engineering screen, not a causal proof.
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
RUN = 1801
CONTROL = {"run": 1701, "falls": 178, "episodes_completed": 708, "track_per_step": .894513,
           "speed_ratio": .884107, "source": "run-log 2026-10-27, logs/p2/accel1701.37ZhDk/comparison.json"}
CEILING = {"track_per_step_pooled": .9546, "falls": 3,
           "source": "run-log: ceiling s42 10k .9536 3/321; s43 10k .955695 0/320"}
SCREEN = {"max_falls": 89, "min_tracking": .874513, "min_speed": .8}
WALKER = {"max_falls": 6, "min_tracking": .859, "min_speed": .8}


def verdicts(pooled, records):
    speeds = all(r["summary"]["speed_ratio"] >= .8 for r in records)
    screen = pooled["falls"] <= SCREEN["max_falls"] and pooled["track_per_step"] >= SCREEN["min_tracking"] and speeds
    walker = pooled["falls"] <= WALKER["max_falls"] and pooled["track_per_step"] >= WALKER["min_tracking"] and speeds
    return {"screen_pass": bool(screen), "walker_pass": bool(walker), "both_speeds_ok": bool(speeds)}


def run(args, status):
    repo, out = Path.cwd(), args.out
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "scripts", "source"], cwd=repo, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                            text=True).stdout.strip()
    if list((repo / "logs/offline/anymal_d_flat").glob(f"*_{RUN}")):
        raise RuntimeError(f"Run {RUN} already exists; refusing duplicate training")
    diag = args.diagnostic
    for name in ("baseline.npz", "reference.npz"):
        if not (diag / name).is_file():
            raise RuntimeError(f"Missing {diag / name}")
    wm = repo / "assets/models/pretrain_rnn_ens.pt"
    status("calibrating")
    calib = out / "termination_calibration.json"
    proc = subprocess.run([args.python, str(MB / "analysis/calibrate_termination.py"),
                           "--baseline", str(diag / "baseline.npz"), "--reference", str(diag / "reference.npz"),
                           "--wm", str(wm), "--out", str(calib), "--device", args.calib_device],
                          cwd=repo, stdout=(out / "calibration.log").open("x"), stderr=subprocess.STDOUT)
    if proc.returncode == 3:
        raise RuntimeError("No feasible termination rule; no training started")
    if proc.returncode != 0:
        raise RuntimeError(f"Calibration failed (rc={proc.returncode}); see calibration.log")
    chosen = json.loads(calib.read_text())["chosen"]
    rule, threshold = chosen["rule"], float(chosen["threshold"])
    pair.write_json(out / "manifest.json", {
        "run": RUN, "training_seed": 1, "commit": commit, "termination_rule": rule,
        "termination_threshold": threshold, "calibration_row": chosen,
        "control": CONTROL, "ceiling": CEILING, "screen_rule": SCREEN, "walker_rule": WALKER,
        "world_model_sha256": pair.sha256(wm), "diagnostic": str(diag),
        "data_sha256": pair.sha256(repo / "assets/data/state_action_data_0.csv"),
    })
    if pair_gpu_free() < 12:
        raise RuntimeError("Less than 12 GiB free on GPU 0; calibration saved, no training started")
    status("training", run=RUN, rule=rule, threshold=threshold)
    train_log = out / "train.log"
    pair.execute([args.python, str(Path(__file__).resolve()), "--training-child", rule, str(threshold)],
                 train_log, repo)
    new_dir = pair.run_directory(repo, RUN)
    checkpoint = pair.completed(new_dir, train_log)
    record = pair.provenance(new_dir, RUN, "paper", 1)
    if record["args"].get("termination_rule") != rule or float(record["args"].get("termination_threshold")) != threshold:
        raise RuntimeError("Saved run_args do not record the intended termination rule")
    records = []
    for seed in (42, 43):
        if pair_gpu_free() < 12:
            raise RuntimeError("Less than 12 GiB free before evaluation; training saved, no evaluation started")
        status("evaluating", evaluation_seed=seed)
        d = out / f"eval_{seed}"
        d.mkdir()
        pair.execute([args.python, str(pair.EVALUATOR), "--task", pair.TASK, "--checkpoint", str(checkpoint),
                      "--num_envs", "32", "--num_steps", "10000", "--seed", str(seed), "--std", ".5",
                      "--cmd_resample_s", "10", "--headless", "--out_csv", str(d / "episodes.csv")],
                     d / "eval.log", repo)
        rec = json.loads((d / "episodes_summary.json").read_text())
        validate_reference(rec, pair.TASK, checkpoint, seed, 10000, True)
        validate_episode_csv(d / "episodes.csv", rec["summary"])
        records.append(rec)
    pooled = pair.pool(records)
    report = {"run": RUN, "rule": rule, "threshold": threshold, "pooled": pooled,
              "per_seed": [{k: r["summary"][k] for k in ("track_per_step", "falls", "episodes_completed", "speed_ratio")}
                           for r in records],
              "control": CONTROL, "ceiling": CEILING, **verdicts(pooled, records),
              "scope": "One training seed. Screen selects replication (seeds 2, 3); not a causal proof."}
    pair.write_json(out / "comparison.json", report)
    text = (f"run {RUN} ({rule} > {threshold}): tracking={pooled['track_per_step']:.6f}, "
            f"falls={pooled['falls']}/{pooled['episodes_completed']}, speed_ratio={pooled['speed_ratio']:.6f}\n"
            f"control 1701: tracking={CONTROL['track_per_step']}, falls={CONTROL['falls']}/{CONTROL['episodes_completed']}\n"
            f"screen_pass={report['screen_pass']} (falls<=89, tracking>=.874513, speeds>=.8)\n"
            f"walker_pass={report['walker_pass']} (falls<=6, tracking>=.859, speeds>=.8)\n")
    (out / "comparison.txt").write_text(text)
    print(text, flush=True)
    status("complete", comparison=str(out / "comparison.json"))


def pair_gpu_free():
    from run_acceleration_probe import gpu_free_gib
    return gpu_free_gib()


def training_child(rule, threshold):
    import runpy
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.5, 22 * 1024**3 / total), 0)
    script = str(MB / "p15_train.py")
    sys.path.insert(0, str(MB.resolve()))
    sys.argv = [script, "paper", str(RUN), "anymal_d_flat", "--seed", "1",
                "--termination-rule", rule, "--termination-threshold", threshold]
    runpy.run_path(script, run_name="__main__")


def main():
    if sys.argv[1:2] == ["--training-child"]:
        training_child(sys.argv[2], sys.argv[3])
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostic", type=Path, required=True, help="transition_fidelity.* folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default="/isaac-sim/python.sh")
    parser.add_argument("--calib-device", default="cuda:0")
    args = parser.parse_args()
    args.out, args.diagnostic = args.out.resolve(), args.diagnostic.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {"queue.log", "queue.pid"} or not p.is_file() for p in args.out.iterdir()):
        parser.error("Output directory is not fresh")

    def status(state, **details):
        pair.write_json(args.out / "status.json", dict(state=state, utc=datetime.now(timezone.utc).isoformat(), **details))
        print(f"[termination-probe] {state}: {details}", flush=True)
    lock_path = Path("logs/p2/termination_probe.lock")
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
