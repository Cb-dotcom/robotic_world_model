# Clean-repro test and lambda sweep (prepared 30 Oct 2026)

**Why.** Chenhao Li (meeting 28 Oct): the repo should work as shipped. Friends trained offline
policies with the repo defaults, without matching the paper, and saw (almost) no falls. Our offline
policies fall often (run 1601: 156/689 episodes). So we test the shipped recipe on an UNMODIFIED
clone of ETH's repo, before any other work.

## What runs (one GPU job at a time, `run_all.sh`)

| Phase | Job | Decides? |
|---|---|---|
| 0 | Versions (`versions.txt`), clean clone at ETH `82147f9` (`setup.txt`), effective config (`config_dryrun.txt`) | provenance |
| 1 | R1, R2: README Option 2 with repo defaults (500 it, lr 1e-4, entropy 1e-4, 8192 envs x 24 steps, lambda 1). Eval std protocol seeds 42/43 | **yes (locked rule)** |
| 1 | R1, R2, ceiling on the Play task (`Isaac-Velocity-Flat-Anymal-D-Play-v0`), seed 42 | reported only |
| 1 | R3 only if R1 and R2 disagree | majority |
| 2 | Lambda sweep: lambda 0, 5, 20 (lambda 1 = R1). Each: std and cmd30 evals, seeds 42/43 | exploratory |

Eval protocols use our evaluator in the fork checkout: 32 envs x 10000 steps, tracking sigma 0.5.
- std: `Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0`, commands every 10 s, pushes on (same as all earlier numbers).
- cmd30: same, commands every 30 s (no command change inside a 20 s episode).
- play: Isaac Lab Play task defaults (what a user who "just plays it" sees).

The clean clone is never edited. `train_clean.py` only (a) tees `wandb.log` to `metrics.jsonl`,
(b) sets lambda in phase 2, (c) restores `torch.load(weights_only=False)` if torch >= 2.6.

## Locked rule, phase 1 (approve before the run; never change afterwards)

Per clean policy, pooled seeds 42+43, std protocol:

| Class | Condition |
|---|---|
| WORKS | falls <= 6 and tracking >= 0.859 and speed ratio >= 0.8 |
| BETTER | not WORKS; falls <= 78 (half of 1601), tracking >= 0.859, speed ratio >= 0.8 |
| SAME | otherwise (tag SLOW if speed ratio < 0.8) |

Overall: R1 and R2 in the same class gives the verdict; else MIXED, train R3, majority decides.

| Verdict | Meaning | Next |
|---|---|---|
| WORKS | Chenhao is right; our fork or our paper settings caused the falls | Use the clean default as our base; find which difference (fork vs settings) |
| BETTER | Part of the problem is on our side; part remains | Same, plus transition analysis on the clean policy |
| SAME | Not settings or fork code. Suspect versions (ETH: Isaac Lab 2.2.1) or evaluation protocol | Compare `versions.txt` with ETH; read the Play results |

## Phase 2 predictions (written before any result)

Blind-uncertainty hypothesis H: uncertainty is low exactly at command transitions.
- If H holds: higher lambda lowers speed and imagined uncertainty, and lowers cmd30 falls, but the
  falls from command changes (rate 10 s minus rate 30 s) do not drop in proportion.
- H is contradicted if some lambda makes the falls from command changes about 0 while speed ratio >= 0.8.
- lambda 20 may stop the robot (speed ratio << 0.8): that shows the penalty acts, but globally.
- lambda 0 vs 1 with the same falls: the penalty does nothing at lambda 1.
One training per lambda: exploratory, no claim without more seeds.

## Commands (inside the container)

```bash
cd /workspace/Uncertainty-Aware-Robotic-World-Model-Go2/upstream/robotic_world_model
git fetch origin lab-tools
rm -rf /workspace/labtools && mkdir -p /workspace/labtools
git archive origin/lab-tools lab | tar -x -C /workspace/labtools        # does not touch this checkout
nohup bash /workspace/labtools/lab/clean_repro/run_all.sh > /workspace/clean_repro_run.log 2>&1 &
```

Check: `tail -n 5 $(cat /workspace/clean_repro_latest.txt)/status.txt` and `cat $(cat /workspace/clean_repro_latest.txt)/verdict.txt`.
Results also go to branch `lab-results/clean-repro-live` after every step.
Resume after a stop: `OUT=$(cat /workspace/clean_repro_latest.txt) nohup bash /workspace/labtools/lab/clean_repro/run_all.sh >> /workspace/clean_repro_run.log 2>&1 &`
Phase 1 only: add `SKIP_LAMBDA=1` before `nohup`.
