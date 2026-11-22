#!/bin/bash
# A4 D3 (22 Nov 2026), the milestone: one short update from s2 in the qualified Go2 model, scored in Isaac Lab under
# the frozen ground-friction shift against unchanged s2 (A4 §1; gate: analysis/go2_d3_gate.py).
# Model: option-1 D2 model (741b352f...), warm resets (Q1 reset type), lambda 1, s2 reward contract, pooled D2 start data,
# PPO initialized from s2 (new optimizer), 300 iterations, PPO seed 1000, run 4001.
# Evaluation: s2's G6 protocol at 10 s commands, seeds 42 + 43, friction x0.30 read back in the simulator, for the D3
# policy and for unchanged s2. Development evidence (A4); A3 stays closed.
# Usage, from the robotic_world_model root in the container:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_dev_go2_d3.sh logs/p2/a4_d3_1122 > /tmp/a4_d3.out 2>&1 < /dev/null &
set -uo pipefail
OUT=${1:?out dir}
MODE=${2:-run}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
EVAL=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0
D2=logs/p2/a4_d2_1121
WM=$D2/option1/run/d2s_wm_seed2.pt
WM_SHA=${WM_SHA:-741b352fab202f0b898d23908ccdc7f3453409492f970feacc145476cdd16d11}   # run-log 22 Nov (option 1)
START=$D2/data/start_data
E16=${E16:-logs/p2/v6w2_rest_1118/e16_s2_check.npz}
ROUND0=${ROUND0:-logs/p2/go2_round0_1118/round0_s2_seed101.npz}
FRICTION=0.30; ITERS=300; PPO_SEED=1000; RUN=4001
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
case "$MODE" in
  run)
    [ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
    mkdir -p "$OUT/eval" || exit 1 ;;
  resume-eval)
    grep -Fq "FAILED: train.log lacks 'Learning iteration 299/'" "$OUT/queue.log" 2>/dev/null \
      || { echo "No old iteration-guard failure in $OUT/queue.log"; exit 2; }
    [ -d "$OUT/eval" ] && [ -z "$(ls -A "$OUT/eval")" ] \
      || { echo "Expected an empty evaluation directory in $OUT"; exit 2; }
    mkdir "$OUT/resume-eval.lock" \
      || { echo "Evaluation recovery is already in progress"; exit 2; }
    trap 'rmdir "$OUT/resume-eval.lock"' EXIT ;;
  *) echo "Mode must be run or resume-eval"; exit 2 ;;
esac
status() { echo "$(date -u +%FT%TZ) [a4-d3] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
gpu_wait() { until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do
  status "waiting for $(( $1 / 1024 )) GiB GPU before $2"; sleep 300; done; }
for f in "$WM" "$E16" "$ROUND0" "$START/start_data.json"; do [ -s "$f" ] || fail "missing $f"; done
[ "$(sha256sum "$WM" | cut -d' ' -f1)" = "$WM_SHA" ] || fail "model hash differs from the run-log"
CK=$(CUDA_VISIBLE_DEVICES="" "$PY" -c "import json, sys, numpy as np; print(json.loads(str(np.load(sys.argv[1])['metadata']))['checkpoint'])" "$ROUND0" 2>/dev/null | tail -1) \
  || fail "could not read s2 checkpoint from the round-0 trace"
[ -s "$CK" ] || fail "s2 checkpoint not found: $CK"
S2_SHA=$(sha256sum "$CK" | cut -d' ' -f1) || fail "could not hash s2"
START_ITER=$(CUDA_VISIBLE_DEVICES="" "$PY" - "$ROUND0" "$CK" "$S2_SHA" <<'CHECK_S2' | tail -1
import json
import sys
import numpy as np
import torch

trace, checkpoint, sha = sys.argv[1:]
with np.load(trace, allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata"]))
if metadata["checkpoint_sha256"] != sha:
    raise SystemExit("s2 checkpoint hash differs from the round-0 trace")
print(int(torch.load(checkpoint, map_location="cpu", weights_only=False)["iter"]))
CHECK_S2
) || fail "could not verify s2 hash and starting iteration"
[[ "$START_ITER" =~ ^[0-9]+$ ]] || fail "invalid s2 iteration: $START_ITER"

if [ "$MODE" = run ]; then
  { git rev-parse HEAD; git status --short | grep -v '^??'; sha256sum $AN/go2_d3_gate.py; } > "$OUT/git.txt"
  CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_d3_gate.py $AN/tests/test_go2_g0.py > "$OUT/tests.log" 2>&1 || fail "tests (tests.log)"
  grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log)"
  status "tests: $(tail -1 "$OUT/tests.log"); s2 $CK (${S2_SHA:0:12})"

  # ---- 1. one short update from s2 in the qualified model
  gpu_wait 12288 "training"
  status "train: run $RUN, paper_warm, $ITERS iterations from s2, PPO seed $PPO_SEED, model ${WM_SHA:0:8}"
  CUDA_VISIBLE_DEVICES=0 $PY $MB/e1_train.py --wm-checkpoint "$WM" --penalty-weight -1.0 --reward-contract s2 \
    --asset-trace "$E16" --start-data "$START" --init-policy-checkpoint "$CK" --max-iterations $ITERS \
    paper_warm $RUN go2_flat --seed $PPO_SEED > "$OUT/train.log" 2>&1 || fail "training (train.log)"
fi
for pat in "\[go2\] reward contract s2: 16 terms" "\[go2\] start data" "\[dev\] max_iterations = $ITERS" "warm_start_resets= True"; do
  grep -q "$pat" "$OUT/train.log" || fail "train.log lacks '$pat'"
done
P=$(CUDA_VISIBLE_DEVICES="" "$PY" - "$OUT/train.log" "$RUN" "$PPO_SEED" "$START_ITER" "$ITERS" "$WM_SHA" "$S2_SHA" <<'CHECK_TRAINING' | tail -1
import json
import re
import sys
from pathlib import Path

path, run, seed, start, length, wm_sha, s2_sha = sys.argv[1:]
start, length = int(start), int(length)
log = Path(path).read_text()
end = start + length
iterations = [(int(i), int(total)) for i, total in re.findall(r"Learning iteration (\d+)/(\d+)", log)]
if iterations != [(i, end) for i in range(start, end)]:
    raise SystemExit(f"expected {length} PPO iterations from {start} to {end - 1}")
completed = re.findall(r"^Training completed\. Policy saved to (logs/offline/go2_flat/[^/]+)\.$", log, re.M)
if len(completed) != 1 or not completed[0].endswith(f"_{run}"):
    raise SystemExit(f"missing unique completion record for run {run}")
run_dir = Path(completed[0])
args = json.loads((run_dir / "run_args.json").read_text())
if (str(args["run_num"]), str(args["seed"]), args["task"], args["mode"],
        args["world_model_checkpoint"]["sha256"], args["initial_policy_checkpoint"]["sha256"]) != (
        run, seed, "go2_flat", "paper_warm", wm_sha, s2_sha):
    raise SystemExit("run provenance differs from the D3 configuration")
print(run_dir / f"policy_{end - 1}.pt")
CHECK_TRAINING
) || fail "could not verify completed D3 training"
[ -s "$P" ] || fail "missing final trained policy $P"
status "$MODE: verified $ITERS PPO iterations starting at $START_ITER; $P $(sha256sum "$P" | cut -c1-12)"
grep -E "Mean reward|Mean episode length" "$OUT/train.log" | tail -2 | tee -a "$OUT/queue.log"

# ---- 2. physics evaluation under the friction shift: D3 and unchanged s2
for ARM in "D3:$P" "S2:$CK"; do
  TAG=${ARM%%:*}; CKPT=${ARM#*:}
  for S in 42 43; do
    DIR="$OUT/eval/${TAG}_cmd10_s$S"; mkdir -p "$DIR"
    gpu_wait 8192 "eval $TAG s$S"
    status "eval $TAG seed $S (friction x$FRICTION)"
    CUDA_VISIBLE_DEVICES=0 $PY $EVAL --task $TASK --checkpoint "$CKPT" --num_envs 32 --num_steps 10000 --seed $S --std .5 \
      --cmd_resample_s 10 --friction-scale $FRICTION --headless --out_csv "$DIR/episodes.csv" > "$DIR/eval.log" 2>&1 \
      || fail "eval $TAG s$S ($DIR/eval.log)"
    grep -q "\[friction\] verified in the simulator" "$DIR/eval.log" || fail "no friction read-back in $DIR/eval.log"
    [ -s "$DIR/episodes_summary.json" ] || fail "no summary for $TAG s$S"
  done
done

# ---- 3. gate
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_d3_gate.py --eval "$OUT/eval" --friction $FRICTION --out "$OUT/d3_gate.json" \
  > "$OUT/gate.log" 2>&1 || { cat "$OUT/gate.log"; fail "gate scoring (gate.log)"; }
cat "$OUT/gate.log" | tee -a "$OUT/queue.log"
status "done: $(grep VERDICT "$OUT/gate.log"); $(sha256sum "$OUT/d3_gate.json")"
