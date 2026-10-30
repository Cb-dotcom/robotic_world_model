#!/bin/bash
# Clean-repro test (phase 1, decides) + lambda sweep (phase 2, exploratory). One GPU job at a time.
# Start (inside the container):
#   nohup bash lab/clean_repro/run_all.sh > /workspace/clean_repro_run.log 2>&1 &
# Resumable: run it again with the same OUT=... and finished steps are skipped.
set -uo pipefail
LAB=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/isaac-sim/python.sh}
FORK=${FORK:-/workspace/Uncertainty-Aware-Robotic-World-Model-Go2/upstream/robotic_world_model}
CLEAN=${CLEAN:-/workspace/rwm_clean}
OUT=${OUT:-/workspace/clean_repro/$(date +%m%d_%H%M)}
LAMBDAS=${LAMBDAS:-"0 5 20"}          # lambda = 1 is the repro run R1 itself
SKIP_LAMBDA=${SKIP_LAMBDA:-0}
TRAIN_GIB=${TRAIN_GIB:-12}; EVAL_GIB=${EVAL_GIB:-6}
STD_TASK=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
PLAY_TASK=Isaac-Velocity-Flat-Anymal-D-Play-v0
EVALUATOR=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
CEILING=$FORK/assets/models/pretrain_rnn_ens.pt
mkdir -p "$OUT"; echo "$OUT" > /workspace/clean_repro_latest.txt

status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/status.txt"; bash "$LAB/push_results.sh" "$OUT" clean-repro-live "$FORK"; }
fail()   { status "FAILED: $*"; exit 1; }

gpu_wait() {  # wait until <need> GiB are free; never stop other processes
  local need=$(( $1 * 1024 )) t=0
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $1-$2}')
    [ "${free:-0}" -ge "$need" ] && return 0
    [ $t -eq 0 ] && status "waiting for GPU: ${free} MiB free, need ${need}"
    t=$((t + 1)); [ $t -gt 96 ] && fail "GPU busy for 8 h"
    sleep 300
  done
}

train() {  # train <run> <run_num> [lambda]
  local run=$1 num=$2 lam=${3:-} d="$OUT/$1/train"
  mkdir -p "$d"
  if [ -f "$d/checkpoint.txt" ]; then return 0; fi
  gpu_wait "$TRAIN_GIB"; status "training $run (run_num $num, lambda ${lam:-repo default})"
  local extra=(); [ -n "$lam" ] && extra=(--lambda_ "$lam")
  ( cd "$CLEAN" && WANDB_MODE=offline PYTHONUNBUFFERED=1 "$PY" "$LAB/train_clean.py" --clone "$CLEAN" --run_num "$num" \
      "${extra[@]}" --metrics_jsonl "$d/metrics.jsonl" --manifest "$d/manifest.json" ) > "$d/train.log" 2>&1 \
      || fail "training $run exited non-zero (see $d/train.log)"
  grep -q "Training completed. Policy saved to" "$d/train.log" || fail "no completion marker for $run"
  local ck; ck=$(ls "$CLEAN"/logs/offline/anymal_d_flat/*_"$num"/policy_*.pt 2>/dev/null | sort -V | tail -1)
  [ -n "$ck" ] || fail "no checkpoint for $run"
  echo "$ck" > "$d/checkpoint.txt"; sha256sum "$ck" >> "$d/checkpoint.txt"
  status "trained $run -> $(basename "$(dirname "$ck")")/$(basename "$ck")"
}

evaluate() {  # evaluate <run> <proto std|cmd30|play> <seed> <checkpoint> [optional]
  local run=$1 proto=$2 seed=$3 ck=$4 opt=${5:-} d="$OUT/$1/eval_$2_$3"
  [ -f "$d/episodes_summary.json" ] && return 0
  mkdir -p "$d"; gpu_wait "$EVAL_GIB"; status "evaluating $run $proto seed $seed"
  local args=(--checkpoint "$ck" --num_envs 32 --num_steps 10000 --seed "$seed" --std 0.5 --headless --out_csv "$d/episodes.csv")
  case $proto in
    std)   args+=(--task "$STD_TASK" --cmd_resample_s 10) ;;
    cmd30) args+=(--task "$STD_TASK" --cmd_resample_s 30) ;;
    play)  args+=(--task "$PLAY_TASK") ;;
  esac
  if ! ( cd "$FORK" && WANDB_MODE=offline PYTHONUNBUFFERED=1 "$PY" "$EVALUATOR" "${args[@]}" ) > "$d/eval.log" 2>&1 \
       || [ ! -f "$d/episodes_summary.json" ]; then
    [ -n "$opt" ] && { status "optional eval $run $proto failed (see eval.log); continuing"; return 0; }
    fail "evaluation $run $proto $seed"
  fi
}

report() { "$PY" "$LAB/verdict.py" "$OUT" > "$OUT/verdict.txt" 2>&1; status "verdict updated: $(grep -E 'REPRO VERDICT' "$OUT/verdict.txt")"; }
ck_of() { head -1 "$OUT/$1/train/checkpoint.txt"; }

# ---------- phase 0: provenance ----------
status "start OUT=$OUT"
[ -f "$OUT/versions.txt" ] || bash "$LAB/versions.sh" > "$OUT/versions.txt" 2>&1
bash "$LAB/setup_clean.sh" "$CLEAN" "$FORK" > "$OUT/setup.txt" 2>&1 || fail "setup (see setup.txt)"
"$PY" "$LAB/train_clean.py" --clone "$CLEAN" --run_num 0 --metrics_jsonl /dev/null --manifest /dev/null --dry_run > "$OUT/config_dryrun.txt" 2>&1
[ -f "$FORK/$EVALUATOR" ] || fail "evaluator missing in fork checkout"
status "phase 0 done: $(grep -m1 isaac_sim "$OUT/versions.txt")"

# ---------- phase 1: clean repro (decides, locked rule in verdict.py) ----------
train R1 2001;  evaluate R1 std 42 "$(ck_of R1)"; evaluate R1 std 43 "$(ck_of R1)"; report
train R2 2002;  evaluate R2 std 42 "$(ck_of R2)"; evaluate R2 std 43 "$(ck_of R2)"; report
evaluate R1 play 42 "$(ck_of R1)" optional
evaluate R2 play 42 "$(ck_of R2)" optional
mkdir -p "$OUT/ceiling"; evaluate ceiling play 42 "$CEILING" optional
report
if grep -q "MIXED" "$OUT/verdict.txt"; then
  train R3 2003; evaluate R3 std 42 "$(ck_of R3)"; evaluate R3 std 43 "$(ck_of R3)"; report
fi
status "PHASE 1 DONE"

# ---------- phase 2: lambda sweep (exploratory, 1 training per lambda) ----------
[ "$SKIP_LAMBDA" = 1 ] && { status "lambda sweep skipped"; exit 0; }
evaluate R1 cmd30 42 "$(ck_of R1)"; evaluate R1 cmd30 43 "$(ck_of R1)"; report
for lam in $LAMBDAS; do
  run="L${lam//./p}"; num=$((2100 + ${lam%.*}))
  train "$run" "$num" "$lam"
  for proto in std cmd30; do for s in 42 43; do evaluate "$run" "$proto" "$s" "$(ck_of "$run")"; done; done
  report
done
status "ALL DONE"
