#!/bin/bash
# Go2 start policy, iteration 2 (14 Nov 2026 evening): Start-Gait-v0 (= Start-v0 + Spot foot clearance,
# diagonal-gait reward, air-time variance penalty), seeds 1 and 2 in parallel. Same gate (decision 14 Nov),
# June walker reference reused from the first start-policy folder. Run from the robotic_world_model root:
#   setsid nohup scripts/reinforcement_learning/model_based/start_gait_queue.sh <first_start_dir> <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
set -uo pipefail
PREV=${1:?first start-policy folder (has eval/june_*)}
OUT=${2:?fresh output dir}
MB=scripts/reinforcement_learning/model_based
RL=scripts/reinforcement_learning/rsl_rl
TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-Gait-v0
SPEEDS=0.1,0.2,0.3,0.45,0.6,0.9,1.0
PY=/isaac-sim/python.sh

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
ls -d "$PREV"/eval/june_cmd10_s42 >/dev/null 2>&1 || { echo "no June evaluations in $PREV/eval"; exit 2; }
mkdir -p "$OUT/eval"
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
LANE=main
status() { echo "$(date -u +%FT%TZ) [$LANE] $*" | tee "$OUT/status_$LANE.txt" >> "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
acquire_gpu() {
  until mkdir "$OUT/gpu.lock" 2>/dev/null; do sleep 20; done
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12288 ]; do
    status "waiting for 12 GiB GPU before $1"; sleep 300; done; }
release_gpu_later() { ( sleep 180; rmdir "$OUT/gpu.lock" 2>/dev/null ) & }

status "preflight"
git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; git -C ../rsl_rl_rwm rev-parse HEAD; } > "$OUT/git.txt"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $MB/analysis/tests/test_plan_fall.py $MB/analysis/tests/test_start_policy_gate.py \
  $MB/analysis/tests/test_gait_metrics_lib.py > "$OUT/tests.log" 2>&1 || fail "CPU tests (tests.log)"
cp -r "$PREV"/eval/june_* "$OUT/eval/"
cp "$PREV/gait_june.json" "$OUT/" 2>/dev/null
echo "June reference copied from $PREV" > "$OUT/june_source.txt"

gpu_job() {   # NAME LOG command...
  local NAME=$1 LOG=$2; shift 2
  acquire_gpu "$NAME"; status "$NAME"; release_gpu_later
  CUDA_VISIBLE_DEVICES=0 "$@" > "$LOG" 2>&1
}

lane() {   # SEED
  local S=$1 TAG="gait_s$1" RUN="start_gait_s$1" D CK CMD E
  touch "$OUT/.t0_$TAG"
  gpu_job "train $TAG ($TASK, seed $S, 2000 it)" "$OUT/train_$TAG.log" $PY $RL/train.py --task "$TASK" --headless \
    --seed "$S" agent.run_name="$RUN" || fail "$TAG: training (train_$TAG.log)"
  D=$(find logs/rsl_rl/unitree_go2_flat -maxdepth 1 -type d -name "*_$RUN" -newer "$OUT/.t0_$TAG" | sort | tail -1)
  CK=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1)
  [ -n "$D" ] && [ -s "$CK" ] || fail "$TAG: no checkpoint for run *_$RUN"
  echo "$CK" > "$OUT/policy_$TAG.txt"
  gpu_job "gait $TAG" "$OUT/gait_$TAG.log" $PY $RL/gait_metrics.py --pkg mbrl --task "$TASK" --checkpoint "$CK" \
    --label "$TAG" --speeds $SPEEDS --out "$OUT/gait_$TAG.json" --headless || fail "$TAG: gait"
  for CMD in 10 30; do for E in 42 43; do
    mkdir -p "$OUT/eval/${TAG}_cmd${CMD}_s${E}"
    gpu_job "eval $TAG cmd$CMD s$E" "$OUT/eval/${TAG}_cmd${CMD}_s${E}/eval.log" $PY $RL/eval_episode_stats.py \
      --task "$TASK" --checkpoint "$CK" --num_envs 32 --num_steps 10000 --seed $E --std .5 --cmd_resample_s $CMD \
      --headless --out_csv "$OUT/eval/${TAG}_cmd${CMD}_s${E}/episodes.csv" || fail "$TAG: eval cmd$CMD s$E"
  done; done
  CUDA_VISIBLE_DEVICES="" $PY $MB/start_policy_gate.py --gait "$OUT/gait_$TAG.json" --eval-dir "$OUT/eval" \
    --tag "$TAG" --out "$OUT/gate_$TAG.txt" > /dev/null 2>&1 || fail "$TAG: gate script"
  status "done: $(tail -1 "$OUT/gate_$TAG.txt")"
}

( LANE=s1; lane 1 ) & P1=$!
sleep 30
( LANE=s2; lane 2 ) & P2=$!
wait $P1; wait $P2
{ for T in gait_s1 gait_s2; do cat "$OUT/gate_$T.txt" 2>/dev/null || echo "no gate for $T"; echo; done
  CUDA_VISIBLE_DEVICES="" $PY $RL/gait_metrics.py --compare "$OUT"/gait_*.json
} > "$OUT/summary.txt" 2>&1
status "done"
