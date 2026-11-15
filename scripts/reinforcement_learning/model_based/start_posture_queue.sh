#!/bin/bash
# Go2 start policy, iteration 3 = the last one (decision 15 Nov 2026): Start-Posture-v0 = Start-v0 + three
# terms of Unitree's Go2 task copied unchanged (posture penalty -0.7, joint limits -10, action rate -0.1),
# seeds 1 and 2. Gate of 15 Nov (start_policy_gate.py --reference-gait): the 14 Nov bars, G4 set from the
# reference, plus G7 base height and G8 bounce against the Unitree reference measured in the same gait test.
# Hard stop: if no seed passes, Start-v0 is the start policy. Run from the robotic_world_model root:
#   setsid nohup scripts/reinforcement_learning/model_based/start_posture_queue.sh <first_start_dir> <iter2_dir> <unitree_gait_json> <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
# Preflight (CPU) also rescores Start-v0 and Gait s1 with the 15 Nov gate, for information only.
set -uo pipefail
PREV=${1:?first start-policy folder (has eval/june_*, eval/vel_*, gait_vel.json)}
IT2=${2:?iteration-2 folder (has eval/gait_s1_*, gait_gait_s1.json)}
REFJ=${3:?Unitree reference gait JSON (gait_metrics.py)}
OUT=${4:?fresh output dir}
MB=scripts/reinforcement_learning/model_based
RL=scripts/reinforcement_learning/rsl_rl
TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-Posture-v0
SPEEDS=0.1,0.2,0.3,0.45,0.6,0.9,1.0
PY=/isaac-sim/python.sh

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
ls -d "$PREV"/eval/june_cmd10_s42 >/dev/null 2>&1 || { echo "no June evaluations in $PREV/eval"; exit 2; }
[ -s "$REFJ" ] || { echo "no reference gait JSON $REFJ"; exit 2; }
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
  $MB/analysis/tests/test_gait_metrics_lib.py $MB/analysis/tests/test_posture_reward.py > "$OUT/tests.log" 2>&1 \
  || fail "CPU tests (tests.log)"
cp -r "$PREV"/eval/june_* "$OUT/eval/"
cp "$PREV/gait_june.json" "$OUT/" 2>/dev/null
cp "$REFJ" "$OUT/reference_gait.json"
{ echo "June reference copied from $PREV"; echo "reference gait: $REFJ"; sha256sum "$REFJ"; } > "$OUT/sources.txt"
GATE="$PY $MB/start_policy_gate.py --reference-gait $OUT/reference_gait.json"
CUDA_VISIBLE_DEVICES="" $GATE --gait "$PREV/gait_vel.json" --eval-dir "$PREV/eval" --tag vel \
  --out "$OUT/info_gate15_startv0.txt" > /dev/null 2>&1 || fail "info gate Start-v0"
CUDA_VISIBLE_DEVICES="" $GATE --gait "$IT2/gait_gait_s1.json" --eval-dir "$IT2/eval" --tag gait_s1 \
  --out "$OUT/info_gate15_gait_s1.txt" > /dev/null 2>&1 || fail "info gate Gait s1"
status "preflight done; info gates written (Start-v0, Gait s1)"

gpu_job() {   # NAME LOG command...
  local NAME=$1 LOG=$2; shift 2
  acquire_gpu "$NAME"; status "$NAME"; release_gpu_later
  CUDA_VISIBLE_DEVICES=0 "$@" > "$LOG" 2>&1
}

lane() {   # SEED
  local S=$1 TAG="posture_s$1" RUN="start_posture_s$1" D CK CMD E
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
  CUDA_VISIBLE_DEVICES="" $GATE --gait "$OUT/gait_$TAG.json" --eval-dir "$OUT/eval" --tag "$TAG" \
    --out "$OUT/gate_$TAG.txt" > /dev/null 2>&1 || fail "$TAG: gate script"
  status "done: $(tail -1 "$OUT/gate_$TAG.txt")"
}

( LANE=s1; lane 1 ) & P1=$!
sleep 30
( LANE=s2; lane 2 ) & P2=$!
wait $P1; wait $P2
{ for T in posture_s1 posture_s2; do cat "$OUT/gate_$T.txt" 2>/dev/null || echo "no gate for $T"; echo; done
  echo "== for information (15 Nov gate on earlier candidates)"; cat "$OUT"/info_gate15_*.txt
  CUDA_VISIBLE_DEVICES="" $PY $RL/gait_metrics.py --compare "$OUT"/gait_posture_*.json "$PREV/gait_vel.json" "$OUT/reference_gait.json"
} > "$OUT/summary.txt" 2>&1
status "done"
