#!/bin/bash
# Go2 clean start policy (decision 14 Nov 2026, doubt E15). Run from the robotic_world_model root:
#   setsid nohup scripts/reinforcement_learning/model_based/start_policy_queue.sh <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
#
# Lanes (parallel, one GPU start at a time, each start after >= 12 GiB free):
#   vel   : train Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0       (policy sees base velocity), seed 1
#   novel : train Template-Isaac-Velocity-Flat-Unitree-Go2-Start-NoVel-v0 (no base velocity),         seed 1
#   june  : the June walker (model_2000.pt) in the Start task, for the G6 reference
# Both Start tasks = Pretrain env + plan v6 fall rule (base < 0.15 m or tilt > 60 deg) as termination
# and in system_termination. PPO: Baseline runner (paper settings, [128,128,128], 2000 it).
# After training: gait_metrics.py at 0.1-1.0 m/s (flat, no pushes), eval_episode_stats.py (32 x 10000,
# std .5, pushes on, seeds 42/43, commands 10 s and 30 s), then start_policy_gate.py per candidate.
set -uo pipefail
OUT=${1:?fresh output dir}
MB=scripts/reinforcement_learning/model_based
RL=scripts/reinforcement_learning/rsl_rl
JUNE=logs/rsl_rl/unitree_go2_flat/2026-06-12_09-07-38_pretrain/model_2000.pt
T_VEL=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0
T_NOVEL=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-NoVel-v0
SPEEDS=0.1,0.2,0.3,0.45,0.6,0.9,1.0
PY=/isaac-sim/python.sh

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
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
[ -s "$JUNE" ] || fail "missing June walker $JUNE"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $MB/analysis/tests/test_plan_fall.py $MB/analysis/tests/test_start_policy_gate.py \
  $MB/analysis/tests/test_gait_metrics_lib.py > "$OUT/tests.log" 2>&1 || fail "CPU tests (tests.log)"
sha256sum "$JUNE" $RL/gait_metrics.py $RL/gait_metrics_lib.py $RL/eval_episode_stats.py $RL/train.py \
  $MB/start_policy_gate.py $MB/start_policy_queue.sh > "$OUT/inputs.sha256"

gpu_job() {   # NAME LOG command...
  local NAME=$1 LOG=$2; shift 2
  acquire_gpu "$NAME"; status "$NAME"; release_gpu_later
  CUDA_VISIBLE_DEVICES=0 "$@" > "$LOG" 2>&1
}

measure() {   # TAG TASK CHECKPOINT
  local TAG=$1 TASK=$2 CK=$3 CMD S
  gpu_job "gait $TAG" "$OUT/gait_$TAG.log" $PY $RL/gait_metrics.py --pkg mbrl --task "$TASK" --checkpoint "$CK" \
    --label "$TAG" --speeds $SPEEDS --out "$OUT/gait_$TAG.json" --headless || return 1
  for CMD in 10 30; do for S in 42 43; do
    mkdir -p "$OUT/eval/${TAG}_cmd${CMD}_s${S}"
    gpu_job "eval $TAG cmd$CMD s$S" "$OUT/eval/${TAG}_cmd${CMD}_s${S}/eval.log" $PY $RL/eval_episode_stats.py \
      --task "$TASK" --checkpoint "$CK" --num_envs 32 --num_steps 10000 --seed $S --std .5 --cmd_resample_s $CMD \
      --headless --out_csv "$OUT/eval/${TAG}_cmd${CMD}_s${S}/episodes.csv" || return 1
  done; done
}

train() {   # TAG TASK RUNNAME
  local TAG=$1 TASK=$2 RUN=$3 D CK
  touch "$OUT/.t0_$TAG"
  gpu_job "train $TAG ($TASK, seed 1, 2000 it)" "$OUT/train_$TAG.log" $PY $RL/train.py --task "$TASK" --headless --seed 1 \
    || fail "$TAG: training (train_$TAG.log)"
  D=$(find logs/rsl_rl/unitree_go2_flat -maxdepth 1 -type d -name "*_$RUN" -newer "$OUT/.t0_$TAG" | sort | tail -1)
  CK=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1)
  [ -n "$D" ] && [ -s "$CK" ] || fail "$TAG: no checkpoint found for run *_$RUN"
  grep -q "model_2000.pt" <<< "$CK" || status "$TAG: last checkpoint is $(basename "$CK") (expected model_2000.pt)"
  echo "$CK" > "$OUT/policy_$TAG.txt"
  measure "$TAG" "$TASK" "$CK" || fail "$TAG: measurement"
  CUDA_VISIBLE_DEVICES="" $PY $MB/start_policy_gate.py --gait "$OUT/gait_$TAG.json" --eval-dir "$OUT/eval" \
    --tag "$TAG" --out "$OUT/gate_$TAG.txt" > /dev/null 2>&1 || fail "$TAG: gate script"
  status "done: $(tail -1 "$OUT/gate_$TAG.txt")"
}

( LANE=june; measure june "$T_VEL" "$JUNE" && status "done" || status "FAILED: June walker measurement (G6 will fail)" ) & PJ=$!
sleep 30
( LANE=vel; train vel "$T_VEL" start_vel ) & PV=$!
sleep 30
( LANE=novel; train novel "$T_NOVEL" start_novel ) & PN=$!
wait $PV; wait $PN; wait $PJ
# G6 needs the June evaluations; rescore now that every lane has finished
for TAG in vel novel; do
  [ -s "$OUT/gait_$TAG.json" ] && CUDA_VISIBLE_DEVICES="" $PY $MB/start_policy_gate.py --gait "$OUT/gait_$TAG.json" \
    --eval-dir "$OUT/eval" --tag "$TAG" --out "$OUT/gate_$TAG.txt" > /dev/null 2>&1
done
{ for TAG in vel novel; do cat "$OUT/gate_$TAG.txt" 2>/dev/null || echo "no gate for $TAG"; echo; done
  ls "$OUT"/gait_*.json >/dev/null 2>&1 && CUDA_VISIBLE_DEVICES="" $PY $RL/gait_metrics.py --compare "$OUT"/gait_*.json
} > "$OUT/summary.txt" 2>&1
status "done"
