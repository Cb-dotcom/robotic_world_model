#!/bin/bash
# E1-A queue (plan v5.1). Run from the robotic_world_model root inside the container:
#   nohup scripts/reinforcement_learning/model_based/e1_queue.sh <recording_dir> <out_dir> > <out_dir>.nohup 2>&1 &
# Steps: wait for the recording -> prepare data (CPU) -> fine-tune ETH WM (GPU)
#        -> model check (CPU) -> train policy 1901, seed 1, paper recipe (GPU)
#        -> evaluate 1901 and control 1601: seeds 42/43, commands 30 s and 10 s (GPU) -> summary.
# Each GPU step waits until 12 GiB are free. Any failure stops the queue; status.txt says where.
set -uo pipefail
REC=${1:?recording dir, e.g. logs/p2/e1_record.67MmM0}
OUT=${2:?fresh output dir}
RUN=1901; SEED=1; FT_ITERS=500; FT_LR=1e-4
MB=scripts/reinforcement_learning/model_based
EVAL=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
TASK=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
ETH=assets/models/pretrain_rnn_ens.pt
BASE=logs/offline/anymal_d_flat/2026-10-16_14-50-46_1601/policy_2499.pt
PY=/isaac-sim/python.sh

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"

status() { echo "$(date -u +%FT%TZ) $*" > "$OUT/status.txt"; echo "$(date -u +%FT%TZ) $*" >> "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
wait_gpu() {
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12288 ]; do
    status "waiting for 12 GiB GPU before $1"; sleep 300; done; }

{ git rev-parse HEAD; git status --porcelain --untracked-files=no; } > "$OUT/git.txt"
sha256sum "$ETH" "$BASE" $MB/e1_*.py $MB/e1_queue.sh $MB/fit_world_model_go2_pilot_segments.py $MB/p15_train.py > "$OUT/inputs.sha256"

# 0. recording
until grep -q done "$REC/status.txt" 2>/dev/null; do status "waiting for recording $REC"; sleep 300; done
grep -q "baseline1601 rc=0" "$REC/rc.txt" && grep -q "reference rc=0" "$REC/rc.txt" || fail "recording rc not 0"

# 1. data (CPU)
status "preparing data"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1_prepare_data.py --baseline "$REC/baseline1601.npz" \
  --reference "$REC/reference.npz" --out "$OUT/data" > "$OUT/prepare.log" 2>&1 || fail "prepare data"

# 2. fine-tune the ETH model (GPU)
wait_gpu finetune; status "fine-tuning ETH WM ($FT_ITERS it, lr $FT_LR)"
CUDA_VISIBLE_DEVICES=0 $PY $MB/fit_world_model_go2_pilot_segments.py --config anymal_d_flat \
  --data "$OUT/data/segments" --output "$OUT/wm/e1_wm.pt" --reference_wm "$ETH" --warm_start \
  --norm_stats "$OUT/data/eth_norm.npz" --seed $SEED --iterations $FT_ITERS --lr $FT_LR \
  --termination_pos_weight 1 --lazy_windows --save_interval 100 --device cuda:0 > "$OUT/finetune.log" 2>&1 || fail "fine-tune"
[ -s "$OUT/wm/e1_wm.pt" ] || fail "no fine-tuned checkpoint"

# 3. model check (CPU)
status "model check"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1_model_check.py --eth "$ETH" --finetuned "$OUT/wm/e1_wm.pt" \
  --baseline-test "$OUT/data/test_baseline1601.npz" --reference-test "$OUT/data/test_reference.npz" \
  --out "$OUT/model_check" > "$OUT/model_check.log" 2>&1 || fail "model check"
# e1_model_check.py exits 0 even when a bar fails; the queue must read the verdict
grep -q "MODEL CHECK PASS: True" "$OUT/model_check/report.txt" || fail "model check bars not met (see model_check/report.txt)"

# 4. policy training in the fine-tuned model (GPU); 1601 is the frozen-model control
ls -d logs/offline/anymal_d_flat/*_$RUN >/dev/null 2>&1 && fail "run $RUN already exists"
wait_gpu train; status "training policy $RUN seed $SEED"
CUDA_VISIBLE_DEVICES=0 $PY $MB/e1_train.py --wm-checkpoint "$OUT/wm/e1_wm.pt" paper $RUN anymal_d_flat \
  --seed $SEED > "$OUT/train.log" 2>&1 || fail "training"
NEW=$(ls -d logs/offline/anymal_d_flat/*_$RUN | tail -1)/policy_2499.pt
[ -s "$NEW" ] || fail "no policy_2499.pt for run $RUN"
echo "$NEW" > "$OUT/policy_path.txt"

# 5. evaluations (GPU): same commit and protocol for both policies
for PAIR in e1_$RUN:$NEW control_1601:$BASE; do
  NAME=${PAIR%%:*}; CK=${PAIR#*:}
  for CMD in 30 10; do for S in 42 43; do
    D="$OUT/eval/${NAME}_cmd${CMD}_s${S}"; mkdir -p "$D"
    wait_gpu "eval $NAME cmd$CMD s$S"; status "eval $NAME cmd$CMD seed $S"
    CUDA_VISIBLE_DEVICES=0 $PY $EVAL --task $TASK --checkpoint "$CK" --num_envs 32 --num_steps 10000 \
      --seed $S --std .5 --cmd_resample_s $CMD --headless --out_csv "$D/episodes.csv" > "$D/eval.log" 2>&1 \
      || fail "eval $NAME cmd$CMD s$S"
  done; done
done

# 6. summary
{
  echo "== model check"; cat "$OUT/model_check/report.txt"
  echo "== evaluations (falls / episodes_completed / track_per_step / speed_ratio)"
  for D in "$OUT"/eval/*; do echo "-- $(basename "$D")"
    sed -n "/EPISODE STATS/,/^=====*$/p" "$D/eval.log" | grep -E "falls:|episodes_completed|track_per_step|speed_ratio"; done
} > "$OUT/summary.txt"
status "done"
