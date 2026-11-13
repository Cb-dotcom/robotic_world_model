#!/bin/bash
# E1 replication (plan v5.2 §5; Charbel 13 Nov): arms B and D with training seeds 2 and 3.
# Run from the robotic_world_model root inside the container:
#   setsid nohup scripts/reinforcement_learning/model_based/e1r_queue.sh <e1v_out_dir> <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
#
#   B (frozen ETH WM, lambda 1):          seeds 2, 3 -> runs 2102, 2103   (seed 1 = 1701)
#   D (refit WM of the e1v block, lambda 1): seeds 2, 3 -> runs 2402, 2403 (seed 1 = 2004)
# Same recipe and evaluation as e1v_queue.sh. Each training is checked against 1701's run_args.json
# with e1v_match.py --expect-seed as soon as it starts; a mismatch stops it at once.
# Four lanes run in parallel; one GPU start at a time, each after >= 12 GiB are free.
set -uo pipefail
E1V=${1:?e1v_queue output dir}
OUT=${2:?fresh output dir}
MB=scripts/reinforcement_learning/model_based
EVAL=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
TASK=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
LOGS=logs/offline/anymal_d_flat
ETH=assets/models/pretrain_rnn_ens.pt
ETH_SHA=2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a
WM1="$E1V/wm/e1v_wm.pt"
CTRL_DIR=$LOGS/2026-10-21_13-59-55_1701
PY=/isaac-sim/python.sh

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
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
git merge-base --is-ancestor d9d2a66 HEAD || fail "HEAD lacks the acceleration-reward fix d9d2a66"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; } > "$OUT/git.txt"
[ "$(sha256sum "$ETH" | cut -d' ' -f1)" = "$ETH_SHA" ] || fail "ETH checkpoint hash differs"
grep -q "\] done" "$E1V/status_main.txt" || fail "e1v block not done"
[ -s "$WM1" ] || fail "no refit model $WM1"
CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
sys.exit(0 if ck.get('provenance', {}).get('b3_reset_per_head') is True else 1)" "$WM1" || fail "refit model lacks b3 provenance"
[ -s "$CTRL_DIR/run_args.json" ] || fail "no run_args.json for 1701"
cp "$CTRL_DIR/run_args.json" "$OUT/control_run_args.json"
sha256sum "$ETH" "$WM1" $MB/e1_train.py $MB/e1v_match.py $MB/e1r_queue.sh $MB/p15_train.py $MB/train.py $EVAL > "$OUT/inputs.sha256"

train_arm() {   # RUN WM PENALTY SEED TAG
  local RUN=$1 WM=$2 PEN=$3 SEED=$4 TAG=$5 D="" PID i
  ls -d $LOGS/*_$RUN >/dev/null 2>&1 && fail "$TAG: run $RUN already exists"
  acquire_gpu "train $TAG"; status "training $TAG: run $RUN, seed $SEED, penalty weight $PEN, WM $WM"
  CUDA_VISIBLE_DEVICES=0 setsid $PY $MB/e1_train.py --wm-checkpoint "$WM" --penalty-weight "$PEN" \
    paper $RUN anymal_d_flat --seed $SEED > "$OUT/train_$TAG.log" 2>&1 &
  PID=$!; release_gpu_later
  for i in $(seq 1 240); do
    D=$(ls -d $LOGS/*_$RUN 2>/dev/null | tail -1)
    [ -n "$D" ] && [ -s "$D/run_args.json" ] && break
    kill -0 $PID 2>/dev/null || break; sleep 15; done
  if [ -z "$D" ] || [ ! -s "$D/run_args.json" ]; then
    kill -TERM -- -$PID 2>/dev/null; fail "$TAG: training wrote no run_args.json (train_$TAG.log)"; fi
  if ! $PY $MB/e1v_match.py --control "$OUT/control_run_args.json" --candidate "$D/run_args.json" \
       --expect-wm-sha "$(sha256sum "$WM" | cut -d' ' -f1)" --expect-penalty "$PEN" --expect-seed "$SEED" \
       --control-fix-note "966d5bb contains d9d2a66 (checked 13 Nov)" > "$OUT/match_$TAG.txt" 2>&1; then
    kill -TERM -- -$PID 2>/dev/null; wait $PID 2>/dev/null; fail "$TAG: matching check failed (match_$TAG.txt)"; fi
  status "$TAG: matching check passed; training continues ($D)"
  wait $PID || fail "$TAG: training exited non-zero (train_$TAG.log)"
  [ -s "$D/policy_2499.pt" ] || fail "$TAG: no policy_2499.pt in $D"
  echo "$D/policy_2499.pt" > "$OUT/policy_$TAG.txt"
}

evaluate() {    # TAG CHECKPOINT
  local TAG=$1 CK=$2 CMD S DIR
  for CMD in 30 10; do for S in 42 43; do
    DIR="$OUT/eval/${TAG}_cmd${CMD}_s${S}"; mkdir -p "$DIR"
    acquire_gpu "eval $TAG cmd$CMD s$S"; status "eval $TAG cmd$CMD seed $S"; release_gpu_later
    CUDA_VISIBLE_DEVICES=0 $PY $EVAL --task $TASK --checkpoint "$CK" --num_envs 32 --num_steps 10000 \
      --seed $S --std .5 --cmd_resample_s $CMD --headless --out_csv "$DIR/episodes.csv" > "$DIR/eval.log" 2>&1 \
      || fail "eval $TAG cmd$CMD s$S"
  done; done
}

PIDS=()
for SPEC in B2:2102:$ETH:2 B3:2103:$ETH:3 D2:2402:$WM1:2 D3:2403:$WM1:3; do
  IFS=: read -r TAG RUN WM SEED <<< "$SPEC"
  ( LANE=$TAG; train_arm "$RUN" "$WM" -1.0 "$SEED" "$TAG"; evaluate "$TAG" "$(cat "$OUT/policy_$TAG.txt")"; status "done" ) &
  PIDS+=($!)
  sleep 30
done
RC=0; for P in "${PIDS[@]}"; do wait "$P" || RC=1; done
[ $RC -eq 0 ] || fail "a lane failed (status_*.txt)"

{
  echo "== E1 replication: arms B and D, training seeds 1-3 (lambda 1)"
  echo "== matching checks"; for T in B2 B3 D2 D3; do echo "-- $T: $(tail -1 "$OUT/match_$T.txt")"; done
  echo "== seed 1: B = 1701"; cat logs/p2/e1A_1111_1650/eval_1701/summary.txt 2>/dev/null
  echo "== seed 1: D = 2004"
  for DIR in "$E1V"/eval/D_*; do echo "-- $(basename "$DIR")"
    sed -n "/EPISODE STATS/,/^=====*$/p" "$DIR/eval.log" | grep -E "falls:|episodes_completed|track_per_step|speed_ratio"; done
  echo "== seeds 2-3"
  for DIR in "$OUT"/eval/*; do echo "-- $(basename "$DIR")"
    sed -n "/EPISODE STATS/,/^=====*$/p" "$DIR/eval.log" | grep -E "falls:|episodes_completed|track_per_step|speed_ratio"; done
} > "$OUT/summary.txt"
status "done"
