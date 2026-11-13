#!/bin/bash
# E1 validation block, plan v5.2 §5 (ANYmal, ETH's released five-head model, training seed 1).
# Run from the robotic_world_model root inside the container:
#   setsid nohup scripts/reinforcement_learning/model_based/e1v_queue.sh <fresh_out_dir> > <out_dir>.nohup 2>&1 < /dev/null &
#
# Arms:  A frozen ETH WM, lambda 0  -> new run 2001       B frozen, lambda 1 -> saved 1701 (control)
#        C updated WM,   lambda 0  -> new run 2003       D updated, lambda 1 -> new run 2004
# Lane A (starts at once): train 2001 -> evaluate.
# Main lane: record 1701 + reference (pre-reset, 64 x 5000, seed 101, pushes off, commands 10 s)
#   -> prepare (envs 52-63 held out) -> refit with per-head memory reset (B3 wrapper, provenance checked)
#   -> model check, horizons 1/8/24 (queue stops on a failed bar) -> train 2003 and 2004 -> evaluate.
# Every new training is checked against 1701's run_args.json (e1v_match.py) as soon as it starts;
# a mismatch stops that training at once. Evaluations: seeds 42/43, 32 x 10000, std .5, pushes on,
# commands 30 s and 10 s (as 1901 and 1701). The feedback cycle is a separate step after this queue.
# One GPU process starts at a time (lock), each after >= 12 GiB are free. status_*.txt per lane.
set -uo pipefail
OUT=${1:?fresh output dir}
SEED=1; FT_ITERS=500; FT_LR=1e-4
MB=scripts/reinforcement_learning/model_based
EVAL=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
RECORDER=scripts/reinforcement_learning/rsl_rl/record_pre_reset_transitions.py
TASK=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
LOGS=logs/offline/anymal_d_flat
ETH=assets/models/pretrain_rnn_ens.pt
ETH_SHA=2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a
CTRL_DIR=$LOGS/2026-10-21_13-59-55_1701
CTRL=$CTRL_DIR/policy_2499.pt
PY=/isaac-sim/python.sh

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
LANE=main

status() { echo "$(date -u +%FT%TZ) [$LANE] $*" | tee "$OUT/status_$LANE.txt" >> "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
acquire_gpu() {   # one GPU start at a time; the lock is released 3 min after the start
  until mkdir "$OUT/gpu.lock" 2>/dev/null; do sleep 20; done
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12288 ]; do
    status "waiting for 12 GiB GPU before $1"; sleep 300; done; }
release_gpu_later() { ( sleep 180; rmdir "$OUT/gpu.lock" 2>/dev/null ) & }

# ---- 0. preflight (CPU)
status "preflight"
git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
git merge-base --is-ancestor d9d2a66 HEAD || fail "HEAD lacks the acceleration-reward fix d9d2a66"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; } > "$OUT/git.txt"
[ "$(sha256sum "$ETH" | cut -d' ' -f1)" = "$ETH_SHA" ] || fail "ETH checkpoint hash differs"
[ -s "$CTRL_DIR/run_args.json" ] || fail "no run_args.json for 1701 ($CTRL_DIR): matching check impossible"
cp "$CTRL_DIR/run_args.json" "$OUT/control_run_args.json"
sha256sum "$ETH" "$CTRL" $MB/e1_*.py $MB/e1v_*.py $MB/e1v_queue.sh $MB/refit/fit_reset_per_head.py \
  $MB/fit_world_model_go2_pilot_segments.py $MB/p15_train.py $MB/train.py $EVAL $RECORDER > "$OUT/inputs.sha256"
grep -q "policy hash MATCHES" logs/p2/e1A_1111_1650/eval_1701/hash_check.txt 2>/dev/null \
  || fail "1701 policy hash not confirmed (logs/p2/e1A_1111_1650/eval_1701/hash_check.txt)"

train_arm() {   # RUN WM PENALTY TAG
  local RUN=$1 WM=$2 PEN=$3 TAG=$4 D="" PID i
  ls -d $LOGS/*_$RUN >/dev/null 2>&1 && fail "$TAG: run $RUN already exists"
  acquire_gpu "train $TAG"; status "training arm $TAG: run $RUN, penalty weight $PEN, WM $WM"
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
       --expect-wm-sha "$(sha256sum "$WM" | cut -d' ' -f1)" --expect-penalty "$PEN" \
       --control-fix-note "966d5bb contains d9d2a66 (checked 13 Nov)" > "$OUT/match_$TAG.txt" 2>&1; then
    kill -TERM -- -$PID 2>/dev/null; wait $PID 2>/dev/null; fail "$TAG: matching check failed (match_$TAG.txt)"; fi
  status "arm $TAG: matching check passed; training continues ($D)"
  wait $PID || fail "$TAG: training exited non-zero (train_$TAG.log)"
  [ -s "$D/policy_2499.pt" ] || fail "$TAG: no policy_2499.pt in $D"
  echo "$D/policy_2499.pt" > "$OUT/policy_$TAG.txt"
}

evaluate() {    # TAG CHECKPOINT
  local TAG=$1 CK=$2 CMD S DIR
  for CMD in 30 10; do for S in 42 43; do
    DIR="$OUT/eval/${TAG}_cmd${CMD}_s${S}"; mkdir -p "$DIR"
    acquire_gpu "eval $TAG cmd$CMD s$S"; status "eval arm $TAG cmd$CMD seed $S"; release_gpu_later
    CUDA_VISIBLE_DEVICES=0 $PY $EVAL --task $TASK --checkpoint "$CK" --num_envs 32 --num_steps 10000 \
      --seed $S --std .5 --cmd_resample_s $CMD --headless --out_csv "$DIR/episodes.csv" > "$DIR/eval.log" 2>&1 \
      || fail "eval $TAG cmd$CMD s$S"
  done; done
}

# ---- lane A: frozen ETH model, lambda 0 (needs no new data)
( LANE=A; train_arm 2001 "$ETH" 0.0 A; evaluate A "$(cat "$OUT/policy_A.txt")"; status "done" ) &
LANE_A=$!

# ---- main lane: round-0 data from the deployed frozen-model policy (1701) and the reference
mkdir -p "$OUT/record"
for PAIR in baseline1701:$CTRL reference:$ETH; do
  NAME=${PAIR%%:*}; CK=${PAIR#*:}
  acquire_gpu "record $NAME"; status "recording $NAME (64 x 5000, seed 101, commands 10 s, pushes off)"; release_gpu_later
  CUDA_VISIBLE_DEVICES=0 $PY $RECORDER --task $TASK --checkpoint "$CK" --num_envs 64 --num_steps 5000 \
    --seed 101 --cmd_resample_s 10 --headless --out "$OUT/record/$NAME.npz" > "$OUT/record/$NAME.log" 2>&1 \
    || fail "recording $NAME"
done

status "preparing data"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1_prepare_data.py --baseline "$OUT/record/baseline1701.npz" \
  --reference "$OUT/record/reference.npz" --baseline-tag baseline1701 --out "$OUT/data" > "$OUT/prepare.log" 2>&1 \
  || fail "prepare data"

acquire_gpu "refit"; status "refit with per-head reset: ETH warm start, $FT_ITERS it, lr $FT_LR, seed $SEED"; release_gpu_later
CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config anymal_d_flat \
  --data "$OUT/data/segments" --output "$OUT/wm/e1v_wm.pt" --reference_wm "$ETH" --warm_start \
  --norm_stats "$OUT/data/eth_norm.npz" --seed $SEED --iterations $FT_ITERS --lr $FT_LR \
  --termination_pos_weight 1 --lazy_windows --save_interval 100 --device cuda:0 > "$OUT/refit.log" 2>&1 || fail "refit"
WM_NEW="$OUT/wm/e1v_wm.pt"
[ -s "$WM_NEW" ] || fail "no refit checkpoint"
CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
sys.exit(0 if ck.get('provenance', {}).get('b3_reset_per_head') is True else 1)" "$WM_NEW" \
  || fail "refit checkpoint lacks provenance b3_reset_per_head=True"

status "model check (horizons 1/8/24)"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1_model_check.py --eth "$ETH" --finetuned "$WM_NEW" \
  --baseline-test "$OUT/data/test_baseline1701.npz" --reference-test "$OUT/data/test_reference.npz" \
  --baseline-tag baseline1701 --horizons 1,8,24 --out "$OUT/model_check" > "$OUT/model_check.log" 2>&1 || fail "model check"
grep -q "MODEL CHECK PASS: True" "$OUT/model_check/report.txt" || fail "model check bars not met (model_check/report.txt)"

( LANE=C; train_arm 2003 "$WM_NEW" 0.0 C; status "trained" ) & PID_C=$!
( LANE=D; train_arm 2004 "$WM_NEW" -1.0 D; status "trained" ) & PID_D=$!
wait $PID_C || fail "arm C failed"
wait $PID_D || fail "arm D failed"
evaluate C "$(cat "$OUT/policy_C.txt")"
evaluate D "$(cat "$OUT/policy_D.txt")"
status "waiting for lane A"
wait $LANE_A || fail "lane A failed (status_A.txt)"

# ---- summary
{
  echo "== E1 validation block (plan v5.2), training seed $SEED"
  echo "== matching checks"; for T in A C D; do echo "-- $T: $(tail -1 "$OUT/match_$T.txt")"; done
  echo "== model check"; head -6 "$OUT/model_check/report.txt"
  echo "== arm B = 1701 (frozen, lambda 1)"; cat logs/p2/e1A_1111_1650/eval_1701/summary.txt 2>/dev/null || echo "missing"
  echo "== arms A, C, D (falls / episodes_completed / track_per_step / speed_ratio)"
  for DIR in "$OUT"/eval/*; do echo "-- $(basename "$DIR")"
    sed -n "/EPISODE STATS/,/^=====*$/p" "$DIR/eval.log" | grep -E "falls:|episodes_completed|track_per_step|speed_ratio"; done
} > "$OUT/summary.txt"
status "done"
