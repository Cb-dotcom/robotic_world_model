#!/bin/bash
# Plan v6 W1 noise floor (decision 15 Nov 2026): two refits of the UNCHANGED control recipe (the E1-V refit:
# ETH full warm start, E1-V round-0 segments, ETH normalizer, per-head reset, 500 it, lr 1e-4, pos-weight 1)
# with seeds 2 and 3. Together with the control (seed 1) and the W1 100 % dose fit (same recipe, seed 1:
# a repeat of the control) they show how much two fits of one recipe differ. Every one-change refit
# (C4, C8, C5, C6, C3) is read against this spread.
# Runs inside the running v6_w1_refits.sh folder and shares its GPU lock; that queue is not changed.
#   setsid nohup scripts/reinforcement_learning/model_based/v6_w1_noise.sh <e1v_out_dir> <v6w1_out_dir> <e1v_feedback_dir> > <v6w1_out_dir>/noise.nohup 2>&1 < /dev/null &
set -uo pipefail
E1V=${1:?e1v_queue output dir}
OUT=${2:?existing v6_w1_refits.sh output dir}
FB=${3:?e1v_feedback output dir}
MB=scripts/reinforcement_learning/model_based
ETH=assets/models/pretrain_rnn_ens.pt
DATA="$E1V/data/segments"; NORM="$E1V/data/eth_norm.npz"; CONTROL="$E1V/wm/e1v_wm.pt"
PY=/isaac-sim/python.sh
MIB=13312; ITERS=500
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
LANE=noise
[ -d "$OUT" ] && [ -s "$OUT/c3_data/dose100/cards.json" ] || { echo "not a started v6_w1_refits.sh folder: $OUT"; exit 2; }
status() { echo "$(date -u +%FT%TZ) [$LANE] $*" | tee "$OUT/status_$LANE.txt" >> "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
acquire_gpu() {
  until mkdir "$OUT/gpu.lock" 2>/dev/null; do sleep 20; done
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$MIB" ]; do
    status "waiting for $MIB MiB GPU before $1"; sleep 300; done; }
release_gpu_later() { ( sleep 240; rmdir "$OUT/gpu.lock" 2>/dev/null ) & }

git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
git rev-parse HEAD > "$OUT/git_noise.txt"
NSHA=$(sha256sum "$NORM" | cut -d' ' -f1)
TERMS=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['real_terms_expected'])" "$OUT/c3_data/dose100/cards.json")

for S in 2 3; do
  TAG=c0_seed$S
  [ -e "$OUT/$TAG/wm.pt" ] && fail "$TAG/wm.pt already exists"
  for CH in 1 2 4; do
    acquire_gpu "$TAG"; status "fit $TAG (control recipe, seed $S, grad_accum_chunks $CH)"; release_gpu_later
    mkdir -p "$OUT/$TAG"
    CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config anymal_d_flat --data "$DATA" \
      --output "$OUT/$TAG/wm.pt" --reference_wm "$ETH" --warm_start --norm_stats "$NORM" --iterations $ITERS \
      --lr 1e-4 --termination_pos_weight 1 --lazy_windows --save_interval 100 --device cuda:0 --seed $S \
      --grad_accum_chunks $CH > "$OUT/$TAG/fit_ch$CH.log" 2>&1
    RC=$?
    [ $RC -eq 0 ] && break
    grep -q "CUDA out of memory\|OutOfMemoryError" "$OUT/$TAG/fit_ch$CH.log" || fail "$TAG: fit rc=$RC (fit_ch$CH.log)"
    status "$TAG: out of memory with $CH chunk(s); retrying"; rm -f "$OUT/$TAG"/*.pt
  done
  [ $RC -eq 0 ] || fail "$TAG: out of memory even with 4 chunks"
  CUDA_VISIBLE_DEVICES="" $PY $MB/refit/check_fit.py --ckpt "$OUT/$TAG/wm.pt" --iters $ITERS --heads 5 --state_loss mse \
    --norm_sha256 "$NSHA" --data_dir "$DATA" --real_terms "$TERMS" > "$OUT/$TAG/check.txt" 2>&1 || fail "$TAG: fit check"
  status "$TAG: done"
done

status "waiting for the W1 100 % dose fit"
until grep -q "c3_dose100: done" "$OUT/status_C3b.txt" 2>/dev/null; do
  grep -q "FAILED" "$OUT/status_C3b.txt" 2>/dev/null && fail "W1 lane C3b failed; no dose100 fit"
  sleep 300; done
status "CPU score: control seed 1, dose100 (same recipe, seed 1), seeds 2 and 3"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1v_score.py --model "c0_seed1=$CONTROL" --model "c0_seed1_repeat=$OUT/c3_dose100/wm.pt" \
  --model "c0_seed2=$OUT/c0_seed2/wm.pt" --model "c0_seed3=$OUT/c0_seed3/wm.pt" \
  --trace "heldout1701=$E1V/data/test_baseline1701.npz" --trace "heldout_reference=$E1V/data/test_reference.npz" \
  --trace "batchA=$FB/batch_A.npz" --trace "batchC=$FB/batch_C.npz" --horizons 1,8,24 --out "$OUT/score_noise" \
  > "$OUT/score_noise.log" 2>&1 || fail "CPU score (score_noise.log)"
status "done"
