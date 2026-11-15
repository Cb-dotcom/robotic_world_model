#!/bin/bash
# Plan v6 W1 (16-22 Nov): the GPU refits of the "why" battery, on the E1-V round-0 data.
# Run from the robotic_world_model root inside the container:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_w1_refits.sh <e1v_out_dir> <fresh_out_dir> <e1v_feedback_dir> > <out>.nohup 2>&1 < /dev/null &
#
# Common recipe = the E1-V refit (control, logs/p2/e1v_1113_0939/wm/e1v_wm.pt): ETH full warm start, data
# $E1V/data/segments (1701 envs 0-51 + reference, envs 52-63 held out), ETH normalizer, per-head memory reset
# (B3 wrapper), 500 it, lr 1e-4, termination pos-weight 1, lazy windows, seed 1. Each fit changes one thing:
#   C5   nll          state loss = ETH's own Gaussian-NLL branch (control: sampled MSE)       -> nll/wm.pt
#   C6   c6_widegrav  gravity std widened so that tilt 60 deg -> |z| = 3 (0.289/0.289/0.167)  -> c6_widegrav/wm.pt
#   C3   c3_dose{000,025,050,100}  data keep 0/25/50/100 % of the 1701 fall episodes (nested, seed 0);
#        non-fall 1701 episodes and the reference stay                                       -> c3_dose*/wm.pt
# On CUDA out-of-memory a fit is retried with --grad_accum_chunks 2, then 4 (same samples per step).
# Every checkpoint is checked (refit/check_fit.py: 500 it, b3 reset, the one intended change).
# Last step (CPU), provisional read only: e1v_score on the held-out 1701/reference envs and the E1-V
# feedback batches A and C (falls no refit saw). C6 is scored alone (other normalizer: its disagreement is
# not comparable; velocity RMSE is). The plan rules (HI(25), coverage) are read by the W1 CPU battery.
# No policy training. Three lanes; one GPU start at a time; each start waits for 13 GiB free.
set -uo pipefail
E1V=${1:?e1v_queue output dir}
OUT=${2:?fresh output dir}
FB=${3:?e1v_feedback output dir (batch_A.npz, batch_C.npz)}
MB=scripts/reinforcement_learning/model_based
ETH=assets/models/pretrain_rnn_ens.pt
ETH_SHA=2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a
DATA="$E1V/data/segments"; NORM="$E1V/data/eth_norm.npz"; CONTROL="$E1V/wm/e1v_wm.pt"
PY=/isaac-sim/python.sh
MIB=13312; ITERS=500

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
LANE=main
status() { echo "$(date -u +%FT%TZ) [$LANE] $*" | tee "$OUT/status_$LANE.txt" >> "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
acquire_gpu() {   # NAME
  until mkdir "$OUT/gpu.lock" 2>/dev/null; do sleep 20; done
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$MIB" ]; do
    status "waiting for $MIB MiB GPU before $1"; sleep 300; done; }
release_gpu_later() { ( sleep 240; rmdir "$OUT/gpu.lock" 2>/dev/null ) & }

# ---- 0. preflight (CPU)
status "preflight"
git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; git -C ../rsl_rl_rwm rev-parse HEAD;
  cat /isaac-sim/VERSION 2>/dev/null || echo "no /isaac-sim/VERSION";
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"; } > "$OUT/git.txt" 2>&1
[ "$(sha256sum "$ETH" | cut -d' ' -f1)" = "$ETH_SHA" ] || fail "ETH checkpoint hash differs"
[ -s "$DATA/manifest.csv" ] && [ -s "$NORM" ] && [ -s "$CONTROL" ] || fail "E1-V data, normalizer or control model missing"
[ -s "$E1V/data/test_baseline1701.npz" ] && [ -s "$E1V/data/test_reference.npz" ] || fail "E1-V held-out test traces missing"
[ -s "$FB/batch_A.npz" ] && [ -s "$FB/batch_C.npz" ] || fail "feedback batches A/C missing in $FB"
sha256sum "$ETH" "$NORM" "$CONTROL" "$DATA/manifest.csv" "$FB/batch_A.npz" "$FB/batch_C.npz" \
  $MB/fit_world_model_go2_pilot_segments.py $MB/refit/fit_reset_per_head.py $MB/refit/make_fall_dose.py \
  $MB/refit/make_widened_norm.py $MB/refit/check_fit.py $MB/v6_w1_refits.sh $MB/e1v_score.py > "$OUT/inputs.sha256"
status "CPU tests"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $MB/analysis/tests/test_partial_warm_start.py \
  $MB/analysis/tests/test_w1_refit_inputs.py $MB/analysis/tests/test_run_provenance.py > "$OUT/tests.log" 2>&1 || fail "CPU tests failed (tests.log)"

# ---- 1. inputs (CPU)
status "C6 normalizer and C3 dose folders"
CUDA_VISIBLE_DEVICES="" $PY $MB/refit/make_widened_norm.py --src "$NORM" --out "$OUT/c6_norm/widegrav_norm.npz" \
  > "$OUT/c6_norm.log" 2>&1 || fail "widened normalizer (c6_norm.log)"
WNORM="$OUT/c6_norm/widegrav_norm.npz"; WSHA=$(sha256sum "$WNORM" | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES="" $PY $MB/refit/make_fall_dose.py --src "$DATA" --out "$OUT/c3_data" --doses 0,25,50,100 \
  --seed 0 --fall_tag baseline1701 > "$OUT/c3_data.log" 2>&1 || fail "dose folders (c3_data.log)"
cat "$OUT/c3_data.log" >> "$OUT/queue.log"
terms() { $PY -c "import json, sys; print(json.load(open(sys.argv[1]))['real_terms_expected'])" "$OUT/c3_data/dose$1/cards.json"; }

fit() {   # TAG DATA NORM CHECK_ARGS -- extra fitter args...
  local TAG=$1 FDATA=$2 FNORM=$3 CHECK=$4
  [ "${5:-}" = "--" ] || fail "fit $TAG: fifth argument must be --"; shift 5
  local CH RC
  for CH in 1 2 4; do
    acquire_gpu "$TAG"; status "fit $TAG (grad_accum_chunks $CH)"; release_gpu_later
    mkdir -p "$OUT/$TAG"
    CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config anymal_d_flat --data "$FDATA" \
      --output "$OUT/$TAG/wm.pt" --reference_wm "$ETH" --warm_start --norm_stats "$FNORM" --iterations $ITERS \
      --lr 1e-4 --termination_pos_weight 1 --lazy_windows --save_interval 100 --device cuda:0 --seed 1 \
      --grad_accum_chunks $CH "$@" > "$OUT/$TAG/fit_ch$CH.log" 2>&1
    RC=$?
    [ $RC -eq 0 ] && break
    grep -q "CUDA out of memory\|OutOfMemoryError" "$OUT/$TAG/fit_ch$CH.log" || fail "$TAG: fit rc=$RC (fit_ch$CH.log)"
    status "$TAG: out of memory with $CH chunk(s); retrying"
    rm -f "$OUT/$TAG"/*.pt
  done
  [ $RC -eq 0 ] || fail "$TAG: out of memory even with 4 chunks"
  CUDA_VISIBLE_DEVICES="" $PY $MB/refit/check_fit.py --ckpt "$OUT/$TAG/wm.pt" --iters $ITERS --heads 5 $CHECK \
    > "$OUT/$TAG/check.txt" 2>&1 || fail "$TAG: fit check ($TAG/check.txt)"
  status "$TAG: done"
}

dose() {   # DDD
  fit c3_dose$1 "$OUT/c3_data/dose$1" "$NORM" "--state_loss mse --norm_sha256 $(sha256sum "$NORM" | cut -d' ' -f1) \
--data_dir $OUT/c3_data/dose$1 --real_terms $(terms $1)" --
}
NSHA=$(sha256sum "$NORM" | cut -d' ' -f1)
( LANE=C5C6
  fit nll "$DATA" "$NORM" "--state_loss gaussian_nll --norm_sha256 $NSHA --data_dir $DATA" -- --state_loss gaussian_nll \
  && fit c6_widegrav "$DATA" "$WNORM" "--state_loss mse --norm_sha256 $WSHA --data_dir $DATA" -- ) & PA=$!
sleep 20
( LANE=C3a; dose 000 && dose 050 ) & PB=$!
sleep 20
( LANE=C3b; dose 025 && dose 100 ) & PC=$!
RC=0; for P in $PA $PB $PC; do wait $P || RC=1; done
[ $RC -eq 0 ] || fail "a lane failed (status_*.txt)"

# ---- 2. provisional CPU score (held-out envs and feedback batches A/C)
status "CPU score: control, NLL, C3 doses (ETH normalizer); C6 alone"
TRACES=(--trace "heldout1701=$E1V/data/test_baseline1701.npz" --trace "heldout_reference=$E1V/data/test_reference.npz"
        --trace "batchA=$FB/batch_A.npz" --trace "batchC=$FB/batch_C.npz")
( CUDA_VISIBLE_DEVICES="" $PY $MB/e1v_score.py --model "e1v_control=$CONTROL" --model "nll=$OUT/nll/wm.pt" \
    --model "dose000=$OUT/c3_dose000/wm.pt" --model "dose025=$OUT/c3_dose025/wm.pt" \
    --model "dose050=$OUT/c3_dose050/wm.pt" --model "dose100=$OUT/c3_dose100/wm.pt" \
    "${TRACES[@]}" --horizons 1,8,24 --out "$OUT/score" > "$OUT/score.log" 2>&1 ) & S1=$!
( CUDA_VISIBLE_DEVICES="" $PY $MB/e1v_score.py --model "c6_widegrav=$OUT/c6_widegrav/wm.pt" \
    "${TRACES[@]}" --horizons 1,8,24 --out "$OUT/score_c6" > "$OUT/score_c6.log" 2>&1 ) & S2=$!
wait $S1 || fail "CPU score (score.log)"
wait $S2 || fail "CPU score C6 (score_c6.log)"
status "done"
