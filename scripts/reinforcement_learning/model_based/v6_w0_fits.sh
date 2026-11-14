#!/bin/bash
# Plan v6 W0 (14-15 Nov): the three ANYmal refits for the "why" battery, on the E1-V round-0 data.
# Run from the robotic_world_model root inside the container:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_w0_fits.sh <e1v_out_dir> <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
#
# Common recipe = the E1-V refit (control): ETH start, data $E1V/data/segments (1701 + reference,
# envs 52-63 held out), ETH normalizer, per-head memory reset (B3 wrapper), 500 it, lr 1e-4,
# termination pos-weight 1, lazy windows. Each fit changes one thing:
#   C4  forecast horizon N = 24 (control N = 8), full warm start, seed 1            -> c4_n24/wm.pt
#   C8  20 heads: ETH trunks, 20 fresh heads (seed 1), head lr 1e-3                 -> c8_k20/wm.pt
#   C1c 5 single-head models, member i = ETH trunks + ETH head i, seeds 1-5         -> c1c_m{0..4}/wm.pt
# On CUDA out-of-memory a fit is retried with --grad_accum_chunks 2, then 4 (same samples per step).
# Last step (CPU): score the E1-V refit, C4 and C8 on the held-out 1701 test envs, horizons 1/8/24.
# No policy training. Lanes run in parallel; one GPU start at a time.
set -uo pipefail
E1V=${1:?e1v_queue output dir}
OUT=${2:?fresh output dir}
MB=scripts/reinforcement_learning/model_based
ETH=assets/models/pretrain_rnn_ens.pt
ETH_SHA=2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a
DATA="$E1V/data/segments"; NORM="$E1V/data/eth_norm.npz"; CONTROL="$E1V/wm/e1v_wm.pt"
PY=/isaac-sim/python.sh

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
LANE=main
status() { echo "$(date -u +%FT%TZ) [$LANE] $*" | tee "$OUT/status_$LANE.txt" >> "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
acquire_gpu() {   # NAME MIB
  until mkdir "$OUT/gpu.lock" 2>/dev/null; do sleep 20; done
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$2" ]; do
    status "waiting for $2 MiB GPU before $1"; sleep 300; done; }
release_gpu_later() { ( sleep 240; rmdir "$OUT/gpu.lock" 2>/dev/null ) & }

status "preflight"
git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; git -C ../rsl_rl_rwm rev-parse HEAD; } > "$OUT/git.txt"
[ "$(sha256sum "$ETH" | cut -d' ' -f1)" = "$ETH_SHA" ] || fail "ETH checkpoint hash differs"
[ -s "$DATA/manifest.csv" ] && [ -s "$NORM" ] && [ -s "$CONTROL" ] || fail "E1-V data, normalizer or control model missing"
sha256sum "$ETH" "$NORM" "$CONTROL" "$DATA/manifest.csv" $MB/fit_world_model_go2_pilot_segments.py \
  $MB/refit/fit_reset_per_head.py $MB/v6_w0_fits.sh $MB/e1v_score.py > "$OUT/inputs.sha256"
status "CPU tests"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $MB/analysis/tests/test_partial_warm_start.py > "$OUT/tests.log" 2>&1 \
  || fail "CPU tests failed (tests.log)"

fit() {   # TAG MIB extra args...
  local TAG=$1 MIB=$2; shift 2
  local CH RC
  for CH in 1 2 4; do
    acquire_gpu "$TAG" "$MIB"; status "fit $TAG (grad_accum_chunks $CH)"; release_gpu_later
    mkdir -p "$OUT/$TAG"
    CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config anymal_d_flat --data "$DATA" \
      --output "$OUT/$TAG/wm.pt" --reference_wm "$ETH" --norm_stats "$NORM" --iterations 500 --lr 1e-4 \
      --termination_pos_weight 1 --lazy_windows --save_interval 100 --device cuda:0 --grad_accum_chunks $CH "$@" \
      > "$OUT/$TAG/fit_ch$CH.log" 2>&1
    RC=$?
    [ $RC -eq 0 ] && break
    grep -q "CUDA out of memory\|OutOfMemoryError" "$OUT/$TAG/fit_ch$CH.log" || fail "$TAG: fit rc=$RC (fit_ch$CH.log)"
    status "$TAG: out of memory with $CH chunk(s); retrying"
    rm -f "$OUT/$TAG"/*.pt
  done
  [ $RC -eq 0 ] || fail "$TAG: out of memory even with 4 chunks"
  CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
sys.exit(0 if ck.get('iter') == 500 and ck.get('provenance', {}).get('b3_reset_per_head') is True else 1)" \
    "$OUT/$TAG/wm.pt" || fail "$TAG: checkpoint lacks b3 provenance or 500 iterations"
  status "$TAG: done"
}

( LANE=C4; fit c4_n24 16384 --warm_start --forecast_horizon 24 --seed 1 ) & P4=$!
sleep 20
( LANE=C8; fit c8_k20 16384 --ensemble_size 20 --warm_start_trunk --head_lr 1e-3 --seed 1 ) & P8=$!
sleep 20
( LANE=C1c; for I in 0 1 2 3 4; do fit c1c_m$I 8192 --ensemble_size 1 --warm_start_member $I --seed $((I + 1)); done ) & P1=$!
RC=0; for P in $P4 $P8 $P1; do wait $P || RC=1; done
[ $RC -eq 0 ] || fail "a lane failed (status_*.txt)"

status "CPU score: E1-V refit (control), C4, C8 on held-out 1701 envs"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1v_score.py --model "e1v_n8_k5=$CONTROL" --model "c4_n24=$OUT/c4_n24/wm.pt" \
  --model "c8_k20=$OUT/c8_k20/wm.pt" --trace "heldout1701=$E1V/data/test_baseline1701.npz" \
  --trace "heldout_reference=$E1V/data/test_reference.npz" --horizons 1,8,24 --out "$OUT/score" > "$OUT/score.log" 2>&1 \
  || fail "CPU score (score.log)"
status "done"
