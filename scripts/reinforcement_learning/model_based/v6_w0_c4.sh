#!/bin/bash
# Plan v6 W0: rerun C4 (forecast N = 24) inside an existing v6_w0_fits.sh output folder, then the CPU score.
# The first C4 attempt failed because 41-56-step episodes hold no 56-step window; the fitter now skips
# such segments with --skip_short_segments (names recorded in the checkpoint provenance).
#   setsid nohup scripts/reinforcement_learning/model_based/v6_w0_c4.sh <e1v_out_dir> <v6w0_out_dir> > <v6w0_out_dir>/c4_rerun.nohup 2>&1 < /dev/null &
# Shares the folder's GPU lock with the running C8 and C1c lanes. After C4, waits for C8 and C1c to
# finish, then scores the E1-V refit (control), C4 and C8 on the held-out envs (horizons 1/8/24).
set -uo pipefail
E1V=${1:?e1v_queue output dir}
OUT=${2:?existing v6_w0_fits.sh output dir}
MB=scripts/reinforcement_learning/model_based
ETH=assets/models/pretrain_rnn_ens.pt
DATA="$E1V/data/segments"; NORM="$E1V/data/eth_norm.npz"; CONTROL="$E1V/wm/e1v_wm.pt"
PY=/isaac-sim/python.sh
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
LANE=C4rerun
[ -d "$OUT" ] || { echo "no folder $OUT"; exit 2; }
status() { echo "$(date -u +%FT%TZ) [$LANE] $*" | tee "$OUT/status_$LANE.txt" >> "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
acquire_gpu() {
  until mkdir "$OUT/gpu.lock" 2>/dev/null; do sleep 20; done
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$2" ]; do
    status "waiting for $2 MiB GPU before $1"; sleep 300; done; }
release_gpu_later() { ( sleep 240; rmdir "$OUT/gpu.lock" 2>/dev/null ) & }

git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
git rev-parse HEAD > "$OUT/git_c4_rerun.txt"
[ -e "$OUT/c4_n24/wm.pt" ] && fail "c4_n24/wm.pt already exists"
mkdir -p "$OUT/c4_n24"
for f in "$OUT"/c4_n24/fit_ch*.log; do [ -e "$f" ] && mv "$f" "${f%.log}_attempt1_short_segments.log"; done
for CH in 1 2 4; do
  acquire_gpu "c4_n24" 16384; status "fit c4_n24 (grad_accum_chunks $CH)"; release_gpu_later
  CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config anymal_d_flat --data "$DATA" \
    --output "$OUT/c4_n24/wm.pt" --reference_wm "$ETH" --norm_stats "$NORM" --iterations 500 --lr 1e-4 \
    --termination_pos_weight 1 --lazy_windows --save_interval 100 --device cuda:0 --grad_accum_chunks $CH \
    --warm_start --forecast_horizon 24 --skip_short_segments --seed 1 > "$OUT/c4_n24/fit_ch$CH.log" 2>&1
  RC=$?
  [ $RC -eq 0 ] && break
  grep -q "CUDA out of memory\|OutOfMemoryError" "$OUT/c4_n24/fit_ch$CH.log" || fail "c4_n24: fit rc=$RC (fit_ch$CH.log)"
  status "c4_n24: out of memory with $CH chunk(s); retrying"; rm -f "$OUT"/c4_n24/*.pt
done
[ $RC -eq 0 ] || fail "c4_n24: out of memory even with 4 chunks"
CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
sys.exit(0 if ck.get('iter') == 500 and ck.get('provenance', {}).get('b3_reset_per_head') is True else 1)" \
  "$OUT/c4_n24/wm.pt" || fail "c4_n24: checkpoint lacks b3 provenance or 500 iterations"
status "c4_n24: done; waiting for C8 and C1c"
until grep -q "c8_k20: done" "$OUT/status_C8.txt" 2>/dev/null && grep -q "c1c_m4: done" "$OUT/status_C1c.txt" 2>/dev/null; do
  grep -q "FAILED" "$OUT/status_C8.txt" "$OUT/status_C1c.txt" 2>/dev/null && fail "C8 or C1c failed (status_*.txt)"
  sleep 300; done
status "CPU score: E1-V refit (control), C4, C8 on held-out 1701 and reference envs"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1v_score.py --model "e1v_n8_k5=$CONTROL" --model "c4_n24=$OUT/c4_n24/wm.pt" \
  --model "c8_k20=$OUT/c8_k20/wm.pt" --trace "heldout1701=$E1V/data/test_baseline1701.npz" \
  --trace "heldout_reference=$E1V/data/test_reference.npz" --horizons 1,8,24 --out "$OUT/score" > "$OUT/score.log" 2>&1 \
  || fail "CPU score (score.log)"
status "done"
