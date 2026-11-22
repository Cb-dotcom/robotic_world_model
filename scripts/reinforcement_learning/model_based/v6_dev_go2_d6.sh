#!/bin/bash
# A4 D6 (22 Nov 2026, Charbel's decision 20:47): startup coverage. Real reset-state prefixes in the data, startup windows
# in the fit, D2's recipe otherwise; option 1's stop-layer retrain; Q1-Q3 unchanged plus Q4 startup (A4 §3, 22 Nov 20:47).
# Phases (from the robotic_world_model root in the container); `all` runs them in order and stops at the first failure:
#   prepare  CPU  go2_startup_prepare.py on D2's two recordings (same E22 split, checked against D2's withheld lists)
#   prefit   CPU  go2_d6_prefit.py: (a) first transition, (b) terminal targets, (c) sampler, (d) reset rows; must pass
#   fit      GPU  D2's fit recipe (from scratch, fit seed 2, 2000 it x 20 x 2048, lr 1e-3, pos-weight 1, per-head reset,
#                 D2's frozen normalizer) + startup windows; about 160 min (D2's fit time; estimate)
#   retrain  GPU  option 1 (stop layers only; w chosen on calibration recording 914 as in D2)
#   qualify  CPU  Q1-Q3 (go2_d2_qualify.py, probe 913, round-0 withheld) and Q4 (go2_d6_startup_check.py)
# Example:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_dev_go2_d6.sh all logs/p2/a4_d6_1122 > /tmp/a4_d6.out 2>&1 < /dev/null &
set -uo pipefail
PHASE=${1:?phase: all, prepare, prefit, fit, retrain or qualify}
OUT=${2:?out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
ROUND0=${ROUND0:-logs/p2/go2_round0_1118/round0_s2_seed101.npz}
ROUND0_SHA=${ROUND0_SHA:-3e7ce462a885ebec7e17cf1668d3499a26efe71612b65bd7d55ebf6cbdb41456}
SHIFT=${SHIFT:-logs/p2/a4_d1_1121c/fit_batch.npz}
SHIFT_SHA=${SHIFT_SHA:-d0d011842757980b8c93936736fd953d69d29f02302c628500706e32b4dc9f05}     # run-log 21 Nov (D1 collect)
D2=${D2:-logs/p2/a4_d2_1121}
NORM=$D2/data/go2_norm.npz
D2WM=${D2WM:-$D2/option1/run/d2s_wm_seed2.pt}
D2WM_SHA=${D2WM_SHA:-741b352fab202f0b898d23908ccdc7f3453409492f970feacc145476cdd16d11}          # run-log 22 Nov (option 1)
CAL=${CAL:-logs/p2/a4_d1_cal3/probe.npz}
CAL_SHA=${CAL_SHA:-62e204297c5c08ed5692242ce2a4ee85598fb4a50a1913551eb3b6d0ff8b7db2}            # run-log 22 Nov
PROBE3=${PROBE3:-logs/p2/a4_d1_probe3/probe.npz}
PROBE3_SHA=${PROBE3_SHA:-0acfd1b153f025073e30cdfe0373da4931292054301857b85666c3089e899284}      # run-log 22 Nov
D3PROBE=${D3PROBE:-logs/p2/a4_d4_1122/rec/probe.npz}
D3PROBE_SHA=${D3PROBE_SHA:-3893daac560c18cfdd4c599d2dfe44d76cd46f86f91250203677e96086fa9405}    # run-log 22 Nov (D4)
FIT_SEED=2
WM=$OUT/wm/d6_wm_seed2.pt
WMS=$OUT/option1/run/d6s_wm_seed2.pt
export PYTHONUNBUFFERED=1 PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
mkdir -p "$OUT" || exit 1
status() { echo "$(date -u +%FT%TZ) [a4-d6 $PHASE] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
sha_ok() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || fail "$1: hash differs from the run-log"; }
gpu_wait() { until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$1" ]; do
  status "waiting for $(( $1 / 1024 )) GiB GPU before $2"; sleep 300; done; }
TESTS="$AN/tests/test_go2_startup_prepare.py $AN/tests/test_go2_d6_prefit.py $AN/tests/test_go2_startup_fit.py \
$AN/tests/test_go2_startup_resets.py $AN/tests/test_go2_d6_startup_check.py $AN/tests/test_go2_stop_retrain.py \
$AN/tests/test_go2_d2_qualify.py $AN/tests/test_warm_resets.py"

phase_prepare() {
  [ -e "$OUT/data" ] && { status "exists: $OUT/data"; exit 2; }
  sha_ok "$ROUND0" "$ROUND0_SHA"; sha_ok "$SHIFT" "$SHIFT_SHA"
  { git rev-parse HEAD; git status --short | grep -v '^??'; sha256sum $MB/go2_startup_prepare.py $AN/go2_d6_prefit.py \
    $AN/go2_d6_startup_check.py $MB/fit_world_model_go2_pilot_segments.py $MB/envs/base.py; } > "$OUT/git.txt"
  CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $TESTS > "$OUT/tests.log" 2>&1 || fail "tests (tests.log)"
  grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log)"
  status "tests: $(tail -1 "$OUT/tests.log")"
  CUDA_VISIBLE_DEVICES="" $PY $MB/go2_startup_prepare.py --round0 "$ROUND0" --shift "$SHIFT" --out "$OUT/data" \
    --d2-data "$D2/data" > "$OUT/prepare.log" 2>&1 || { tail -5 "$OUT/prepare.log"; fail "prepare (prepare.log)"; }
  grep "^\[d6\]" "$OUT/prepare.log" | tee -a "$OUT/queue.log"
  chmod -R a-w "$OUT/data"
  status "data ready, write-protected; manifest $(sha256sum "$OUT/data/manifest.json" | cut -c1-12)"
}
phase_prefit() {
  [ -e "$OUT/prefit" ] && { status "exists: $OUT/prefit"; exit 2; }
  [ -s "$OUT/data/manifest.json" ] || fail "no prepared data"
  sha_ok "$D2WM" "$D2WM_SHA"
  CUDA_VISIBLE_DEVICES="" $PY $AN/go2_d6_prefit.py --data "$OUT/data" --round0 "$ROUND0" --shift "$SHIFT" --wm "$D2WM" \
    --norm "$NORM" --out "$OUT/prefit" > "$OUT/prefit.log" 2>&1
  RC=$?
  [ -s "$OUT/prefit/prefit.txt" ] && cat "$OUT/prefit/prefit.txt" | tee -a "$OUT/queue.log"
  [ $RC -eq 5 ] && fail "pre-fit verification failed: the fit does not start"
  [ $RC -eq 0 ] || { tail -5 "$OUT/prefit.log"; fail "pre-fit verification (prefit.log)"; }
}
phase_fit() {
  [ -e "$WM" ] && { status "exists: $WM"; exit 2; }
  grep -q "PRE-FIT: PASS" "$OUT/prefit/prefit.txt" 2>/dev/null || fail "the pre-fit verification has not passed"
  NORM_SHA=$(sha256sum "$NORM" | cut -d' ' -f1)
  mkdir -p "$OUT/wm"
  gpu_wait 12288 "the fit"
  T0=$(date -u +%s)
  status "fit seed $FIT_SEED: D2 recipe + startup windows; D2 normalizer $NORM_SHA"
  CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config go2_flat --data "$OUT/data/segments" \
    --output "$WM" --norm_stats "$NORM" --seed $FIT_SEED --iterations 2000 --num_mini_batches 20 \
    --mini_batch_size 2048 --lr 1e-3 --weight_decay 0 --max_grad_norm 1 --termination_loss_weight 1 \
    --termination_pos_weight 1 --lazy_windows --save_interval 500 --skip_short_segments \
    --startup_segments "$OUT/data/startup.json" --device cuda:0 > "$OUT/fit.log" 2>&1 || fail "fit (fit.log)"
  [ -s "$WM" ] || fail "the fit wrote no $WM"
  grep -q "normalizer loaded from .*$NORM_SHA" "$OUT/fit.log" || fail "the fit did not use D2's normalizer (fit.log)"
  grep "^\[startup\]" "$OUT/fit.log" | tee -a "$OUT/queue.log"
  CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
p = ck.get('provenance', {}); sys.exit(0 if p.get('b3_reset_per_head') is True and ck.get('iter') == 2000 and ck.get('normalized') \
and (p.get('startup') or {}).get('windows', 0) > 0 else 1)" "$WM" \
    || fail "fit checkpoint lacks the per-head reset, iteration 2000, its normalizer or the startup windows"
  grep -E "startup_minibatches=" "$OUT/fit.log" | tail -1 | tee -a "$OUT/queue.log"
  status "fitted in $(( ($(date -u +%s) - T0) / 60 )) min: $(sha256sum "$WM")"
}
phase_retrain() {
  [ -e "$OUT/option1" ] && { status "exists: $OUT/option1"; exit 2; }
  [ -s "$WM" ] || fail "no fitted model"
  sha_ok "$CAL" "$CAL_SHA"
  mkdir -p "$OUT/option1"
  gpu_wait 4096 "the stop-layer retrain"
  CUDA_VISIBLE_DEVICES=0 $PY $AN/go2_stop_retrain.py --wm "$WM" --norm "$NORM" --segments "$OUT/data/segments" \
    --calibration "$CAL" --out "$OUT/option1/run" --device cuda:0 --model-name d6s_wm_seed2.pt > "$OUT/option1/retrain.log" 2>&1
  RC=$?
  grep "^\[option1\]" "$OUT/option1/retrain.log" | tee -a "$OUT/queue.log"
  [ $RC -eq 6 ] && fail "no stop-layer candidate meets false stops <= 0.10: D6 fails (A4 §3)"
  [ $RC -eq 0 ] || { tail -5 "$OUT/option1/retrain.log"; fail "stop retraining (option1/retrain.log)"; }
  status "retrained model $(sha256sum "$WMS" | cut -d' ' -f1)"
}
phase_qualify() {
  [ -e "$OUT/qualify" ] && { status "exists: $OUT/qualify"; exit 2; }
  [ -s "$WMS" ] || fail "no retrained model"
  sha_ok "$PROBE3" "$PROBE3_SHA"; sha_ok "$D3PROBE" "$D3PROBE_SHA"
  CUDA_VISIBLE_DEVICES="" $PY $AN/go2_d2_qualify.py --wm "$WMS" --norm "$NORM" --round0 "$ROUND0" \
    --r0-withheld "$OUT/data/r0/withheld.json" --probe "$PROBE3" --out "$OUT/qualify" > "$OUT/qualify.log" 2>&1 \
    || { tail -5 "$OUT/qualify.log"; fail "Q1-Q3 ($OUT/qualify.log)"; }
  cat "$OUT/qualify/d2_qualify.txt" | tee -a "$OUT/queue.log"
  CUDA_VISIBLE_DEVICES="" $PY $AN/go2_d6_startup_check.py --wm "$WMS" --norm "$NORM" --probe "s2_913=$PROBE3" \
    --withheld "r0_withheld=$ROUND0:$OUT/data/r0/withheld.json" --descriptive "d3_915=$D3PROBE" \
    --out "$OUT/startup" > "$OUT/startup.log" 2>&1
  RC=$?
  [ -s "$OUT/startup/startup_check.txt" ] && cat "$OUT/startup/startup_check.txt" | tee -a "$OUT/queue.log"
  [ $RC -eq 0 ] || [ $RC -eq 7 ] || { tail -5 "$OUT/startup.log"; fail "Q4 ($OUT/startup.log)"; }
  Q13=$(CUDA_VISIBLE_DEVICES="" $PY -c "import json, sys; print(json.load(open(sys.argv[1]))['D2_pass'])" "$OUT/qualify/d2_qualify.json" | tail -1)
  if [ "$Q13" = True ] && [ $RC -eq 0 ]; then V=PASS; else V=FAIL; fi
  status "VERDICT D6: $V (Q1-Q3 $Q13, Q4 $([ $RC -eq 0 ] && echo PASS || echo FAIL)); model $(sha256sum "$WMS" | cut -d' ' -f1)"
}

case "$PHASE" in
all) phase_prepare; phase_prefit; phase_fit; phase_retrain; phase_qualify ;;
prepare) phase_prepare ;; prefit) phase_prefit ;; fit) phase_fit ;; retrain) phase_retrain ;; qualify) phase_qualify ;;
*) echo "phase must be all, prepare, prefit, fit, retrain or qualify"; exit 2 ;;
esac
