#!/bin/bash
# A4 D2 (21 Nov 2026): one mixed-data Go2 world model (round-0 nominal + D1 shift batch at 0.30 x friction).
# Phase fit: input hash checks -> tests -> go2_mixed_prepare.py (E22 split of each source, pooled segments, pooled
# normalizer, pooled start data; data folder write-protected) -> fit with the A3 §3 settings, fit seed 2.
# Phase qualify (same out dir, CPU): v6_dev_go2_d2.sh qualify logs/p2/a4_d2_1121 (Q1-Q3, A4 §1; rules in go2_d2_qualify.py).
# Usage, from the robotic_world_model root in the container, with setsid nohup:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_dev_go2_d2.sh fit logs/p2/a4_d2_1121 > /tmp/a4_d2.out 2>&1 < /dev/null &
set -uo pipefail
PHASE=${1:?fit}
OUT=${2:?fresh out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
ROUND0=${ROUND0:-logs/p2/go2_round0_1118/round0_s2_seed101.npz}
ROUND0_SHA=${ROUND0_SHA:-3e7ce462a885ebec7e17cf1668d3499a26efe71612b65bd7d55ebf6cbdb41456}
SHIFT=${SHIFT:-logs/p2/a4_d1_1121c/fit_batch.npz}
SHIFT_SHA=${SHIFT_SHA:-d0d011842757980b8c93936736fd953d69d29f02302c628500706e32b4dc9f05}   # run-log 21 Nov (D1 collect)
A3_WITHHELD=${A3_WITHHELD:-logs/p2/a3_g0_1121/data/withheld.json}
FIT_SEED=2
export PYTHONUNBUFFERED=1 PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
status() { echo "$(date -u +%FT%TZ) [a4-d2 $PHASE] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
PROBE=${PROBE:-logs/p2/a4_d1_1121c/probe.npz}
PROBE_SHA=${PROBE_SHA:-ac67b6ec38cc3e4b9dac3a097f0de32ce9c576665c5fe1af6d04927cbddf796e}   # run-log 21 Nov (D1 collect)
if [ "$PHASE" = qualify ]; then
  # Q1-Q3 (A4 §1, §3) on data never fitted; rules fixed in analysis/go2_d2_qualify.py before the model was read.
  WM="$OUT/wm/d2_wm_seed2.pt"; Q="$OUT/qualify"
  [ -s "$WM" ] || { echo "no fitted model $WM"; exit 1; }
  [ -e "$Q" ] && { echo "exists: $Q"; exit 2; }
  [ "$(sha256sum "$PROBE" | cut -d' ' -f1)" = "$PROBE_SHA" ] || fail "probe hash differs from the run-log"
  [ -z "${WM_SHA:-}" ] || [ "$(sha256sum "$WM" | cut -d' ' -f1)" = "$WM_SHA" ] || fail "model hash differs from WM_SHA"
  mkdir -p "$Q"
  { git rev-parse HEAD; git status --short | grep -v '^??'; sha256sum $AN/go2_d2_qualify.py; } > "$Q/git.txt"
  CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_d2_qualify.py > "$Q/tests.log" 2>&1 || fail "qualify tests ($Q/tests.log)"
  grep -qE "[0-9]+ skipped" "$Q/tests.log" && fail "qualify tests were skipped"
  status "qualify tests: $(tail -1 "$Q/tests.log"); model $(sha256sum "$WM" | cut -d' ' -f1)"
  CUDA_VISIBLE_DEVICES="" $PY $AN/go2_d2_qualify.py --wm "$WM" --norm "$OUT/data/go2_norm.npz" --round0 "$ROUND0" \
    --r0-withheld "$OUT/data/r0/withheld.json" --probe "$PROBE" --out "$Q/run" > "$Q/qualify.log" 2>&1 \
    || { tail -5 "$Q/qualify.log"; fail "qualification ($Q/qualify.log)"; }
  cat "$Q/run/d2_qualify.txt" | tee -a "$OUT/queue.log"
  status "qualified; $(sha256sum "$Q/run/d2_qualify.json")"
  exit 0
fi
[ "$PHASE" = fit ] || { echo "phase must be fit or qualify"; exit 2; }
[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
for f in "$ROUND0" "$SHIFT" "$A3_WITHHELD"; do [ -s "$f" ] || fail "missing $f"; done
[ "$(sha256sum "$ROUND0" | cut -d' ' -f1)" = "$ROUND0_SHA" ] || fail "round-0 hash differs from the run-log"
[ "$(sha256sum "$SHIFT" | cut -d' ' -f1)" = "$SHIFT_SHA" ] || fail "D1 fitting-batch hash differs from the run-log"
{ git rev-parse HEAD; git status --short | grep -v '^??'; } > "$OUT/git.txt"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_mixed_prepare.py $AN/tests/test_go2_g0.py > "$OUT/tests.log" 2>&1 \
  || fail "tests (tests.log)"
grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log)"
status "tests: $(tail -1 "$OUT/tests.log")"
CUDA_VISIBLE_DEVICES="" $PY $MB/go2_mixed_prepare.py --round0 "$ROUND0" --shift "$SHIFT" --out "$OUT/data" \
  --a3-withheld "$A3_WITHHELD" 2>&1 | tee -a "$OUT/queue.log" || fail "mixed manifest"
chmod -R a-w "$OUT/data"
NORM_SHA=$(sha256sum "$OUT/data/go2_norm.npz" | cut -d' ' -f1)
status "manifest $(sha256sum "$OUT/data/manifest.json" | cut -d' ' -f1); normalizer $NORM_SHA"
mkdir -p "$OUT/wm"
WM="$OUT/wm/d2_wm_seed$FIT_SEED.pt"
until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12288 ]; do
  status "waiting for 12 GiB GPU before the fit"; sleep 300; done
T0=$(date -u +%s)
status "fit seed $FIT_SEED: from scratch, 2000 it x 20 x 2048, lr 1e-3, pos-weight 1, per-head reset (A3 §3 settings)"
CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config go2_flat --data "$OUT/data/segments" \
  --output "$WM" --norm_stats "$OUT/data/go2_norm.npz" --seed $FIT_SEED --iterations 2000 --num_mini_batches 20 \
  --mini_batch_size 2048 --lr 1e-3 --weight_decay 0 --max_grad_norm 1 --termination_loss_weight 1 \
  --termination_pos_weight 1 --lazy_windows --save_interval 500 --device cuda:0 > "$OUT/fit.log" 2>&1 || fail "fit (fit.log)"
[ -s "$WM" ] || fail "the fit wrote no $WM"
grep -q "normalizer loaded from .*$NORM_SHA" "$OUT/fit.log" || fail "the fit did not use the pooled normalizer (fit.log)"
CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
sys.exit(0 if ck.get('provenance', {}).get('b3_reset_per_head') is True and ck.get('iter') == 2000 and ck.get('normalized') else 1)" "$WM" \
  || fail "fit checkpoint lacks the per-head reset, iteration 2000 or its normalizer"
grep -E "total_real_terms|pos_weight" "$OUT/fit.log" | tail -2 | tee -a "$OUT/queue.log"
status "fitted in $(( ($(date -u +%s) - T0) / 60 )) min: $(sha256sum "$WM")"
