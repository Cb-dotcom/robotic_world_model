#!/bin/bash
# Plan v6 W1: CPU cause battery on the fix-selection batches A and C (plan §11 #4), with the models fitted
# in W0/W1. One c0_battery run per model (same windows: same traces and seed), then c0_compare tables:
#   table 1 (control = E1-V refit seed 1): noise floor seeds 2, 3; C4 (N = 24), C8 (20 heads, subsets
#            2/5/10), C1c (5 independent single-head trunks as one ensemble), C5 (Gaussian NLL),
#            C6 (widened gravity std), C3 doses 0/25/50/100 %, C7 (cold start, control)
#   table 2 (control = ETH): C3(1) = the E1-V refit (trained with falls) vs ETH; C7 (cold start, ETH)
# Also K-subsampling 2-4 on the control's own 5 heads (C8 first part) inside the control run.
# CPU only (CUDA_VISIBLE_DEVICES=""), three lanes; no GPU, no training. Run from the robotic_world_model root:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_w1_causes.sh <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
set -uo pipefail
OUT=${1:?fresh output dir}
[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
ETH=assets/models/pretrain_rnn_ens.pt
CTRL=logs/p2/e1v_1113_0939/wm/e1v_wm.pt
W0=logs/p2/v6w0_1114_1457
W1=logs/p2/v6w1_1115_1022
FB=logs/p2/e1v_fb_1113_1713
TRACES=(--trace "batchA=$FB/batch_A.npz" --trace "batchC=$FB/batch_C.npz")
C1C="$W0/c1c_m0/wm.pt,$W0/c1c_m1/wm.pt,$W0/c1c_m2/wm.pt,$W0/c1c_m3/wm.pt,$W0/c1c_m4/wm.pt"

mkdir -p "$OUT"
status() { echo "$(date -u +%FT%TZ) [${LANE:-main}] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }

# ---- 0. inputs, provenance, tests
MODELS=("$ETH" "$CTRL" "$W1/c0_seed2/wm.pt" "$W1/c0_seed3/wm.pt" "$W0/c4_n24/wm.pt" "$W0/c8_k20/wm.pt"
        "$W1/nll/wm.pt" "$W1/c6_widegrav/wm.pt" "$W1/c3_dose000/wm.pt" "$W1/c3_dose025/wm.pt"
        "$W1/c3_dose050/wm.pt" "$W1/c3_dose100/wm.pt" ${C1C//,/ })
for f in "${MODELS[@]}" "$FB/batch_A.npz" "$FB/batch_C.npz"; do [ -s "$f" ] || fail "missing $f"; done
{ git rev-parse HEAD; git status --short | grep -v '^??' ; } > "$OUT/git.txt"
sha256sum "${MODELS[@]}" "$FB/batch_A.npz" "$FB/batch_C.npz" "$AN/c0_battery.py" "$AN/c0_compare.py" "$0" > "$OUT/inputs.sha256"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_c0_battery.py $AN/tests/test_c0_compare.py > "$OUT/tests.log" 2>&1 \
  || fail "CPU tests failed (tests.log)"
status "tests passed; starting three CPU lanes"

c0() {  # c0 <tag> <wm> [extra args]
  local tag=$1 wm=$2; shift 2
  status "c0 $tag"
  CUDA_VISIBLE_DEVICES="" $PY $AN/c0_battery.py --wm "$wm" "${TRACES[@]}" --out "$OUT/$tag" "$@" \
    > "$OUT/$tag.log" 2>&1 || { status "FAILED c0 $tag ($tag.log)"; return 1; }
  status "done c0 $tag"
}

# ---- 1. one battery run per model (three lanes)
( LANE=L1; c0 control "$CTRL" --head-subsets 2,3,4 && c0 seed2 "$W1/c0_seed2/wm.pt" && c0 seed3 "$W1/c0_seed3/wm.pt" \
    && c0 c4_n24 "$W0/c4_n24/wm.pt" && c0 control_cold "$CTRL" --imag-context 1 ) & P1=$!
( LANE=L2; c0 c8_k20 "$W0/c8_k20/wm.pt" --head-subsets 2,5,10 && c0 c1c "$C1C" ) & P2=$!
( LANE=L3; c0 eth "$ETH" && c0 eth_cold "$ETH" --imag-context 1 && c0 nll "$W1/nll/wm.pt" \
    && c0 c6_widegrav "$W1/c6_widegrav/wm.pt" && c0 dose000 "$W1/c3_dose000/wm.pt" && c0 dose025 "$W1/c3_dose025/wm.pt" \
    && c0 dose050 "$W1/c3_dose050/wm.pt" && c0 dose100 "$W1/c3_dose100/wm.pt" ) & P3=$!
RC=0; for P in $P1 $P2 $P3; do wait $P || RC=1; done
[ $RC -eq 0 ] || fail "a lane failed (queue.log)"

# ---- 2. cause tables
status "compare: table 1 (control = E1-V refit seed 1)"
CUDA_VISIBLE_DEVICES="" $PY $AN/c0_compare.py --control "control=$OUT/control" \
  --noise "seed2=$OUT/seed2" --noise "seed3=$OUT/seed3" \
  --treat "C4_n24=$OUT/c4_n24" --treat "C8_k20=$OUT/c8_k20" --treat "C1c_indep=$OUT/c1c" \
  --treat "C5_nll=$OUT/nll" --treat "C6_widegrav=$OUT/c6_widegrav" \
  --treat "C3_dose000=$OUT/dose000" --treat "C3_dose025=$OUT/dose025" --treat "C3_dose050=$OUT/dose050" \
  --treat "C3_dose100=$OUT/dose100" --treat "C7_cold=$OUT/control_cold" \
  --out "$OUT/table_control" > "$OUT/table_control.log" 2>&1 || fail "compare table 1 (table_control.log)"
status "compare: table 2 (control = ETH)"
CUDA_VISIBLE_DEVICES="" $PY $AN/c0_compare.py --control "eth=$OUT/eth" \
  --treat "C3-1_e1v_refit=$OUT/control" --treat "C7_cold_eth=$OUT/eth_cold" \
  --out "$OUT/table_eth" > "$OUT/table_eth.log" 2>&1 || fail "compare table 2 (table_eth.log)"
status "done"
cat "$OUT/table_control/compare.txt" "$OUT/table_eth/compare.txt"
for R in control c8_k20 c1c eth eth_cold control_cold; do
  echo "--- $R (pooled A+C):"; awk '/^== pooled/{p=1;next} /^== /{p=0} p && /HR\(0\)|C0 rule|C1a|C8/' "$OUT/$R/c0_report.txt"
done
