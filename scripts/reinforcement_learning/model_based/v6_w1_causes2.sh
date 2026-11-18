#!/bin/bash
# Plan v6 W1, second CPU cause queue on batches A and C: C5 (sampled vs mean feedback, aleatoric sigma by
# depth, 1-sigma coverage, scale s(d), Spearman) and C9 (head per step, ensemble mean, most-tilted head)
# with the E1-V control model; C5 sampled feedback also with ETH. Same windows as v6_w1_causes.sh.
# Tables: vs the control in head mode (C5, C9) and vs ETH in head mode (C5 on ETH).
# CPU only. Run from the robotic_world_model root:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_w1_causes2.sh <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
set -uo pipefail
OUT=${1:?fresh output dir}
[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
ETH=assets/models/pretrain_rnn_ens.pt
CTRL=logs/p2/e1v_1113_0939/wm/e1v_wm.pt
FB=logs/p2/e1v_fb_1113_1713
TRACES=(--trace "batchA=$FB/batch_A.npz" --trace "batchC=$FB/batch_C.npz")

mkdir -p "$OUT"
status() { echo "$(date -u +%FT%TZ) [${LANE:-main}] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
for f in "$ETH" "$CTRL" "$FB/batch_A.npz" "$FB/batch_C.npz"; do [ -s "$f" ] || fail "missing $f"; done
{ git rev-parse HEAD; git status --short | grep -v '^??' ; } > "$OUT/git.txt"
sha256sum "$ETH" "$CTRL" "$FB/batch_A.npz" "$FB/batch_C.npz" "$AN/c0_battery.py" "$AN/c0_compare.py" "$0" > "$OUT/inputs.sha256"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_c0_battery.py $AN/tests/test_c0_compare.py > "$OUT/tests.log" 2>&1 \
  || fail "CPU tests failed (tests.log)"
status "tests passed"

c0() {  # c0 <tag> <wm> <feedback>
  status "c0 $1 (feedback $3)"
  CUDA_VISIBLE_DEVICES="" $PY $AN/c0_battery.py --wm "$2" "${TRACES[@]}" --feedback "$3" --out "$OUT/$1" \
    > "$OUT/$1.log" 2>&1 || { status "FAILED c0 $1 ($1.log)"; return 1; }
  status "done c0 $1"
}
( LANE=L1; c0 control_head "$CTRL" head && c0 control_sampled "$CTRL" sampled && c0 control_random_step "$CTRL" random_step ) & P1=$!
( LANE=L2; c0 control_mean "$CTRL" mean && c0 control_most_tilted "$CTRL" most_tilted ) & P2=$!
( LANE=L3; c0 eth_head "$ETH" head && c0 eth_sampled "$ETH" sampled ) & P3=$!
RC=0; for P in $P1 $P2 $P3; do wait $P || RC=1; done
[ $RC -eq 0 ] || fail "a lane failed (queue.log)"

status "compare: control (head) vs feedback modes"
CUDA_VISIBLE_DEVICES="" $PY $AN/c0_compare.py --control "control_head=$OUT/control_head" \
  --treat "C5_sampled=$OUT/control_sampled" --treat "C9_random_step=$OUT/control_random_step" \
  --treat "C9_mean=$OUT/control_mean" --treat "C9_most_tilted=$OUT/control_most_tilted" \
  --out "$OUT/table_control" > "$OUT/table_control.log" 2>&1 || fail "compare control (table_control.log)"
CUDA_VISIBLE_DEVICES="" $PY $AN/c0_compare.py --control "eth_head=$OUT/eth_head" --treat "C5_sampled_eth=$OUT/eth_sampled" \
  --out "$OUT/table_eth" > "$OUT/table_eth.log" 2>&1 || fail "compare ETH (table_eth.log)"
status "done"
cat "$OUT/table_control/compare.txt" "$OUT/table_eth/compare.txt"
for R in control_head control_sampled control_mean eth_head eth_sampled; do
  echo "--- $R (pooled A+C):"; awk '/^== pooled/{p=1;next} /^== /{p=0} p && /C5|C1a|coverage|reach/' "$OUT/$R/c0_report.txt"
done
