#!/bin/bash
# Plan v6 W1 rerun after the 18 Nov external review. Reruns C0 and every CPU cause run of queues 1-3 with
# the corrected analysis code:
#   * fall / non-fall ratios (C0 rule, C6 saturation) use jointly resampled matched pairs;
#   * Spearman uses average ranks (ties) and gives NaN for a constant input;
#   * tilt is an angle in degrees (atan2(|g_xy|, -g_z)), any predicted vector length; |g| is reported;
#   * C9 random_step / most_tilted use the termination probability of the head whose state is fed; every
#     mode also stores the ensemble-mean probability (reach is reported under both rules);
#   * a failed 27 Oct reproduction or HR(0) stops the analysis (reports kept); a zero head spread stops it.
# Supersedes logs/p2/c0_1118, v6w1_causes_1118 and v6w1_causes2_1118 (kept, not deleted). Queue 3 was not run.
#
# Order: tests -> C0 (ETH, the three C0 sources, reproduction gate). If C0 stops, nothing else runs.
# Then four CPU lanes of c0 runs on batches A and C, C6 (three models), C2 (four runs), five tables.
# The C0 traces are read from the 18 Nov C0 report (paths and SHA-256 must still match).
# Override paths with env vars C0OLD, REPRO_TRACE, REPRO_JSON if the defaults do not exist.
# CPU only. Run from the robotic_world_model root:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_w1_rerun.sh <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
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
CSV=assets/data/state_action_data_0.csv
REFTRACE=logs/p2/e1v_1113_0939/record/reference.npz
C0OLD=${C0OLD:-logs/p2/c0_1118/c0_report.json}
REPRO_TRACE=${REPRO_TRACE:-logs/p2/transition_fidelity.ELf50z/baseline.npz}
REPRO_JSON=${REPRO_JSON:-logs/p2/paired_fall_replay.cqaEuU/paired_fall_replay.json}
TRACES=(--trace "batchA=$FB/batch_A.npz" --trace "batchC=$FB/batch_C.npz")
C1C="$W0/c1c_m0/wm.pt,$W0/c1c_m1/wm.pt,$W0/c1c_m2/wm.pt,$W0/c1c_m3/wm.pt,$W0/c1c_m4/wm.pt"
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$OUT"
status() { echo "$(date -u +%FT%TZ) [${LANE:-main}] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }

# ---- 0. inputs, provenance, tests
MODELS=("$ETH" "$CTRL" "$W1/c0_seed2/wm.pt" "$W1/c0_seed3/wm.pt" "$W0/c4_n24/wm.pt" "$W0/c8_k20/wm.pt"
        "$W1/nll/wm.pt" "$W1/c6_widegrav/wm.pt" "$W1/c3_dose000/wm.pt" "$W1/c3_dose025/wm.pt"
        "$W1/c3_dose050/wm.pt" "$W1/c3_dose100/wm.pt" ${C1C//,/ })
for f in "${MODELS[@]}" "$FB/batch_A.npz" "$FB/batch_C.npz" "$CSV" "$C0OLD" "$REPRO_TRACE" "$REPRO_JSON"; do
  [ -s "$f" ] || fail "missing $f (set C0OLD / REPRO_TRACE / REPRO_JSON if it moved)"
done
$PY -c '
import json, sys
for name, v in json.load(open(sys.argv[1]))["meta"]["inputs"].items():
    print("C0TRACE", name, v["path"], v["sha256"])
' "$C0OLD" | grep '^C0TRACE ' > "$OUT/c0_traces.txt" || fail "cannot read the C0 inputs from $C0OLD"
C0TRACES=()
while read -r _ name path sha; do
  [ -s "$path" ] || fail "C0 trace $name missing: $path"
  [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$sha" ] || fail "C0 trace $name changed since 18 Nov: $path"
  C0TRACES+=(--trace "$name=$path")
done < "$OUT/c0_traces.txt"
[ ${#C0TRACES[@]} -eq 6 ] || fail "expected 3 C0 sources in $C0OLD, got $(( ${#C0TRACES[@]} / 2 ))"
REFARGS=(--reference-csv "$CSV")
if [ -s "$REFTRACE" ]; then REFARGS+=(--reference-trace "$REFTRACE"); else status "no $REFTRACE: C2 reference = ETH CSV only"; fi
{ git rev-parse HEAD; git status --short | grep -v '^??' ; } > "$OUT/git.txt"
sha256sum "${MODELS[@]}" "$FB/batch_A.npz" "$FB/batch_C.npz" "$CSV" "$REPRO_TRACE" "$REPRO_JSON" \
  "$AN/c0_battery.py" "$AN/c0_compare.py" "$AN/c2_attractor.py" "$AN/c6_saturation.py" "$0" > "$OUT/inputs.sha256"
$PY -m pytest -q $AN/tests/test_c0_battery.py $AN/tests/test_c0_compare.py $AN/tests/test_c2_c6.py \
  > "$OUT/tests.log" 2>&1 || fail "CPU tests failed (tests.log)"
status "tests passed"

c0() {  # c0 <tag> <wm> [extra args]
  local tag=$1 wm=$2; shift 2
  status "c0 $tag $*"
  $PY $AN/c0_battery.py --wm "$wm" --out "$OUT/$tag" "$@" > "$OUT/$tag.log" 2>&1 \
    || { status "FAILED c0 $tag ($tag.log: $(tail -1 "$OUT/$tag.log"))"; return 1; }
  status "done c0 $tag"
}
ac() { local tag=$1 wm=$2; shift 2; c0 "$tag" "$wm" "${TRACES[@]}" "$@"; }   # batches A and C
c6() {  # c6 <tag> <wm> <run>
  status "c6 $1"
  $PY $AN/c6_saturation.py --wm "$2" --run "$3" "${TRACES[@]}" --out "$OUT/c6_$1" > "$OUT/c6_$1.log" 2>&1 \
    || { status "FAILED c6 $1 (c6_$1.log)"; return 1; }
  status "done c6 $1"
}
compare() {  # compare <tag> <args...>
  local tag=$1; shift
  status "compare $tag"
  $PY $AN/c0_compare.py "$@" --out "$OUT/$tag" > "$OUT/$tag.log" 2>&1 || fail "compare $tag ($tag.log)"
}

# ---- 1. C0 gate: ETH on the three C0 sources with the 27 Oct reproduction
c0 c0_eth "$ETH" "${C0TRACES[@]}" --repro-trace "$REPRO_TRACE" --repro-json "$REPRO_JSON" \
  || fail "C0 stopped: no cause run started (c0_eth.log)"

# ---- 2. c0 runs on batches A and C (four lanes)
( LANE=L1; ac control "$CTRL" --head-subsets 2,3,4 && ac seed2 "$W1/c0_seed2/wm.pt" && ac seed3 "$W1/c0_seed3/wm.pt" \
    && ac c4_n24 "$W0/c4_n24/wm.pt" && ac control_cold "$CTRL" --imag-context 1 && ac control_held "$CTRL" --actions held ) & P1=$!
( LANE=L2; ac c8_k20 "$W0/c8_k20/wm.pt" --head-subsets 2,5,10 && ac c1c "$C1C" \
    && ac control_sampled "$CTRL" --feedback sampled && ac control_random_step "$CTRL" --feedback random_step \
    && ac control_mean "$CTRL" --feedback mean && ac control_most_tilted "$CTRL" --feedback most_tilted ) & P2=$!
( LANE=L3; ac eth "$ETH" && ac eth_cold "$ETH" --imag-context 1 && ac eth_sampled "$ETH" --feedback sampled \
    && ac eth_held "$ETH" --actions held && ac nll "$W1/nll/wm.pt" && ac widegrav "$W1/c6_widegrav/wm.pt" ) & P3=$!
( LANE=L4; ac dose000 "$W1/c3_dose000/wm.pt" && ac dose025 "$W1/c3_dose025/wm.pt" \
    && ac dose050 "$W1/c3_dose050/wm.pt" && ac dose100 "$W1/c3_dose100/wm.pt" ) & P4=$!
RC=0; for P in $P1 $P2 $P3 $P4; do wait $P || RC=1; done
[ $RC -eq 0 ] || fail "a lane failed (queue.log)"

# ---- 3. C6 saturation (three models) and C2 attraction (four runs)
( LANE=L1; c6 control "$CTRL" "$OUT/control" ) & P1=$!
( LANE=L2; c6 eth "$ETH" "$OUT/eth" ) & P2=$!
( LANE=L3; c6 widegrav "$W1/c6_widegrav/wm.pt" "$OUT/widegrav" ) & P3=$!
( LANE=L4; status "c2"; $PY $AN/c2_attractor.py --run "control=$OUT/control" --run "control_held=$OUT/control_held" \
    --run "eth=$OUT/eth" --run "eth_held=$OUT/eth_held" "${REFARGS[@]}" --out "$OUT/c2" > "$OUT/c2.log" 2>&1 \
    || { status "FAILED c2 (c2.log)"; exit 1; }; status "done c2" ) & P4=$!
RC=0; for P in $P1 $P2 $P3 $P4; do wait $P || RC=1; done
[ $RC -eq 0 ] || fail "C6 or C2 failed (queue.log)"

# ---- 4. tables
compare table_control --control "control=$OUT/control" --noise "seed2=$OUT/seed2" --noise "seed3=$OUT/seed3" \
  --treat "C4_n24=$OUT/c4_n24" --treat "C8_k20=$OUT/c8_k20" --treat "C1c_indep=$OUT/c1c" \
  --treat "C5_nll=$OUT/nll" --treat "C6_widegrav=$OUT/widegrav" \
  --treat "C3_dose000=$OUT/dose000" --treat "C3_dose025=$OUT/dose025" --treat "C3_dose050=$OUT/dose050" \
  --treat "C3_dose100=$OUT/dose100" --treat "C7_cold=$OUT/control_cold"
compare table_eth --control "eth=$OUT/eth" --treat "C3-1_e1v_refit=$OUT/control" --treat "C7_cold_eth=$OUT/eth_cold"
compare table_dose --control "dose000=$OUT/dose000" --treat "dose025=$OUT/dose025" --treat "dose050=$OUT/dose050" \
  --treat "dose100=$OUT/dose100"
compare table_feedback --control "control_head=$OUT/control" --treat "C5_sampled=$OUT/control_sampled" \
  --treat "C9_random_step=$OUT/control_random_step" --treat "C9_mean=$OUT/control_mean" \
  --treat "C9_most_tilted=$OUT/control_most_tilted"
compare table_feedback_eth --control "eth_head=$OUT/eth" --treat "C5_sampled_eth=$OUT/eth_sampled"
status "done"

# ---- 5. summary to stdout (full reports in each folder)
echo "=== C0 (ETH, three sources)"; awk '/^== /{n++} n<=1' "$OUT/c0_eth/c0_report.txt" | head -40
for T in table_control table_eth table_dose table_feedback table_feedback_eth; do echo "=== $T"; cat "$OUT/$T/compare.txt"; done
for R in control eth c8_k20 c1c control_sampled eth_sampled control_held eth_held; do
  echo "--- $R (pooled A+C):"; awk '/^== pooled/{p=1;next} /^== /{p=0} p && /HR\(0\)|C0 rule|C1a|C5|C8|reach|coverage/' "$OUT/$R/c0_report.txt"
done
cat "$OUT/c2/c2_report.txt" "$OUT/c6_control/c6_report.txt" "$OUT/c6_eth/c6_report.txt" "$OUT/c6_widegrav/c6_report.txt"
