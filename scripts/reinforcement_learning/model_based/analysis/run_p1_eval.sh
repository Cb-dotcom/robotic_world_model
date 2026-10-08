#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# P1 evaluation: every signal (epi / alea / term / kNN) on identical index sets,
# 3 WMs x 2 exploit traces, each WM paired with its own training set for kNN.
#
# Run inside the container, from scripts/reinforcement_learning/model_based:
#     bash analysis/run_p1_eval.sh
# Overrides (env vars): PY, ROOT, OUT, DEVICE, BOOT, RUN_SCORER=0 (skip the parity re-run
# of the original scorer), ONLY="curated_t9 plusfail_t9" (subset of runs).
set -euo pipefail

PY=${PY:-/isaac-sim/python.sh}
ROOT=${ROOT:-../../..}                       # robotic_world_model repo root
OUT=${OUT:-$ROOT/logs/p1_eval_2026-10-09}
DEVICE=${DEVICE:-cuda:0}
BOOT=${BOOT:-2000}
RUN_SCORER=${RUN_SCORER:-1}
ONLY=${ONLY:-}

declare -A WM_CKPT=(
  [curated]="$ROOT/logs/wm_fit/go2_curated_wm.pt"
  [plusfail]="$ROOT/logs/wm_fit/go2_plus_fail_ens5/model_2000.pt"
  [truemixed]="$ROOT/logs/wm_fit/go2_true_mixed_1m_stage_rollout_ens5_matched_fileout/model_2000.pt"
)
# kNN training set of each WM (globs are expanded by python; manifest.csv is skipped there)
declare -A WM_KNN=(
  [curated]="$ROOT/assets/data/go2_noise/state_action_data_0.csv"
  [plusfail]="$ROOT/assets/data/go2_pilot_1m/segments_plus_fail_train_flat/*.csv"
  [truemixed]="$ROOT/assets/data/go2_true_mixed_1m_stage_rollout/*.csv"
)
declare -A TRACE=(
  [t9]="$ROOT/logs/wm_fit/go2_plus_fail_ens5/exploit_trace/2026-06-21_14-30-38_9_policy499_trace.csv"
  [t6]="$ROOT/logs/wm_fit/go2_plus_fail_ens5/exploit_trace/2026-06-21_14-22-41_6_policy499_trace_2000.csv"
)
declare -A STEPS=( [t9]=1000 [t6]=2000 )
WMS=(curated plusfail truemixed)
TRACES=(t9 t6)

want() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }

# ---------------------------------------------------------------- 0. preflight
echo "[p1] preflight"
[[ -x "$PY" || "$PY" != /* ]] || { echo "python not found: $PY"; exit 1; }
[[ -f score_go2_exploit_trace_uncertainty.py && -f analysis/eval_signals.py ]] || {
  echo "run from scripts/reinforcement_learning/model_based"; exit 1; }
for w in "${WMS[@]}"; do
  [[ -f "${WM_CKPT[$w]}" ]] || { echo "missing WM ${WM_CKPT[$w]}"; exit 1; }
  compgen -G "${WM_KNN[$w]}" > /dev/null || { echo "no kNN files for $w: ${WM_KNN[$w]}"; exit 1; }
done
for t in "${TRACES[@]}"; do [[ -f "${TRACE[$t]}" ]] || { echo "missing trace ${TRACE[$t]}"; exit 1; }; done
mkdir -p "$OUT"
git -C "$ROOT" rev-parse HEAD > "$OUT/git_commit.txt" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv > "$OUT/gpu.txt" 2>/dev/null || true

# ---------------------------------------------------------------- 1. main runs (3 WMs x 2 traces)
for w in "${WMS[@]}"; do
  for t in "${TRACES[@]}"; do
    name="${w}_${t}"; want "$name" || continue
    d="$OUT/$name"; mkdir -p "$d"
    echo "[p1] ===== $name ====="
    "$PY" analysis/eval_signals.py \
      --trace "${TRACE[$t]}" --num_envs 64 --steps_per_env "${STEPS[$t]}" \
      --wm "${WM_CKPT[$w]}" \
      --knn_train "${WM_KNN[$w]}" --knn_name "${w}_train" \
      --knn_pair_offset 1 --knn_features state_action --knn_norm own --knn_k 1,5,10,50 --knn_primary_k 10 \
      --num_neg 5000 --clean_gap 25 --lead_ks 0,1,2,3,5,10,15,20,25,50 \
      --bootstrap "$BOOT" --seed 0 --device "$DEVICE" --out "$d" 2>&1 | tee "$d/eval_stdout.txt"
  done
done

# ---------------------------------------------------------------- 2. parity vs the original scorer
# Re-runs the unchanged scorer and diffs its result lines with scorer_parity.txt written above.
# Expected headline (trace _9): curated 0.969, plusfail 0.649 for "ROC-AUC epi(pre5 > walking)".
filt() { grep -E '^(\[score\]|\\n===|fall_transition|walking|pre1_failure|pre5_failure|ratio mean|ROC-AUC)' "$1" || true; }
if [[ "$RUN_SCORER" == 1 ]]; then
  parity_fail=0
  for w in "${WMS[@]}"; do
    for t in "${TRACES[@]}"; do
      name="${w}_${t}"; want "$name" || continue
      d="$OUT/$name"
      echo "[p1] scorer parity: $name"
      "$PY" score_go2_exploit_trace_uncertainty.py --wm "${WM_CKPT[$w]}" --trace "${TRACE[$t]}" \
        --num_envs 64 --steps_per_env "${STEPS[$t]}" --num_neg 5000 --device "$DEVICE" \
        > "$d/scorer_stdout.txt" 2> "$d/scorer_stderr.txt"
      if diff <(filt "$d/scorer_stdout.txt") <(filt "$d/scorer_parity.txt") > "$d/parity_diff.txt"; then
        echo "[p1]   PARITY OK  ($(grep 'epi(pre5' "$d/scorer_parity.txt" || echo 'no pre5 line'))"
      else
        echo "[p1]   PARITY MISMATCH -> $d/parity_diff.txt"; cat "$d/parity_diff.txt"; parity_fail=1
      fi
    done
  done
  [[ $parity_fail == 0 ]] && echo "[p1] all parity checks passed" || echo "[p1] WARNING: parity mismatches above"
fi
echo "[p1] headline (expect curated_t9 0.969, plusfail_t9 0.649):"
for d in "$OUT"/*_t9; do [[ -f "$d/scorer_parity.txt" ]] && echo "  $(basename "$d"): $(grep 'epi(pre5' "$d/scorer_parity.txt")"; done

# ---------------------------------------------------------------- 3. comparison table
"$PY" analysis/summarize_p1_eval.py "$OUT" --signals epi,term,knn10 --sets pre5,fall

# ---------------------------------------------------------------- 4. optional robustness block (if time allows)
# kNN variants on trace _9 only: pair offset 0, state-only features, WM normalisation.
# Uncomment to run; results go to $OUT/robust/ and get their own table.
#
# R="$OUT/robust"; mkdir -p "$R"
# for w in "${WMS[@]}"; do
#   t=t9
#   common=(--trace "${TRACE[$t]}" --num_envs 64 --steps_per_env "${STEPS[$t]}"
#           --knn_train "${WM_KNN[$w]}" --knn_name "${w}_train" --knn_k 1,5,10,50 --knn_primary_k 10
#           --num_neg 5000 --clean_gap 25 --bootstrap "$BOOT" --seed 0 --device "$DEVICE")
#   "$PY" analysis/eval_signals.py "${common[@]}" --knn_pair_offset 0 \
#       --out "$R/${w}_${t}_offset0" 2>&1 | tee "$R/${w}_${t}_offset0.log"
#   "$PY" analysis/eval_signals.py "${common[@]}" --knn_features state \
#       --out "$R/${w}_${t}_stateonly" 2>&1 | tee "$R/${w}_${t}_stateonly.log"
#   "$PY" analysis/eval_signals.py "${common[@]}" --wm "${WM_CKPT[$w]}" --knn_norm wm \
#       --out "$R/${w}_${t}_wmnorm" 2>&1 | tee "$R/${w}_${t}_wmnorm.log"
# done
# "$PY" analysis/summarize_p1_eval.py "$R" --signals knn1,knn10,knn50 --sets pre5,fall

echo "[p1] done -> $OUT"
