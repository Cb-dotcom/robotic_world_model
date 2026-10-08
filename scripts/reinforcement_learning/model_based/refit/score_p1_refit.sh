#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Score the P1 re-fit WMs (R1_curated, R2_plusfail, R3_plusfail_115k) with analysis/eval_signals.py
# on exploit traces _9 (1000 steps) and _6 (2000 steps), same flags as analysis/run_p1_eval.sh.
# kNN training set of each WM = its own training data directory (CSV glob; manifest.csv is
# skipped by eval_signals because it does not have 66 columns).
#
# Run inside the container, from scripts/reinforcement_learning/model_based, after
# `bash refit/run_p1_refit.sh status` shows all three fits finished with exit=0:
#     bash refit/score_p1_refit.sh
# Overrides (env vars): PY, ROOT, TAG, FIT_OUT, OUT, DEVICE, ITERATIONS (selects model_<it>.pt),
# BOOT, NUM_NEG, NUM_ENVS, TRACE_T9, TRACE_T6, STEPS_T9, STEPS_T6, KNN_K, KNN_PRIMARY_K, LEAD_KS,
# ONLY="R1_curated_t9 R2_plusfail_t9" (subset), ALLOW_INCOMPLETE=1 (score unfinished fits), RSL.
set -euo pipefail

PY=${PY:-/isaac-sim/python.sh}
ROOT=${ROOT:-../../..}
TAG=${TAG:-p1_refit_2026-10-09}
DEVICE=${DEVICE:-cuda:0}
ITERATIONS=${ITERATIONS:-2000}
BOOT=${BOOT:-2000}
NUM_NEG=${NUM_NEG:-5000}
NUM_ENVS=${NUM_ENVS:-64}
STEPS_T9=${STEPS_T9:-1000}
STEPS_T6=${STEPS_T6:-2000}
KNN_K=${KNN_K:-1,5,10,50}
KNN_PRIMARY_K=${KNN_PRIMARY_K:-10}
LEAD_KS=${LEAD_KS:-0,1,2,3,5,10,15,20,25,50}
ONLY=${ONLY:-}
ALLOW_INCOMPLETE=${ALLOW_INCOMPLETE:-0}
RSL=${RSL:-}
[[ -n "$RSL" ]] && export PYTHONPATH="$RSL"

[[ -f analysis/eval_signals.py && -d refit ]] || { echo "run from scripts/reinforcement_learning/model_based"; exit 1; }
ROOT=$(cd "$ROOT" && pwd -P)
FIT_OUT=${FIT_OUT:-$ROOT/logs/wm_fit/$TAG}
OUT=${OUT:-$ROOT/logs/p1_eval_refit}
TRACE_T9=${TRACE_T9:-$ROOT/logs/wm_fit/go2_plus_fail_ens5/exploit_trace/2026-06-21_14-30-38_9_policy499_trace.csv}
TRACE_T6=${TRACE_T6:-$ROOT/logs/wm_fit/go2_plus_fail_ens5/exploit_trace/2026-06-21_14-22-41_6_policy499_trace_2000.csv}
PLUSFAIL=$ROOT/assets/data/go2_pilot_1m/segments_plus_fail_train_flat

RUNS=(R1_curated R2_plusfail R3_plusfail_115k)
declare -A KNN=(
  [R1_curated]="$FIT_OUT/data/curated_segments/*.csv"
  [R2_plusfail]="$PLUSFAIL/*.csv"
  [R3_plusfail_115k]="$FIT_OUT/data/plus_fail_115k/*.csv"
)
declare -A TRACE=( [t9]="$TRACE_T9" [t6]="$TRACE_T6" )
declare -A STEPS=( [t9]="$STEPS_T9" [t6]="$STEPS_T6" )
TRACES=(t9 t6)
want() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }

# ---------------------------------------------------------------- 0. preflight: finished fits, same normalizer/seed
echo "[score] preflight; FIT_OUT=$FIT_OUT OUT=$OUT"
for t in "${TRACES[@]}"; do [[ -f "${TRACE[$t]}" ]] || { echo "missing trace ${TRACE[$t]}"; exit 1; }; done
ckpts=()
for r in "${RUNS[@]}"; do
  ck="$FIT_OUT/$r/model_${ITERATIONS}.pt"; ckpts+=("$ck")
  [[ -f "$ck" ]] || { echo "missing checkpoint $ck"; exit 1; }
  ec=$(cat "$FIT_OUT/$r/EXIT_CODE" 2>/dev/null || echo none)
  if [[ "$ec" != 0 && "$ALLOW_INCOMPLETE" != 1 ]]; then
    echo "[score] $r not finished successfully (EXIT_CODE=$ec); see $FIT_OUT/$r/stdout.log (ALLOW_INCOMPLETE=1 to score anyway)"; exit 1
  fi
  compgen -G "${KNN[$r]}" > /dev/null || { echo "no kNN training files for $r: ${KNN[$r]}"; exit 1; }
done
mkdir -p "$OUT"
"$PY" - "$ITERATIONS" "$ALLOW_INCOMPLETE" "${ckpts[@]}" <<'EOF' 2>&1 | grep -v -i warn | tee "$OUT/checkpoints.txt"
import sys, torch
it, allow, paths = int(sys.argv[1]), sys.argv[2] == "1", sys.argv[3:]
rows, bad = [], False
for p in paths:
    c = torch.load(p, map_location="cpu", weights_only=False)
    pv = c.get("provenance", {})
    rows.append((p, c.get("iter"), c.get("norm_stats_sha256"), pv.get("seed"), c.get("total_rows"), c.get("total_real_terms"),
                 c.get("num_windows"), pv.get("termination_pos_weight_used")))
    print(f"{p}: iter={c.get('iter')} rows={c.get('total_rows')} real_terms={c.get('total_real_terms')} windows={c.get('num_windows')} "
          f"pos_weight={pv.get('termination_pos_weight_used')} seed={pv.get('seed')} norm_sha256={c.get('norm_stats_sha256')}")
    if c.get("iter") != it:
        print(f"  NOT FINISHED: iter {c.get('iter')} != {it}"); bad = True
if len({r[2] for r in rows}) != 1 or rows[0][2] is None:
    print("  normalizer differs between runs (or missing)"); bad = True
if len({r[3] for r in rows}) != 1:
    print("  seed differs between runs"); bad = True
print("checkpoint check:", "FAILED" if bad else "OK (same normalizer and seed, all at the expected iteration)")
sys.exit(1 if bad and not allow else 0)
EOF
git -C "$ROOT" rev-parse HEAD > "$OUT/git_commit.txt" 2>/dev/null || true

# ---------------------------------------------------------------- 1. eval_signals (3 WMs x 2 traces)
for r in "${RUNS[@]}"; do
  for t in "${TRACES[@]}"; do
    name="${r}_${t}"; want "$name" || continue
    d="$OUT/$name"; mkdir -p "$d"
    echo "[score] ===== $name ====="
    "$PY" analysis/eval_signals.py \
      --trace "${TRACE[$t]}" --num_envs "$NUM_ENVS" --steps_per_env "${STEPS[$t]}" \
      --wm "$FIT_OUT/$r/model_${ITERATIONS}.pt" \
      --knn_train "${KNN[$r]}" --knn_name "${r}_train" \
      --knn_pair_offset 1 --knn_features state_action --knn_norm own --knn_k "$KNN_K" --knn_primary_k "$KNN_PRIMARY_K" \
      --num_neg "$NUM_NEG" --clean_gap 25 --lead_ks "$LEAD_KS" \
      --bootstrap "$BOOT" --seed 0 --device "$DEVICE" --out "$d" 2>&1 | tee "$d/eval_stdout.txt"
  done
done

# ---------------------------------------------------------------- 2. comparison table
"$PY" analysis/summarize_p1_eval.py "$OUT" --signals "epi,term_logit,knn${KNN_PRIMARY_K}" --sets pre5,pre1,lead10,fall
echo "[score] done -> $OUT (comparison.txt, comparison.csv)"
