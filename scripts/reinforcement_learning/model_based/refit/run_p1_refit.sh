#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# P1 controlled re-fit. Three world models that differ ONLY in the training data:
#   R1_curated        curated go2_noise segments (seam labels removed, num_envs=1 per segment)
#   R2_plusfail       full +fail training set (segments_plus_fail_train_flat, 1.288M rows)
#   R3_plusfail_115k  +fail subsampled (whole env blocks, stratified per file, seed 0) to the
#                     curated row count
# Same for all three: boundary-aware fitter (fit_world_model_go2_pilot_segments.py), ONE shared
# normalizer (union of curated segments + full +fail, population std + 1e-6), June settings
# (2000 it x 20 minibatches x 2048, lr 1e-3, ensemble 5, termination_pos_weight 0 = auto,
# termination_loss_weight 1.0), --seed 0, --lazy_windows (identical batches, ~12 GB less GPU
# memory for R2).
#
# Run inside the container, from scripts/reinforcement_learning/model_based:
#     bash refit/run_p1_refit.sh            # build data, GPU preflight, launch R1-R3 (background)
#     bash refit/run_p1_refit.sh status     # progress / completion of the three fits
#
# Overrides (env vars): PY, ROOT, TAG, OUT, DEVICE (cpu skips GPU checks), ITERATIONS,
# NUM_MINI_BATCHES, MINI_BATCH_SIZE, LR, POS_WEIGHT, TERM_LOSS_WEIGHT, SEED, LAZY (1),
# SEQUENTIAL=1 (one fit after the other), MIN_FREE_MB_PARALLEL (12288), PROBE (1 = measure the
# GPU peak of one fit with a 2-iteration probe on the R2 data before launching), FORCE=1 (move an
# existing $OUT aside), RSL=<rsl_rl_rwm path> (exported as PYTHONPATH, as in the June fit docs),
# WAIT=1 (block until the fits finish; used by the tests), LOG_INTERVAL, SAVE_INTERVAL.
set -euo pipefail

PY=${PY:-/isaac-sim/python.sh}
ROOT=${ROOT:-../../..}                       # robotic_world_model repo root
TAG=${TAG:-p1_refit_2026-10-09}
DEVICE=${DEVICE:-cuda:0}
ITERATIONS=${ITERATIONS:-2000}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-20}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-2048}
LR=${LR:-1e-3}
POS_WEIGHT=${POS_WEIGHT:-0.0}
TERM_LOSS_WEIGHT=${TERM_LOSS_WEIGHT:-1.0}
SEED=${SEED:-0}
LAZY=${LAZY:-1}
SEQUENTIAL=${SEQUENTIAL:-0}
MIN_FREE_MB_PARALLEL=${MIN_FREE_MB_PARALLEL:-12288}
PROBE=${PROBE:-1}
FORCE=${FORCE:-0}
RSL=${RSL:-}
WAIT=${WAIT:-0}
LOG_INTERVAL=${LOG_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}

[[ -f fit_world_model_go2_pilot_segments.py && -d refit ]] || { echo "run from scripts/reinforcement_learning/model_based"; exit 1; }
MB_DIR=$(pwd -P)
ROOT=$(cd "$ROOT" && pwd -P)
OUT=${OUT:-$ROOT/logs/wm_fit/$TAG}

CURATED_DIR=$ROOT/assets/data/go2_noise
SEGS=(seg_n00 seg_n02 seg_n04 seg_n08 seg_n10 seg_n12)
CURATED_CONCAT=$CURATED_DIR/state_action_data_0.csv
PLUSFAIL=$ROOT/assets/data/go2_pilot_1m/segments_plus_fail_train_flat
DATA_OUT=$OUT/data
CUR_SEG_DIR=$DATA_OUT/curated_segments
SUB_DIR=$DATA_OUT/plus_fail_115k
NORM=$DATA_OUT/shared_norm.npz
RUNS=(R1_curated R2_plusfail R3_plusfail_115k)
declare -A RUN_DATA=( [R1_curated]="$CUR_SEG_DIR" [R2_plusfail]="$PLUSFAIL" [R3_plusfail_115k]="$SUB_DIR" )
CKPT_NAME=model_${ITERATIONS}.pt

if [[ -n "$RSL" ]]; then export PYTHONPATH="$RSL"; fi
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ------------------------------------------------------------------ status subcommand
if [[ "${1:-run}" == status ]]; then
  [[ -d "$OUT" ]] || { echo "no $OUT"; exit 1; }
  [[ -f "$OUT/STATUS.txt" ]] && head -n 20 "$OUT/STATUS.txt"
  for r in "${RUNS[@]}"; do
    d=$OUT/$r
    if [[ -f "$d/EXIT_CODE" ]]; then st="finished exit=$(cat "$d/EXIT_CODE")"
    elif [[ -f "$d/stdout.log" ]]; then st="running or killed (no EXIT_CODE yet)"
    else st="not started"; fi
    last=$(grep -E '^\[fit\] it ' "$d/stdout.log" 2>/dev/null | tail -n 1 || true)
    done_line=$(grep -E '^\[fit\] DONE' "$d/stdout.log" 2>/dev/null | tail -n 1 || true)
    echo "$r: $st | ${last:-no iteration logged yet} ${done_line:+| $done_line}"
  done
  exit 0
fi

# ------------------------------------------------------------------ 0. preflight
echo "[refit] $(ts) preflight; OUT=$OUT"
[[ -x "$PY" || "$PY" != /* ]] || { echo "python not found: $PY"; exit 1; }
for s in "${SEGS[@]}"; do [[ -f "$CURATED_DIR/$s.csv" ]] || { echo "missing $CURATED_DIR/$s.csv"; exit 1; }; done
[[ -f "$CURATED_CONCAT" ]] || { echo "missing $CURATED_CONCAT"; exit 1; }
[[ -f "$PLUSFAIL/manifest.csv" ]] || { echo "missing $PLUSFAIL/manifest.csv"; exit 1; }
if [[ -e "$OUT" ]]; then
  if [[ "$FORCE" == 1 ]]; then
    bak="$OUT.bak.$(date -u +%Y%m%dT%H%M%SZ)"; mv "$OUT" "$bak"; echo "[refit] moved existing $OUT -> $bak"
  else
    echo "[refit] $OUT exists; refusing to overwrite (FORCE=1 moves it aside)"; exit 1
  fi
fi
if pgrep -f fit_world_model_go2_pilot_segments.py > /dev/null 2>&1; then
  echo "[refit] WARNING: another fit_world_model_go2_pilot_segments.py is running:"; pgrep -af fit_world_model_go2_pilot_segments.py || true
fi
mkdir -p "$OUT" "$DATA_OUT"
{
  echo "date: $(ts)"; echo "host: $(hostname)"; echo "pwd: $MB_DIR"; echo "PY: $PY"; echo "PYTHONPATH: ${PYTHONPATH:-}"
  echo "git: $(git -C "$MB_DIR" rev-parse HEAD 2>/dev/null || echo n/a) branch $(git -C "$MB_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
  echo "git status (tracked):"; git -C "$MB_DIR" status --porcelain --untracked-files=no 2>/dev/null || true
  echo "settings: DEVICE=$DEVICE ITERATIONS=$ITERATIONS NUM_MINI_BATCHES=$NUM_MINI_BATCHES MINI_BATCH_SIZE=$MINI_BATCH_SIZE LR=$LR POS_WEIGHT=$POS_WEIGHT TERM_LOSS_WEIGHT=$TERM_LOSS_WEIGHT SEED=$SEED LAZY=$LAZY SEQUENTIAL=$SEQUENTIAL"
} > "$OUT/preflight.txt"
"$PY" -c "import sys, torch, numpy, pandas, rsl_rl.modules as m; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__); print('rsl_rl', m.__file__)" 2>&1 | grep -v -i warn | tee -a "$OUT/preflight.txt"
"$PY" -c "import rsl_rl.modules" > /dev/null 2>&1 || { echo "[refit] cannot import rsl_rl with $PY (set RSL=<rsl_rl_rwm path>)"; exit 1; }

# ------------------------------------------------------------------ 1. curated segments (seams removed)
echo "[refit] $(ts) 1/3 curated segments -> $CUR_SEG_DIR"
seg_paths=(); for s in "${SEGS[@]}"; do seg_paths+=("$CURATED_DIR/$s.csv"); done
"$PY" refit/make_curated_segments.py --segs "${seg_paths[@]}" --concat "$CURATED_CONCAT" --out "$CUR_SEG_DIR" \
  2>&1 | tee "$DATA_OUT/make_curated_segments.log"
CUR_ROWS=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['totals']['rows'])" "$CUR_SEG_DIR/cards.json" 2>/dev/null | tail -n 1)
[[ "$CUR_ROWS" =~ ^[0-9]+$ ]] || { echo "[refit] could not read curated row count"; exit 1; }

# ------------------------------------------------------------------ 2. +fail subsample to the curated row count
echo "[refit] $(ts) 2/3 +fail subsample (target $CUR_ROWS rows) -> $SUB_DIR"
"$PY" refit/make_subsample.py --src "$PLUSFAIL" --out "$SUB_DIR" --target_rows "$CUR_ROWS" --seed 0 \
  2>&1 | tee "$DATA_OUT/make_subsample.log"
[[ -f "$SUB_DIR/manifest.csv" ]] || { echo "[refit] subsample failed"; exit 1; }

# ------------------------------------------------------------------ 3. shared normalizer (curated segments + full +fail)
echo "[refit] $(ts) 3/3 shared normalizer -> $NORM"
"$PY" refit/make_shared_norm.py --inputs "$CUR_SEG_DIR" "$PLUSFAIL" --out "$NORM" 2>&1 | tee "$DATA_OUT/make_shared_norm.log"
[[ -f "$NORM" ]] || { echo "[refit] normalizer failed"; exit 1; }
# expected counts (docs / workstation check of 8 Oct): warnings only, read them before the fits finish
"$PY" - "$CUR_SEG_DIR/cards.json" "$SUB_DIR/cards.json" "$NORM.json" "$PLUSFAIL" <<'EOF' 2>&1 | tee "$DATA_OUT/expectations.txt"
import json, sys
cur, sub, norm = (json.load(open(p)) for p in sys.argv[1:4])
plus = norm["by_input"].get(sys.argv[4], {})
ct, st = cur["totals"], sub["totals"]
checks = [
    ("curated rows", ct["rows"], 115000),
    ("curated real falls after seam removal", ct["real_terms"], 171),
    ("curated interior terminations with a NON-zero action (reset-row hypothesis)", ct["interior_terms_nonzero_action"], 0),
    ("curated concat check ok", cur.get("concat_check", {}).get("ok"), True),
    ("+fail rows", plus.get("rows"), 1288000),
    ("+fail terminations", plus.get("terms"), 302),
    ("+fail terminations with an all-zero action (reset-row hypothesis)", plus.get("terms_zero_action"), plus.get("terms")),
    ("+fail terminations on an env-block last row (seams should be pre-cleaned)", plus.get("terms_on_block_last_row"), 0),
]
for name, got, exp in checks:
    print(f"[expect] {'ok   ' if got == exp else 'CHECK'} {name}: {got} (expected {exp})")
print(f"[expect] info curated seams removed: {ct['seams_removed']} (6 if the seg files carry the forced seam label, 0 if not)")
print(f"[expect] info subsample rows {st['rows']} vs curated {ct['rows']} ({100 * (st['rows'] - ct['rows']) / ct['rows']:+.2f}%), "
      f"subsample real terminations {st['real_terms']} (curated {ct['real_terms']}, full +fail {plus.get('terms')})")
EOF
# informational: reset-row check on the derived data (code review 1.2 Q1.4)
"$PY" analysis/check_reset_rows.py "$CUR_SEG_DIR/*.csv" "$SUB_DIR/*.csv" 2>&1 | grep -v -i warn > "$DATA_OUT/check_reset_rows.txt" || true

# ------------------------------------------------------------------ 4. job scripts
fit_cmd() {  # $1 = data dir, $2 = output checkpoint
  local c=("$PY" fit_world_model_go2_pilot_segments.py --data "$1" --output "$2"
    --iterations "$ITERATIONS" --num_mini_batches "$NUM_MINI_BATCHES" --mini_batch_size "$MINI_BATCH_SIZE"
    --lr "$LR" --weight_decay 0.0 --max_grad_norm 1.0
    --termination_loss_weight "$TERM_LOSS_WEIGHT" --termination_pos_weight "$POS_WEIGHT"
    --ensemble_size 5 --config go2_flat --device "$DEVICE"
    --log_interval "$LOG_INTERVAL" --save_interval "$SAVE_INTERVAL"
    --seed "$SEED" --norm_stats "$NORM")
  [[ "$LAZY" == 1 ]] && c+=(--lazy_windows)
  printf '%q ' "${c[@]}"
}
write_job() {  # $1 = run dir, $2 = command line
  cat > "$1/job.sh" <<EOF
#!/usr/bin/env bash
# generated by refit/run_p1_refit.sh at $(ts)
cd $(printf '%q' "$MB_DIR")
${PYTHONPATH:+export PYTHONPATH=$(printf '%q' "$PYTHONPATH")}
echo "[job] start \$(date -u +%Y-%m-%dT%H:%M:%SZ) host=\$(hostname) pid=\$\$"
$2
rc=\$?
echo "\$rc" > $(printf '%q' "$1/EXIT_CODE")
echo "[job] end \$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=\$rc"
exit \$rc
EOF
}
for r in "${RUNS[@]}"; do
  mkdir -p "$OUT/$r"
  write_job "$OUT/$r" "$(fit_cmd "${RUN_DATA[$r]}" "$OUT/$r/$CKPT_NAME")"
done

# ------------------------------------------------------------------ 5. GPU preflight
# GPU_CHECK=auto: on when DEVICE is cuda*. (GPU_CHECK=1 with DEVICE=cpu is only for the tests,
# with a fake nvidia-smi on PATH; the probe then reports no CUDA peak and counts as 0 MiB.)
FREE_MB=n/a; PEAK_MB=n/a
GPU_CHECK=${GPU_CHECK:-auto}
[[ "$GPU_CHECK" == auto ]] && { [[ "$DEVICE" == cuda* ]] && GPU_CHECK=1 || GPU_CHECK=0; }
if [[ "$GPU_CHECK" == 1 ]]; then
  gpu_idx=${GPU_INDEX:-}
  if [[ -z "$gpu_idx" ]]; then gpu_idx=${DEVICE#cuda}; gpu_idx=${gpu_idx#:}; [[ "$gpu_idx" =~ ^[0-9]+$ ]] || gpu_idx=0; fi
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "[refit] note: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; nvidia-smi index $gpu_idx may not be the same GPU (set GPU_INDEX)"
  command -v nvidia-smi > /dev/null || { echo "[refit] nvidia-smi not found; cannot check free GPU memory"; exit 3; }
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv | tee "$OUT/gpu_before.txt"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$OUT/gpu_before.txt" 2>/dev/null || true
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_idx" | head -n 1 | tr -dc '0-9')
  [[ "$FREE_MB" =~ ^[0-9]+$ ]] || { echo "[refit] could not read free GPU memory"; exit 3; }
  echo "[refit] GPU $gpu_idx free memory: ${FREE_MB} MiB"
  NEED_ONE=0
  if [[ "$PROBE" == 1 ]]; then
    echo "[refit] $(ts) probe: 2 iterations on the R2 data to measure the per-fit GPU peak (~1-3 min)"
    mkdir -p "$OUT/probe"
    probe_cmd=$(ITERATIONS=2 LOG_INTERVAL=1 SAVE_INTERVAL=1000000 fit_cmd "$PLUSFAIL" "$OUT/probe/probe.pt")
    echo "$probe_cmd" > "$OUT/probe/probe_cmd.txt"
    eval "$probe_cmd" > "$OUT/probe/probe.log" 2>&1 || { echo "[refit] PROBE FAILED (see $OUT/probe/probe.log):"; tail -n 20 "$OUT/probe/probe.log"; exit 3; }
    PEAK_MB=$(grep -oE 'peak_reserved_mb=[0-9]+' "$OUT/probe/probe.log" | tail -n 1 | cut -d= -f2 || true)
    if ! [[ "$PEAK_MB" =~ ^[0-9]+$ ]]; then
      [[ "$DEVICE" == cuda* ]] && { echo "[refit] probe did not report peak_reserved_mb (see $OUT/probe/probe.log)"; exit 3; }
      PEAK_MB=0
    fi
    grep -E 's/it|peak_' "$OUT/probe/probe.log" | tail -n 3 || true
    NEED_ONE=$(( PEAK_MB + 1024 ))   # + CUDA context and margin
    echo "[refit] probe peak reserved ${PEAK_MB} MiB -> need ~${NEED_ONE} MiB per fit"
  fi
  if [[ "$SEQUENTIAL" == 1 ]]; then
    if (( FREE_MB < NEED_ONE )); then echo "[refit] REFUSING: free ${FREE_MB} MiB < ${NEED_ONE} MiB needed for one fit"; exit 3; fi
  else
    need=$(( 3 * NEED_ONE )); (( need < MIN_FREE_MB_PARALLEL )) && need=$MIN_FREE_MB_PARALLEL
    if (( FREE_MB < need )); then
      echo "[refit] REFUSING to start all three in parallel: free ${FREE_MB} MiB < ${need} MiB"
      echo "[refit]   (3 x measured per-fit need ${NEED_ONE} MiB, floor MIN_FREE_MB_PARALLEL=${MIN_FREE_MB_PARALLEL})"
      echo "[refit]   data, normalizer and job scripts are ready in $OUT; to run one fit at a time (~3 x 92 min):"
      echo "[refit]   SEQUENTIAL=1 FORCE=1 bash refit/run_p1_refit.sh"
      exit 3
    fi
  fi
fi

# ------------------------------------------------------------------ 6. launch
pids=()
{
  echo "P1 re-fit launched $(ts) on $(hostname)"
  echo "mode: $([[ "$SEQUENTIAL" == 1 ]] && echo sequential || echo parallel)  device: $DEVICE  gpu_free_mib_at_launch: $FREE_MB  probe_peak_reserved_mib: $PEAK_MB"
  echo "normalizer: $NORM"
} > "$OUT/STATUS.txt"
if [[ "$SEQUENTIAL" == 1 ]]; then
  {
    echo "#!/usr/bin/env bash"
    for r in "${RUNS[@]}"; do echo "bash $(printf '%q' "$OUT/$r/job.sh") > $(printf '%q' "$OUT/$r/stdout.log") 2>&1"; done
  } > "$OUT/sequential.sh"
  nohup bash "$OUT/sequential.sh" > "$OUT/sequential.log" 2>&1 < /dev/null &
  pids+=($!)
  echo "sequential runner pid=$! started=$(ts) order=${RUNS[*]}" >> "$OUT/STATUS.txt"
else
  for r in "${RUNS[@]}"; do
    nohup bash "$OUT/$r/job.sh" > "$OUT/$r/stdout.log" 2>&1 < /dev/null &
    pids+=($!)
    echo "$r pid=$! started=$(ts) data=${RUN_DATA[$r]}" >> "$OUT/STATUS.txt"
  done
fi
for r in "${RUNS[@]}"; do echo "$r command: $(sed -n '/fit_world_model_go2_pilot_segments.py/p' "$OUT/$r/job.sh")" >> "$OUT/STATUS.txt"; done
cat "$OUT/STATUS.txt"

cat <<EOF

[refit] launched. Expected: ~2.8 s/it alone -> ~92 min per fit; in parallel on a shared GPU each fit slows
        down (estimate 1.5-3x), sequential = ~4.6 h. Checkpoints: $OUT/<run>/$CKPT_NAME (also written at
        every $SAVE_INTERVAL iterations, so the file existing does NOT mean the fit finished).
Monitor:
  tail -f $OUT/R1_curated/stdout.log $OUT/R2_plusfail/stdout.log $OUT/R3_plusfail_115k/stdout.log
  watch -n 60 nvidia-smi
  bash refit/run_p1_refit.sh status          # (same TAG/OUT env vars)
Completion: each run writes EXIT_CODE (0 = success) and its log ends with "[fit] DONE"; then
  bash refit/score_p1_refit.sh
EOF

if [[ "$WAIT" == 1 ]]; then
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  for r in "${RUNS[@]}"; do [[ "$(cat "$OUT/$r/EXIT_CODE" 2>/dev/null)" == 0 ]] || rc=1; done
  echo "[refit] all fits finished, rc=$rc"
  exit $rc
fi
