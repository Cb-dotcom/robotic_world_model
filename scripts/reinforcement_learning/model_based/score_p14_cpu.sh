#!/usr/bin/env bash
# Score the six already-fitted P1.4 models. Never trains or uses the GPU.
set -euo pipefail
cd "${1:-.}"
PYTHON=${2:-/isaac-sim/python.sh}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
LOTTERY=logs/wm_fit/p1_seed_lottery_2026-10-13
TRACE=logs/wm_fit/go2_plus_fail_ens5/exploit_trace/2026-06-21_14-30-38_9_policy499_trace.csv
test -s "$TRACE"
for SEED in 1 2 3 4 5 6; do
  test -s "$LOTTERY/own_seed$SEED/model_2000.pt"
done
OUT=$(mktemp -d logs/p14_cpu.XXXXXX)
echo "P1.4 CPU outputs: $OUT"
for SEED in 1 2 3 4 5 6; do
  nice -n 10 "$PYTHON" scripts/reinforcement_learning/model_based/analysis/eval_signals.py \
    --trace "$TRACE" --steps_per_env 1000 \
    --wm "$LOTTERY/own_seed$SEED/model_2000.pt" \
    --device cpu --out "$OUT/own_seed$SEED" > "$OUT/seed$SEED.log" 2>&1
done
"$PYTHON" - "$OUT" <<'PY'
import csv, json, math, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
rows = []
for seed in range(1, 7):
    d = json.loads((root / f"own_seed{seed}/summary.json").read_text())
    assert d["device_used"] == "cpu", d["device_used"]
    value = d["metrics"]["epi"]["pre5"]["clean"]["auc"]
    assert value is not None and math.isfinite(value), f"Invalid AUC for seed {seed}"
    rows.append({"seed": seed, "auc_pre5_clean": value, "blind_le_0_75": value <= .75})
    print(f"seed {seed}: epi pre5, clean-negative AUC = {value:.6f}")
with (root / "six_seed_scores.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
values = [r["auc_pre5_clean"] for r in rows]
print(f"Six new seeds: median={statistics.median(values):.6f}, "
      f"range={min(values):.6f}..{max(values):.6f}, blind={sum(v <= .75 for v in values)}/6")
print("P1.4 still requires the two existing own-normalizer models for its locked all-eight summary.")
PY
echo "P1.4 six-seed CPU scoring COMPLETE: $OUT/six_seed_scores.csv"
