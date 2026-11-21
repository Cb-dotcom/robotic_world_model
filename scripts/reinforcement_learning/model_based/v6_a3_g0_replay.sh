#!/bin/bash
# A3 follow-up (21 Nov 2026, review decision): read-only replay of the G0 world model on held-out s2 data.
# CPU only, no training, no GPU. G0 stays a valid FAIL (A3 §4); this looks for the cause of the early stops.
# Writes only a new <g0_out>/replay/ folder. Analysis: analysis/go2_g0_replay.py (reading rule in its docstring).
# Usage, from the robotic_world_model root in the container:
#   scripts/reinforcement_learning/model_based/v6_a3_g0_replay.sh logs/p2/a3_g0_1121
set -uo pipefail
OUT=${1:?the G0 out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
ROUND0=${ROUND0:-logs/p2/go2_round0_1118/round0_s2_seed101.npz}
G0=${G0:-logs/offline/go2_flat/2026-11-21_14-03-25_3001/policy_2499.pt}
WM=$OUT/wm/g0_wm_seed1.pt
WM_SHA=${WM_SHA:-9a08ffcbe71b70d19203217b6ea4c856ec05ac70f37e9c1ba951f3ec89c655b2}   # run-log 21 Nov; override only in dry runs
R=$OUT/replay
[ -e "$R" ] && { echo "exists: $R"; exit 2; }
for f in "$ROUND0" "$WM" "$OUT/data/go2_norm.npz" "$OUT/data/withheld.json" "$G0"; do
  [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
[ "$(sha256sum "$WM" | cut -d' ' -f1)" = "$WM_SHA" ] || { echo "model hash differs from the run-log (21 Nov)"; exit 1; }
mkdir -p "$R"
{ git rev-parse HEAD; git status --short | grep -v '^??'; } > "$R/git.txt"
export PYTHONUNBUFFERED=1 PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_g0_replay.py > "$R/tests.log" 2>&1 \
  || { echo "tests failed ($R/tests.log)"; tail -3 "$R/tests.log"; exit 1; }
grep -qE "[0-9]+ skipped" "$R/tests.log" && { echo "tests were skipped ($R/tests.log)"; exit 1; }
echo "tests: $(tail -1 "$R/tests.log")"
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_g0_replay.py --wm "$WM" --norm "$OUT/data/go2_norm.npz" --physics "$ROUND0" \
  --withheld "$OUT/data/withheld.json" --g0-policy "$G0" --out "$R/run" --steps 100 --start-every 50 --device cpu \
  > "$R/replay.log" 2>&1 || { echo "replay failed ($R/replay.log)"; tail -5 "$R/replay.log"; exit 1; }
cat "$R/run/g0_replay.txt"
sha256sum "$R/run/g0_replay.json"
