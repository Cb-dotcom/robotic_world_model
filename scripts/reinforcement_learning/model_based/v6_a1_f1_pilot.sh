#!/bin/bash
# Plan v6 amendment A1 (19 Nov 2026): the F1 mean-matched development pilot, in two phases.
# Protocol (frozen before training): vault 10 Research Direction/plan-amendment-a1-f1-pilot.ai.md.
#
#   Phase 1, CPU (~2 min):  v6_a1_f1_pilot.sh calibrate <fresh_out_dir>
#       tests, then analysis/f1_calibrate.py on D's model and the never-fitted reference envs 52-63:
#       s(d) for d = 0..255 and the gain g, frozen in <out>/f1_calibration.json. Its SHA-256 is printed and goes
#       to the run-log before phase 2.
#   Phase 2, GPU (~one paper training + 4 evaluations):
#       EXPECT_CAL_SHA=<sha of phase 1> v6_a1_f1_pilot.sh train <the same out dir>
#       run 2601 = arm D's recipe (e1_train.py, paper, seed 1, D's model, lambda 1) + F1 from the frozen file;
#       e1v_match.py against run 2004 (arm D) while training continues; f1_settings.json beside run_args.json;
#       evaluation as D1 (Pretrain, 32 x 10000, seeds 42/43, std .5, commands 10 and 30 s); go / no-go by
#       analysis/f1_pilot_score.py against D1's existing evaluation logs. NO-GO = report and stop.
# Run from the robotic_world_model root in the container, phase 2 with setsid nohup.
set -uo pipefail
PHASE=${1:?calibrate or train}
OUT=${2:?out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
LOGS=logs/offline/anymal_d_flat
E1V=logs/p2/e1v_1113_0939
WM=${WM:-$E1V/wm/e1v_wm.pt}
CAL_TRACE=${CAL_TRACE:-$E1V/data/test_reference.npz}
D_EVAL=${D_EVAL:-$E1V/eval}
TASK=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
EVAL=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
RUN=2601
SEED=1
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
status() { echo "$(date -u +%FT%TZ) [$PHASE] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
need() { for f in "$@"; do [ -s "$f" ] || fail "missing $f"; done; }

if [ "$PHASE" = calibrate ]; then
  [ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
  mkdir -p "$OUT"
  need "$WM" "$CAL_TRACE"
  { git rev-parse HEAD; git status --short | grep -v '^??' ; } > "$OUT/git.txt"
  CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_f1_env.py $AN/tests/test_f1_calibrate.py $AN/tests/test_c0_battery.py \
    > "$OUT/tests.log" 2>&1 || fail "CPU tests failed (tests.log)"
  grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log): rsl_rl not importable?"
  status "tests passed"
  CUDA_VISIBLE_DEVICES="" $PY $AN/f1_calibrate.py --wm "$WM" --trace "$CAL_TRACE" --out "$OUT/calibration" \
    > "$OUT/calibration.log" 2>&1 || fail "calibration ($(tail -1 "$OUT/calibration.log"))"
  cp "$OUT/calibration/f1_calibration.json" "$OUT/f1_calibration.json"
  chmod a-w "$OUT/f1_calibration.json"
  status "calibration frozen: $(sha256sum "$OUT/f1_calibration.json")"
  cat "$OUT/calibration/f1_calibration.txt"
  echo "NEXT: record the SHA-256 above in the run-log, then:"
  echo "  EXPECT_CAL_SHA=$(sha256sum "$OUT/f1_calibration.json" | cut -d' ' -f1) setsid nohup $0 train $OUT > $OUT.train.nohup 2>&1 < /dev/null &"
  exit 0
fi

[ "$PHASE" = train ] || { echo "phase must be calibrate or train"; exit 2; }
[ -d "$OUT" ] || { echo "run phase 1 first"; exit 2; }
CAL="$OUT/f1_calibration.json"
need "$CAL" "$WM"
[ "$(sha256sum "$CAL" | cut -d' ' -f1)" = "${EXPECT_CAL_SHA:?set EXPECT_CAL_SHA to the phase-1 SHA-256}" ] \
  || fail "f1_calibration.json differs from the frozen SHA-256"
DDIRS=( $LOGS/*_2004 ); [ ${#DDIRS[@]} -eq 1 ] || fail "need exactly one $LOGS/*_2004 (arm D)"
D_DIR=${DDIRS[0]}
need "$D_DIR/run_args.json"
CTRL_WM=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['world_model_checkpoint']['sha256'])" "$D_DIR/run_args.json" 2>/dev/null | tail -1)
[ "$CTRL_WM" = "$(sha256sum "$WM" | cut -d' ' -f1)" ] || fail "run 2004 was trained with another world model (${CTRL_WM:-none}); A1 needs the same file"
for C in 10 30; do for S in 42 43; do need "$D_EVAL/D_cmd${C}_s${S}/eval.log"; done; done
ls -d $LOGS/*_$RUN >/dev/null 2>&1 && fail "run $RUN already exists"
git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
{ git rev-parse HEAD; git status --short | grep -v '^??' ; } > "$OUT/git_train.txt"
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12288 ]; do
  status "waiting for 12 GiB GPU before training"; sleep 300; done
T0=$(date -u +%s)
status "training run $RUN: arm D recipe + F1 (calibration $(sha256sum "$CAL" | cut -c1-12)), seed $SEED, WM $WM"
CUDA_VISIBLE_DEVICES=0 setsid $PY $MB/e1_train.py --wm-checkpoint "$WM" --penalty-weight -1.0 --f1-calibration "$CAL" \
  paper $RUN anymal_d_flat --seed $SEED > "$OUT/train_F1.log" 2>&1 &
PID=$!
RD=""
for i in $(seq 1 240); do
  RD=$(ls -d $LOGS/*_$RUN 2>/dev/null | tail -1)
  [ -n "$RD" ] && [ -s "$RD/run_args.json" ] && break
  kill -0 $PID 2>/dev/null || break; sleep 15; done
if [ -z "$RD" ] || [ ! -s "$RD/run_args.json" ]; then
  kill -TERM -- -$PID 2>/dev/null; fail "training wrote no run_args.json (train_F1.log: $(tail -1 "$OUT/train_F1.log"))"; fi
grep -q "\[e1\] F1 on: members 5" "$OUT/train_F1.log" || { kill -TERM -- -$PID 2>/dev/null; fail "F1 not switched on (train_F1.log)"; }
if ! $PY $MB/e1v_match.py --control "$D_DIR/run_args.json" --candidate "$RD/run_args.json" \
     --expect-wm-sha "$(sha256sum "$WM" | cut -d' ' -f1)" --expect-penalty -1.0 > "$OUT/match_F1.txt" 2>&1; then
  kill -TERM -- -$PID 2>/dev/null; wait $PID 2>/dev/null; fail "matching check vs run 2004 failed (match_F1.txt)"; fi
cp "$CAL" "$RD/f1_settings.json"
status "matching check vs 2004 passed; f1_settings.json stored in $RD; training continues"
wait $PID || fail "training exited non-zero (train_F1.log)"
T1=$(date -u +%s)
[ -s "$RD/policy_2499.pt" ] || fail "no policy_2499.pt in $RD"
status "trained in $(( (T1 - T0) / 60 )) min"
D_START=$(grep -m1 "training arm D" "$E1V/queue.log" 2>/dev/null | cut -d' ' -f1)
D_END=$(grep -m1 "\[D\] trained" "$E1V/queue.log" 2>/dev/null | cut -d' ' -f1)
if [ -n "$D_START" ] && [ -n "$D_END" ]; then
  D_MIN=$(( ($(date -u -d "$D_END" +%s) - $(date -u -d "$D_START" +%s)) / 60 ))
  status "cost: F1 $(( (T1 - T0) / 60 )) min vs arm D $D_MIN min (D shared the GPU with arm C; plan §8: K = 3 for a full study if > 2x)"
else
  status "cost: F1 $(( (T1 - T0) / 60 )) min; arm D time not found in $E1V/queue.log"
fi
for C in 30 10; do for S in 42 43; do
  DIR="$OUT/eval/F1_cmd${C}_s${S}"; mkdir -p "$DIR"
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 8192 ]; do
    status "waiting for 8 GiB GPU before eval cmd$C s$S"; sleep 300; done
  status "eval cmd$C seed $S"
  CUDA_VISIBLE_DEVICES=0 $PY $EVAL --task $TASK --checkpoint "$RD/policy_2499.pt" --num_envs 32 --num_steps 10000 \
    --seed $S --std .5 --cmd_resample_s $C --headless --out_csv "$DIR/episodes.csv" > "$DIR/eval.log" 2>&1 \
    || fail "eval cmd$C s$S"
done; done
CUDA_VISIBLE_DEVICES="" $PY $AN/f1_pilot_score.py --pilot "$OUT/eval" --d1 "$D_EVAL" --out "$OUT/f1_pilot_score.json" \
  > "$OUT/score.log" 2>&1 || fail "scoring (score.log)"
status "done"
cat "$OUT/score.log"; cat "$OUT/match_F1.txt" | tail -3; grep "cost:" "$OUT/queue.log"
