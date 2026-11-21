#!/bin/bash
# A4 D1 (21 Nov 2026): Go2 failure probe and fitting batch under a static ground-friction shift (Charbel: friction,
# start at 0.60 x nominal). Recorder: s2, Go2-Start-v0, 10 s commands, no pushes, ground friction scaled and read
# back from the simulator before stepping (friction_shift.py). Counts: analysis/go2_friction_shift.py.
#   Phase select:  v6_dev_go2_d1.sh select <fresh out>   levels 0.60 -> 0.45 -> 0.30, 64 envs x 1000 steps, seed 900;
#                  stops at the first level meeting the rule; writes choice.json (frozen, read-only).
#   Phase collect: v6_dev_go2_d1.sh collect <same out> [level]   at the frozen level: held-out probe 64 x 2000,
#                  seed 911 (minimum 30 falls and 30 surviving episodes, A4 §3 option A; else stop and report);
#                  fitting batch 64 x 5000, seed 102. If no level met the selection rule, [level] must be given: it is
#                  the level fixed by Charbel's decision (A4 §3) and is stored read-only in decision.json.
# Count files are *.counts.json (the recorder writes its own <name>.json sidecar).
# Run from the robotic_world_model root in the container, with setsid nohup. Development evidence (A4); A3 stays closed.
set -uo pipefail
PHASE=${1:?select or collect}
OUT=${2:?out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
REC=scripts/reinforcement_learning/rsl_rl/record_pre_reset_transitions.py
TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0
ROUND0=${ROUND0:-logs/p2/go2_round0_1118/round0_s2_seed101.npz}
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
status() { echo "$(date -u +%FT%TZ) [a4-d1 $PHASE] $*" 2>&1 | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
gpu_wait() { until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 8192 ]; do
  status "waiting for 8 GiB GPU before $1"; sleep 300; done; }
record() {   # factor seed steps out
  gpu_wait "$4"
  status "recording $4: friction x$1, seed $2, 64 envs x $3 steps"
  CUDA_VISIBLE_DEVICES=0 $PY $REC --task $TASK --checkpoint "$CK" --num_envs 64 --num_steps $3 --seed $2 \
    --cmd_resample_s 10 --friction-scale $1 --headless --out "$4" > "$4.log" 2>&1 || fail "recording $4 ($4.log)"
  grep -q "\[friction\] verified in the simulator" "$4.log" || fail "no friction read-back in $4.log"
  [ -s "$4" ] || fail "the recorder exited without writing $4 (Isaac can exit 0 after a crash; see $4.log)"
}
[ -s "$ROUND0" ] || { echo "missing $ROUND0"; exit 1; }
CK=$(CUDA_VISIBLE_DEVICES="" $PY -c "import json, sys, numpy as np; print(json.loads(str(np.load(sys.argv[1])['metadata']))['checkpoint'])" "$ROUND0" 2>/dev/null | tail -1)
[ -s "$CK" ] || { echo "s2 checkpoint not found: $CK"; exit 1; }

if [ "$PHASE" = select ]; then
  [ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
  mkdir -p "$OUT/select"
  { git rev-parse HEAD; git status --short | grep -v '^??'; } > "$OUT/git.txt"
  CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_friction_shift.py > "$OUT/tests.log" 2>&1 || fail "tests (tests.log)"
  grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log)"
  status "tests: $(tail -1 "$OUT/tests.log"); s2 checkpoint $CK"
  COUNTS=()
  for F in 0.60 0.45 0.30; do
    record $F 900 1000 "$OUT/select/f$F.npz"
    CUDA_VISIBLE_DEVICES="" $PY $AN/go2_friction_shift.py count --trace "$OUT/select/f$F.npz" --out "$OUT/select/f$F.counts.json" \
      2>&1 | tee -a "$OUT/queue.log" || fail "count f$F"
    COUNTS+=("$OUT/select/f$F.counts.json")
    if CUDA_VISIBLE_DEVICES="" $PY $AN/go2_friction_shift.py choose --counts "${COUNTS[@]}" --out "$OUT/choice.json" \
        2>&1 | tee -a "$OUT/queue.log"; then
      chmod a-w "$OUT/choice.json" "$OUT"/select/*; status "level frozen: $(sha256sum "$OUT/choice.json")"; exit 0
    fi
  done
  fail "no level meets the rule (choice.json); reported; Charbel decides (A4 §1)"
fi

[ "$PHASE" = collect ] || { echo "phase must be select or collect"; exit 2; }
[ -s "$OUT/choice.json" ] || fail "no frozen choice.json (run select first)"
F=$(CUDA_VISIBLE_DEVICES="" $PY -c "import json, sys; c = json.load(open(sys.argv[1]))['chosen']; print('' if c is None else f'{c:.2f}')" "$OUT/choice.json" | tail -1)
if [ -n "$F" ]; then
  [ -z "${3:-}" ] || [ "$3" = "$F" ] || fail "level $3 differs from the frozen choice $F"
else
  F=${3:-}
  [ -n "$F" ] || fail "no level met the selection rule: give the level fixed by Charbel's decision (A4 §3)"
  [ -s "$OUT/select/f$F.counts.json" ] || fail "level $F was not in the selection ladder"
  if [ -e "$OUT/decision.json" ]; then
    grep -q "\"level\": $F," "$OUT/decision.json" || fail "decision.json holds another level"
  else
    printf '{"level": %s, "source": "Charbel 21 Nov 2026 23:26 Rome, A4 section 3 option A", "negatives": "surviving (time-out) episodes"}\n' "$F" > "$OUT/decision.json"
    chmod a-w "$OUT/decision.json"
  fi
  status "level $F from the decision (decision.json)"
fi
[ -e "$OUT/probe.npz" ] && fail "probe.npz exists"
record $F 911 2000 "$OUT/probe.npz"
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_friction_shift.py count --trace "$OUT/probe.npz" --out "$OUT/probe.counts.json" \
  --min-falls 30 --min-time-outs 30 2>&1 | tee -a "$OUT/queue.log" \
  || { chmod a-w "$OUT/probe.npz" "$OUT/probe.counts.json"; fail "probe minimum (30 falls, 30 surviving episodes) not met; reported; Charbel decides (A4 §3)"; }
record $F 102 5000 "$OUT/fit_batch.npz"
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_friction_shift.py count --trace "$OUT/fit_batch.npz" --out "$OUT/fit_batch.counts.json" \
  2>&1 | tee -a "$OUT/queue.log" || fail "count fit batch"
chmod a-w "$OUT/probe.npz" "$OUT/probe.counts.json" "$OUT/fit_batch.npz" "$OUT/fit_batch.counts.json"
status "done; $(sha256sum "$OUT/probe.npz" "$OUT/fit_batch.npz" | tr '\n' ' ')"
