#!/bin/bash
# A4 D4 (22 Nov 2026, Charbel's decision after D3 NOT MET): test first on the D3 policy, before anything is fitted.
# Phases (from the robotic_world_model root in the container; each writes into its own folder under OUT):
#   timing <out>  CPU   where D3's extra falls occur, from the saved D3 evaluations (no model)
#   record <out>  GPU   D3 at 0.30 friction: held-out probe (seed 915, 64 x 2000) and a batch for the later matched
#                       pilot (seed 103, 64 x 5000; not read by D4); friction read back; recordings made read-only
#   score  <out>  CPU   the unchanged qualified model (741b352f...) on the D3 probe, s2's probe 913 as reference:
#                       recorded vs free arm on the same fall windows (analysis/go2_test_first.py)
# Example:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_dev_go2_d4.sh timing logs/p2/a4_d4_1122 > /tmp/a4_d4_timing.out 2>&1 < /dev/null &
set -uo pipefail
PHASE=${1:?phase: timing, record or score}
OUT=${2:?out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
REC=scripts/reinforcement_learning/rsl_rl/record_pre_reset_transitions.py
TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0
D2=logs/p2/a4_d2_1121
D3=logs/p2/a4_d3_1122
WM=$D2/option1/run/d2s_wm_seed2.pt
WM_SHA=${WM_SHA:-741b352fab202f0b898d23908ccdc7f3453409492f970feacc145476cdd16d11}        # run-log 22 Nov (option 1)
GATE_SHA=${GATE_SHA:-d82dba1576e3d7887e9a73f48ae7c8d41b640542f44e5b38340ea82725c9ba2f}      # run-log 22 Nov (D3 result)
PROBE3=logs/p2/a4_d1_probe3/probe.npz
PROBE3_SHA=${PROBE3_SHA:-0acfd1b153f025073e30cdfe0373da4931292054301857b85666c3089e899284}    # run-log 22 Nov (s2 probe 913)
FRICTION=0.30; PROBE_SEED=915; BATCH_SEED=103
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
mkdir -p "$OUT" || exit 1
status() { echo "$(date -u +%FT%TZ) [a4-d4 $PHASE] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
gpu_wait() { until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 8192 ]; do
  status "waiting for 8 GiB GPU before $1"; sleep 300; done; }
tests() {  # $1: log name; the analysis tests of this step, never skipped
  { git rev-parse HEAD; git status --short | grep -v '^??'; sha256sum $AN/go2_d3_fall_timing.py $AN/go2_test_first.py; } > "$OUT/git_$PHASE.txt"
  CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_d3_fall_timing.py $AN/tests/test_go2_test_first.py \
    $AN/tests/test_go2_c0_check.py > "$OUT/$1" 2>&1 || fail "tests ($1)"
  grep -qE "[0-9]+ skipped" "$OUT/$1" && fail "tests were skipped ($1)"
  status "tests: $(tail -1 "$OUT/$1")"
}
d3_policy() {  # the verified D3 policy from the resume-eval line; its sha256 must start with the logged prefix
  local line p h
  line=$(grep "resume-eval: verified 300 PPO iterations" "$D3/queue.log" | tail -1)
  p=$(echo "$line" | awk '{print $(NF-1)}'); h=$(echo "$line" | awk '{print $NF}')
  [ -s "$p" ] || fail "D3 policy not found (from $D3/queue.log): '$p'"
  [ "$(sha256sum "$p" | cut -c1-12)" = "$h" ] || fail "D3 policy hash differs from $D3/queue.log"
  echo "$p"
}

case "$PHASE" in
timing)
  T=$OUT/timing; [ -e "$T" ] && { echo "exists: $T"; exit 2; }
  [ "$(sha256sum "$D3/d3_gate.json" | cut -d' ' -f1)" = "$GATE_SHA" ] || fail "d3_gate.json differs from the run-log"
  tests tests_timing.log
  CUDA_VISIBLE_DEVICES="" $PY $AN/go2_d3_fall_timing.py --eval "$D3/eval" --out "$T" > "$OUT/timing.log" 2>&1 \
    || { tail -5 "$OUT/timing.log"; fail "fall timing ($OUT/timing.log)"; }
  cat "$T/d3_fall_timing.txt" | tee -a "$OUT/queue.log"
  status "done; $(sha256sum "$T/d3_fall_timing.json")" ;;
record)
  R=$OUT/rec; [ -e "$R" ] && { echo "exists: $R"; exit 2; }
  P=$(d3_policy) || exit 1
  mkdir -p "$R"
  status "D3 policy $P ($(sha256sum "$P" | cut -c1-12))"
  for job in "probe $PROBE_SEED 2000" "batch $BATCH_SEED 5000"; do
    set -- $job; F="$R/$1.npz"
    gpu_wait "$1"
    status "recording $1: D3, friction x$FRICTION, seed $2, 64 envs x $3 steps"
    CUDA_VISIBLE_DEVICES=0 $PY $REC --task $TASK --checkpoint "$P" --num_envs 64 --num_steps $3 --seed $2 \
      --cmd_resample_s 10 --friction-scale $FRICTION --headless --out "$F" > "$F.log" 2>&1 || fail "recording $F ($F.log)"
    grep -q "\[friction\] verified in the simulator" "$F.log" || fail "no friction read-back in $F.log"
    [ -s "$F" ] || fail "the recorder exited without writing $F ($F.log)"
    MIN=""; [ "$1" = probe ] && MIN="--min-falls 30 --min-time-outs 30"
    CUDA_VISIBLE_DEVICES="" $PY $AN/go2_friction_shift.py count --trace "$F" --out "$R/$1.counts.json" $MIN 2>&1 \
      | tee -a "$OUT/queue.log" || status "$1: minimum (30 falls, 30 surviving) not met; reported; Charbel decides"
    chmod a-w "$F" "$R/$1.counts.json"
  done
  status "done; $(sha256sum "$R/probe.npz" "$R/batch.npz" | tr '\n' ' ')" ;;
score)
  S=$OUT/score; [ -e "$S" ] && { echo "exists: $S"; exit 2; }
  [ -s "$OUT/rec/probe.npz" ] || fail "no D3 probe; run the record phase first"
  [ "$(sha256sum "$WM" | cut -d' ' -f1)" = "$WM_SHA" ] || fail "model hash differs from the run-log"
  [ "$(sha256sum "$PROBE3" | cut -d' ' -f1)" = "$PROBE3_SHA" ] || fail "probe 913 hash differs from the run-log"
  status "D3 probe $(sha256sum "$OUT/rec/probe.npz" | cut -c1-12)"
  tests tests_score.log
  CUDA_VISIBLE_DEVICES="" $PY $AN/go2_test_first.py --wm "$WM" --norm "$D2/data/go2_norm.npz" \
    --probe "d3_915=$OUT/rec/probe.npz" --probe "s2_913=$PROBE3" --out "$S" > "$OUT/score.log" 2>&1 \
    || { tail -5 "$OUT/score.log"; fail "test-first scoring ($OUT/score.log)"; }
  cat "$S/test_first.txt" | tee -a "$OUT/queue.log"
  status "done; $(sha256sum "$S/test_first.json")" ;;
*) echo "phase must be timing, record or score"; exit 2 ;;
esac
