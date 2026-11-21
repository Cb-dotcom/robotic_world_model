#!/bin/bash
# Plan v6 amendment A3 §4 (21 Nov 2026): gate G0, the first Go2 gate, in two phases.
# Protocol (frozen 21 Nov): vault 10 Research Direction/plan-amendment-a3-go2-study.ai.md.
#
#   Phase 1, CPU (~5 min):  v6_a3_g0.sh prepare <fresh_out_dir>
#       input checks (round-0 recording hash, E16 trace, s2 reference summaries, s2 checkpoint), tests,
#       the round-0 manifest (go2_round0_prepare.py: E22 split seed 101, segments, start data, normalizer),
#       the runtime reward check (training path, must PASS). The data folder is write-protected and
#       manifest.json's SHA-256 printed; it goes to the run-log before phase 2.
#   Phase 2, GPU:  EXPECT_MANIFEST_SHA=<sha> v6_a3_g0.sh g0 <the same out dir>
#       fit (seed 1, from scratch, A3 §3) -> s2 probe in imagination (descriptive) -> one lambda-1 policy
#       (paper recipe, s2 contract, round-0 start data, PPO seed 1000, run 3001) -> physics evaluation (G6
#       protocol, 4 cells) -> go2_g0_gate.py. A valid FAIL stops the scored U/R study (A3 §4 fallback).
# Run from the robotic_world_model root in the container, phase 2 with setsid nohup.
set -uo pipefail
PHASE=${1:?prepare or g0}
OUT=${2:?out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
ROUND0=${ROUND0:-logs/p2/go2_round0_1118/round0_s2_seed101.npz}
ROUND0_SHA=${ROUND0_SHA:-3e7ce462a885ebec7e17cf1668d3499a26efe71612b65bd7d55ebf6cbdb41456}   # run-log 18 Nov; override only in dry runs
E16=${E16:-logs/p2/v6w2_rest_1118/e16_s2_check.npz}
S2_EVAL=${S2_EVAL:-logs/p2/startunitree_1115_2350/eval}
TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0
EVAL=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
LOGS=logs/offline/go2_flat
RUN=3001
FIT_SEED=1
PPO_SEED=1000
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
status() { echo "$(date -u +%FT%TZ) [$PHASE] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
need() { for f in "$@"; do [ -s "$f" ] || fail "missing $f"; done; }
gpu_wait() { until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$1" ]; do
  status "waiting for $(( $1 / 1024 )) GiB GPU before $2"; sleep 300; done; }

if [ "$PHASE" = prepare ]; then
  [ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
  mkdir -p "$OUT"
  need "$ROUND0" "$E16"
  [ "$(sha256sum "$ROUND0" | cut -d' ' -f1)" = "$ROUND0_SHA" ] || fail "round-0 recording hash differs from the run-log (18 Nov)"
  for C in 10 30; do for S in 42 43; do need "$S2_EVAL/unitree_s2_cmd${C}_s${S}/episodes_summary.json"; done; done
  CK=$(CUDA_VISIBLE_DEVICES="" $PY -c "import json, sys, numpy as np; print(json.loads(str(np.load(sys.argv[1])['metadata']))['checkpoint'])" "$ROUND0" 2>/dev/null | tail -1)
  need "$CK"
  { git rev-parse HEAD; git status --short | grep -v '^??' ; } > "$OUT/git.txt"
  CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_g0.py $AN/tests/test_e16.py > "$OUT/tests.log" 2>&1 \
    || fail "CPU tests failed (tests.log)"
  grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log): rsl_rl not importable?"
  status "tests passed: $(tail -1 "$OUT/tests.log")"
  CUDA_VISIBLE_DEVICES="" $PY $MB/go2_round0_prepare.py --trace "$ROUND0" --out "$OUT/data" > "$OUT/prepare.log" 2>&1 \
    || fail "round-0 manifest ($(tail -1 "$OUT/prepare.log"))"
  cat "$OUT/prepare.log"
  CUDA_VISIBLE_DEVICES="" $PY $AN/go2_reward_runtime_check.py --trace "$E16" --out "$OUT/reward_runtime" \
    > "$OUT/reward_runtime.log" 2>&1 || { cat "$OUT/reward_runtime.log"; fail "runtime reward check (reward_runtime.log)"; }
  cat "$OUT/reward_runtime/reward_runtime.txt"
  chmod -R a-w "$OUT/data"
  status "manifest frozen: $(sha256sum "$OUT/data/manifest.json")"
  echo "NEXT: record the SHA-256 above in the run-log, then:"
  echo "  EXPECT_MANIFEST_SHA=$(sha256sum "$OUT/data/manifest.json" | cut -d' ' -f1) setsid nohup $0 g0 $OUT > $OUT.g0.nohup 2>&1 < /dev/null &"
  exit 0
fi

[ "$PHASE" = g0 ] || { echo "phase must be prepare or g0"; exit 2; }
[ -d "$OUT/data" ] || { echo "run phase 1 first"; exit 2; }
need "$OUT/data/manifest.json" "$OUT/data/go2_norm.npz" "$OUT/data/start_data/start_data.json" "$E16" "$ROUND0"
[ "$(sha256sum "$OUT/data/manifest.json" | cut -d' ' -f1)" = "${EXPECT_MANIFEST_SHA:?set EXPECT_MANIFEST_SHA to the phase-1 SHA-256}" ] \
  || fail "manifest.json differs from the frozen SHA-256"
ls -d $LOGS/*_$RUN >/dev/null 2>&1 && fail "run $RUN already exists"
git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
{ git rev-parse HEAD; git status --short | grep -v '^??' ; } > "$OUT/git_g0.txt"
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
NORM_SHA=$(sha256sum "$OUT/data/go2_norm.npz" | cut -d' ' -f1)
mkdir -p "$OUT/wm"
WM="$OUT/wm/g0_wm_seed$FIT_SEED.pt"
T0=$(date -u +%s)

# ---- 1. fit (A3 §3)
gpu_wait 12288 "the fit"
status "fit seed $FIT_SEED: from scratch, 2000 it x 20 x 2048, lr 1e-3, pos-weight 1, per-head reset"
CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config go2_flat --data "$OUT/data/segments" \
  --output "$WM" --norm_stats "$OUT/data/go2_norm.npz" --seed $FIT_SEED --iterations 2000 --num_mini_batches 20 \
  --mini_batch_size 2048 --lr 1e-3 --weight_decay 0 --max_grad_norm 1 --termination_loss_weight 1 \
  --termination_pos_weight 1 --lazy_windows --save_interval 500 --device cuda:0 > "$OUT/fit.log" 2>&1 || fail "fit (fit.log)"
need "$WM"
grep -q "normalizer loaded from .*$NORM_SHA" "$OUT/fit.log" || fail "the fit did not use the frozen normalizer (fit.log)"
CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
sys.exit(0 if ck.get('provenance', {}).get('b3_reset_per_head') is True and ck.get('iter') == 2000 and ck.get('normalized') else 1)" "$WM" \
  || fail "fit checkpoint lacks the per-head reset, iteration 2000 or its normalizer"
T1=$(date -u +%s)
status "fitted in $(( (T1 - T0) / 60 )) min: $(sha256sum "$WM")"

# ---- 2. s2 probe in imagination (descriptive; a broken observation layout stops here)
gpu_wait 8192 "the s2 probe"
CUDA_VISIBLE_DEVICES=0 $PY $AN/go2_imagination_probe.py --wm "$WM" --start-data "$OUT/data/start_data" \
  --asset-trace "$E16" --physics "$ROUND0" --out "$OUT/s2_probe" --envs 256 --steps 1000 --seed 0 --device cuda:0 \
  > "$OUT/s2_probe.log" 2>&1 || fail "s2 probe (s2_probe.log)"
cat "$OUT/s2_probe/s2_probe.txt"
T2=$(date -u +%s)

# ---- 3. one lambda-1 policy (paper recipe, s2 contract, round-0 start data)
gpu_wait 12288 "policy training"
status "training run $RUN: paper recipe, lambda 1, s2 contract, round-0 start data, PPO seed $PPO_SEED"
CUDA_VISIBLE_DEVICES=0 setsid $PY $MB/e1_train.py --wm-checkpoint "$WM" --penalty-weight -1.0 --reward-contract s2 \
  --asset-trace "$E16" --start-data "$OUT/data/start_data" paper $RUN go2_flat --seed $PPO_SEED > "$OUT/train_G0.log" 2>&1 &
PID=$!
RD=""
for i in $(seq 1 240); do
  RD=$(ls -d $LOGS/*_$RUN 2>/dev/null | tail -1)
  [ -n "$RD" ] && [ -s "$RD/run_args.json" ] && break
  kill -0 $PID 2>/dev/null || break; sleep 15; done
if [ -z "$RD" ] || [ ! -s "$RD/run_args.json" ]; then
  kill -TERM -- -$PID 2>/dev/null; fail "training wrote no run_args.json (train_G0.log: $(tail -1 "$OUT/train_G0.log"))"; fi
for pat in "\[go2\] reward contract s2: 16 terms" "\[go2\] start data" "uncertainty_penalty_weight = -1.0"; do
  grep -q "$pat" "$OUT/train_G0.log" || { kill -TERM -- -$PID 2>/dev/null; fail "training log lacks '$pat'"; }
done
CUDA_VISIBLE_DEVICES="" $PY - "$RD/go2_settings.json" "$E16" "$OUT/data" "$WM" <<'PYEOF' || { kill -TERM -- -$PID 2>/dev/null; fail "go2_settings.json"; }
import json, sys
sys.path.insert(0, "scripts/reinforcement_learning/model_based")
import go2_study, train as T
out, e16, data, wm = sys.argv[1:]
cfg = T.resolve_task_config("go2_flat")
rec = dict(contract=go2_study.apply_s2_contract(cfg, e16), start=go2_study.apply_start_data(cfg, data + "/start_data"),
           manifest_sha256=go2_study.sha256(data + "/manifest.json"), wm=wm, wm_sha256=go2_study.sha256(wm),
           protocol="plan v6 amendment A3 §4 gate G0")
json.dump(rec, open(out, "x"), indent=1)
PYEOF
status "run $RUN started in $RD; go2_settings.json stored"
wait $PID || fail "training exited non-zero (train_G0.log)"
need "$RD/policy_2499.pt"
T3=$(date -u +%s)
status "trained in $(( (T3 - T2) / 60 )) min"

# ---- 4. physics evaluation, s2's G6 protocol
for C in 30 10; do for S in 42 43; do
  DIR="$OUT/eval/G0_cmd${C}_s${S}"; mkdir -p "$DIR"
  gpu_wait 8192 "eval cmd$C s$S"
  status "eval cmd$C seed $S"
  CUDA_VISIBLE_DEVICES=0 $PY $EVAL --task $TASK --checkpoint "$RD/policy_2499.pt" --num_envs 32 --num_steps 10000 \
    --seed $S --std .5 --cmd_resample_s $C --headless --out_csv "$DIR/episodes.csv" > "$DIR/eval.log" 2>&1 \
    || fail "eval cmd$C s$S"
done; done
T4=$(date -u +%s)

# ---- 5. gate
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_g0_gate.py --g0 "$OUT/eval" --s2 "$S2_EVAL" --out "$OUT/g0_gate.json" \
  > "$OUT/gate.log" 2>&1 || fail "gate scoring (gate.log)"
status "wall time: fit $(( (T1 - T0) / 60 )) min, probe $(( (T2 - T1) / 60 )), train $(( (T3 - T2) / 60 )), eval $(( (T4 - T3) / 60 )) min"
status "done: $(grep VERDICT "$OUT/gate.log")"
cat "$OUT/gate.log"
