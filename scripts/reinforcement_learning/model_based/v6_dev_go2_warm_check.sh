#!/bin/bash
# A4 development check D0 (21 Nov 2026): cold vs warm imagination resets through the real PPO runner.
# Two short PPO runs (300 iterations) in the saved G0 world model, every G0 setting unchanged (s2 reward contract,
# round-0 start data, lambda 1, PPO seed 1000, paper recipe) except the reset history: paper (cold) vs paper_warm.
# Development evidence only; A3 stays closed and G0 stays a valid FAIL. Reading rule: analysis/go2_warm_check.py.
# Usage, from the robotic_world_model root in the container, with setsid nohup:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_dev_go2_warm_check.sh logs/p2/a4_d0_1121 \
#       logs/p2/a3_g0_1121 > /tmp/a4_d0.out 2>&1 < /dev/null &
set -uo pipefail
OUT=${1:?fresh out dir}
G0=${2:?the G0 out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
E16=${E16:-logs/p2/v6w2_rest_1118/e16_s2_check.npz}
WM=$G0/wm/g0_wm_seed1.pt
WM_SHA=${WM_SHA:-9a08ffcbe71b70d19203217b6ea4c856ec05ac70f37e9c1ba951f3ec89c655b2}   # run-log 21 Nov
ITERS=${ITERS:-300}
PPO_SEED=1000
LOGS=logs/offline/go2_flat
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
status() { echo "$(date -u +%FT%TZ) [a4-d0] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
for f in "$WM" "$E16" "$G0/data/start_data/start_data.json"; do [ -s "$f" ] || fail "missing $f"; done
[ "$(sha256sum "$WM" | cut -d' ' -f1)" = "$WM_SHA" ] || fail "model hash differs from the run-log (21 Nov)"
{ git rev-parse HEAD; git status --short | grep -v '^??'; } > "$OUT/git.txt"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_g0_replay.py $AN/tests/test_go2_g0.py > "$OUT/tests.log" 2>&1 \
  || fail "tests failed (tests.log)"
grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log)"
status "tests: $(tail -1 "$OUT/tests.log")"
for ARM in cold:paper:3101 warm:paper_warm:3102; do
  IFS=: read -r NAME MODE RUN <<< "$ARM"
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12288 ]; do
    status "waiting for 12 GiB GPU before $NAME"; sleep 300; done
  status "$NAME: run $RUN, mode $MODE, $ITERS iterations, PPO seed $PPO_SEED"
  CUDA_VISIBLE_DEVICES=0 $PY $MB/e1_train.py --wm-checkpoint "$WM" --penalty-weight -1.0 --reward-contract s2 \
    --asset-trace "$E16" --start-data "$G0/data/start_data" --max-iterations $ITERS $MODE $RUN go2_flat --seed $PPO_SEED \
    > "$OUT/train_$NAME.log" 2>&1 || fail "$NAME training exited non-zero (train_$NAME.log)"
  for pat in "\[go2\] reward contract s2: 16 terms" "\[go2\] start data" "\[dev\] max_iterations = $ITERS" \
             "warm_start_resets= $([ "$NAME" = warm ] && echo True || echo False)"; do
    grep -q "$pat" "$OUT/train_$NAME.log" || fail "train_$NAME.log lacks '$pat'"
  done
  status "$NAME done"
done
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_warm_check.py --cold-log "$OUT/train_cold.log" --warm-log "$OUT/train_warm.log" \
  --expected-iterations "$ITERS" --out "$OUT/warm_check.json" 2>&1 | tee "$OUT/warm_check.txt" \
  || fail "could not score the cold/warm pair (warm_check.txt)"
[ -s "$OUT/warm_check.json" ] || fail "scorer did not write warm_check.json"
status "done; $(sha256sum "$OUT/warm_check.json")"
