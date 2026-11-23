#!/bin/bash
# A4 D7 (22 Nov 2026, Charbel's decision 20:47): D3's exact recipe in the D6 model. Runs only if D6 passed.
# Model: D6's stop-retrained model (hash from the D6 verdict line), warm resets plus D6's startup share q of resets from
# recorded physical reset states (zero memory), lambda 1, s2 reward contract, D6 start data, PPO from s2 (new
# optimizer), 300 iterations, PPO seed 1000, run 4002.
# Evaluation: as D3 (10 s commands, seeds 42 + 43, friction x0.30 read back) for the D7 policy and unchanged s2;
# gate: D3's unchanged all-falls gate (analysis/go2_d3_gate.py, a3584a16...) on a view where D7's folders carry the
# gate's policy tag; startup risk reported separately (go2_startup_hazard.py; go2_d3_fall_timing.py).
# Usage, from the robotic_world_model root in the container:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_dev_go2_d7.sh logs/p2/a4_d7_1122 > /tmp/a4_d7.out 2>&1 < /dev/null &
set -uo pipefail
OUT=${1:?out dir}
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
EVAL=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0
D6=${D6:-logs/p2/a4_d6_1122}   # D6b: D6=logs/p2/a4_d6b_1123
WM=$D6/option1/run/d6s_wm_seed2.pt
START=$D6/data/start_data
RESETS=$D6/data/reset_states.npz
E16=${E16:-logs/p2/v6w2_rest_1118/e16_s2_check.npz}
ROUND0=${ROUND0:-logs/p2/go2_round0_1118/round0_s2_seed101.npz}
GATE_SHA=a3584a16be034c263ec38e9b96ebdcf1d4abbbc1f92f947d5cf05961f465f496
FRICTION=0.30; ITERS=300; PPO_SEED=1000; RUN=4002
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT/eval" || exit 1
status() { echo "$(date -u +%FT%TZ) [a4-d7] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }
gpu_wait() { until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do
  status "waiting for $(( $1 / 1024 )) GiB GPU before $2"; sleep 300; done; }

# ---- 0. D6 must have passed; its model hash comes from the verdict line
VLINE=$(grep "VERDICT D6:" "$D6/queue.log" | tail -1)          # the LAST verdict line counts (bug 15r)
[[ "$VLINE" == *"VERDICT D6: PASS"* ]] || fail "D6's last verdict is not PASS in $D6/queue.log: D7 does not run"
WM_SHA=$(echo "$VLINE" | grep -oE "model [0-9a-f]{64}" | awk '{print $2}')
[ -n "$WM_SHA" ] || fail "no model hash on D6's verdict line"
[ "$(sha256sum "$WM" | cut -d' ' -f1)" = "$WM_SHA" ] || fail "D6 model hash differs from its verdict line"
for f in "$E16" "$ROUND0" "$START/start_data.json" "$RESETS" "$D6/data/manifest.json"; do [ -s "$f" ] || fail "missing $f"; done
[ "$(sha256sum $AN/go2_d3_gate.py | cut -d' ' -f1)" = "$GATE_SHA" ] || fail "the D3 gate code changed"
Q=$(CUDA_VISIBLE_DEVICES="" $PY -c "import json, sys; print(json.load(open(sys.argv[1]))['startup_reset_share'])" "$D6/data/manifest.json" | tail -1)
[[ "$Q" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || fail "invalid startup share from the D6 manifest: $Q"
RT=$(CUDA_VISIBLE_DEVICES="" $PY -c "import json, sys; print('RT', json.load(open(sys.argv[1])).get('reset_torque', 'recorded'))" "$D6/data/manifest.json" | grep '^RT ' | tail -1)
[[ "$RT" =~ ^RT\ (recorded|zero)$ ]] || fail "invalid reset-row torque in the D6 manifest: $RT"
status "D6 data: reset-row torque ${RT#RT }, startup share $Q"
CK=$(CUDA_VISIBLE_DEVICES="" "$PY" -c "import json, sys, numpy as np; print(json.loads(str(np.load(sys.argv[1])['metadata']))['checkpoint'])" "$ROUND0" 2>/dev/null | tail -1) \
  || fail "could not read s2 checkpoint from the round-0 trace"
[ -s "$CK" ] || fail "s2 checkpoint not found: $CK"
S2_SHA=$(sha256sum "$CK" | cut -d' ' -f1)
START_ITER=$(CUDA_VISIBLE_DEVICES="" "$PY" - "$ROUND0" "$CK" "$S2_SHA" <<'CHECK_S2' | tail -1
import json
import sys
import numpy as np
import torch

trace, checkpoint, sha = sys.argv[1:]
with np.load(trace, allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata"]))
if metadata["checkpoint_sha256"] != sha:
    raise SystemExit("s2 checkpoint hash differs from the round-0 trace")
print(int(torch.load(checkpoint, map_location="cpu", weights_only=False)["iter"]))
CHECK_S2
) || fail "could not verify s2 hash and starting iteration"
[[ "$START_ITER" =~ ^[0-9]+$ ]] || fail "invalid s2 iteration: $START_ITER"

{ git rev-parse HEAD; git status --short | grep -v '^??'; sha256sum $AN/go2_d3_gate.py $AN/go2_startup_hazard.py $MB/envs/base.py $MB/e1_train.py; } > "$OUT/git.txt"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_go2_d3_gate.py $AN/tests/test_go2_g0.py $AN/tests/test_go2_startup_resets.py \
  $AN/tests/test_go2_startup_hazard.py $AN/tests/test_go2_d3_fall_timing.py $AN/tests/test_warm_resets.py > "$OUT/tests.log" 2>&1 || fail "tests (tests.log)"
grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log)"
status "tests: $(tail -1 "$OUT/tests.log"); D6 model ${WM_SHA:0:12}; startup share $Q; s2 $CK (${S2_SHA:0:12}), saved iteration $START_ITER"

# ---- 1. D3's recipe in the D6 model
gpu_wait 12288 "training"
status "train: run $RUN, paper_warm + startup share $Q, $ITERS iterations from s2, PPO seed $PPO_SEED"
CUDA_VISIBLE_DEVICES=0 $PY $MB/e1_train.py --wm-checkpoint "$WM" --penalty-weight -1.0 --reward-contract s2 \
  --asset-trace "$E16" --start-data "$START" --init-policy-checkpoint "$CK" --max-iterations $ITERS \
  --startup-resets "$RESETS" --startup-share "$Q" paper_warm $RUN go2_flat --seed $PPO_SEED > "$OUT/train.log" 2>&1 \
  || fail "training (train.log)"
for pat in "\[go2\] reward contract s2: 16 terms" "\[go2\] start data" "\[dev\] max_iterations = $ITERS" "warm_start_resets= True" \
           "\[d6\] startup resets: share $Q" "Mean episode startup_reset_fraction"; do
  grep -q "$pat" "$OUT/train.log" || fail "train.log lacks '$pat'"
done
P=$(CUDA_VISIBLE_DEVICES="" "$PY" - "$OUT/train.log" "$RUN" "$PPO_SEED" "$START_ITER" "$ITERS" "$WM_SHA" "$S2_SHA" <<'CHECK_TRAINING' | tail -1
import json
import re
import sys
from pathlib import Path

path, run, seed, start, length, wm_sha, s2_sha = sys.argv[1:]
start, length = int(start), int(length)
log = Path(path).read_text()
end = start + length
iterations = [(int(i), int(total)) for i, total in re.findall(r"Learning iteration (\d+)/(\d+)", log)]
if iterations != [(i, end) for i in range(start, end)]:
    raise SystemExit(f"expected {length} PPO iterations from {start} to {end - 1}")
completed = re.findall(r"^Training completed\. Policy saved to (logs/offline/go2_flat/[^/]+)\.$", log, re.M)
if len(completed) != 1 or not completed[0].endswith(f"_{run}"):
    raise SystemExit(f"missing unique completion record for run {run}")
run_dir = Path(completed[0])
args = json.loads((run_dir / "run_args.json").read_text())
if (str(args["run_num"]), str(args["seed"]), args["task"], args["mode"],
        args["world_model_checkpoint"]["sha256"], args["initial_policy_checkpoint"]["sha256"]) != (
        run, seed, "go2_flat", "paper_warm", wm_sha, s2_sha):
    raise SystemExit("run provenance differs from the D7 configuration")
print(run_dir / f"policy_{end - 1}.pt")
CHECK_TRAINING
) || fail "could not verify completed D7 training"
[ -s "$P" ] || fail "missing final trained policy $P"
status "verified $ITERS PPO iterations starting at $START_ITER; $P $(sha256sum "$P" | cut -c1-12)"
grep -E "Mean reward|Mean episode length|Mean episode startup_reset_fraction" "$OUT/train.log" | tail -3 | tee -a "$OUT/queue.log"

# ---- 2. physics evaluation under the friction shift: D7 and unchanged s2
for ARM in "D7:$P" "S2:$CK"; do
  TAG=${ARM%%:*}; CKPT=${ARM#*:}
  for S in 42 43; do
    DIR="$OUT/eval/${TAG}_cmd10_s$S"; mkdir -p "$DIR"
    gpu_wait 8192 "eval $TAG s$S"
    status "eval $TAG seed $S (friction x$FRICTION)"
    CUDA_VISIBLE_DEVICES=0 $PY $EVAL --task $TASK --checkpoint "$CKPT" --num_envs 32 --num_steps 10000 --seed $S --std .5 \
      --cmd_resample_s 10 --friction-scale $FRICTION --headless --out_csv "$DIR/episodes.csv" > "$DIR/eval.log" 2>&1 \
      || fail "eval $TAG s$S ($DIR/eval.log)"
    grep -q "\[friction\] verified in the simulator" "$DIR/eval.log" || fail "no friction read-back in $DIR/eval.log"
    [ -s "$DIR/episodes_summary.json" ] || fail "no summary for $TAG s$S"
  done
done

# ---- 3. D3's unchanged gate on a view that gives D7's folders the gate's policy tag; startup risk separately
mkdir -p "$OUT/gate_view"
for S in 42 43; do
  ln -s "../eval/D7_cmd10_s$S" "$OUT/gate_view/D3_cmd10_s$S"
  ln -s "../eval/S2_cmd10_s$S" "$OUT/gate_view/S2_cmd10_s$S"
done
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_d3_gate.py --eval "$OUT/gate_view" --friction $FRICTION --out "$OUT/d7_gate.json" \
  > "$OUT/gate.log" 2>&1 || { cat "$OUT/gate.log"; fail "gate scoring (gate.log)"; }
sed 's/D3/D7/g' "$OUT/gate.log" | tee -a "$OUT/queue.log"
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_startup_hazard.py --eval "$OUT/eval" --tag D7 --ref S2 --out "$OUT/hazard" > "$OUT/hazard.log" 2>&1 \
  || { tail -5 "$OUT/hazard.log"; fail "startup hazard (hazard.log)"; }
cat "$OUT/hazard/startup_hazard.txt" | tee -a "$OUT/queue.log"
CUDA_VISIBLE_DEVICES="" $PY $AN/go2_d3_fall_timing.py --eval "$OUT/gate_view" --out "$OUT/timing" > "$OUT/timing.log" 2>&1 \
  || { tail -5 "$OUT/timing.log"; fail "fall timing (timing.log)"; }
sed 's/^D3/D7/; s/D3 - S2/D7 - S2/' "$OUT/timing/d3_fall_timing.txt" | tee -a "$OUT/queue.log"
status "done: $(grep VERDICT "$OUT/gate.log"); $(sha256sum "$OUT/d7_gate.json")"
