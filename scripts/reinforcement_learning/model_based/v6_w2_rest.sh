#!/bin/bash
# Plan v6, the rest of W1-W3 analysis (18 Nov): closed-loop C0, C10, the F1 screen and the E16 reward check.
#   L1  C0 closed loop: ETH model, the three C0 sources (paths and SHA-256 from the 18 Nov C0 report), the
#       27 Oct reproduction gate, each source's own policy in the loop (checkpoint and hash from the trace
#       metadata; the policy must reproduce the recorded actions first); then a table vs the recorded-action
#       run of rerun 2 (closure < 0 = the closed loop heals more).
#   L2, L3  C10 (descriptive): 8 policies, imagined terminations per 1000 imagined s (CPU rebuild of each
#       run's own imagination) minus real falls per 1000 s (analysis/c10_real_falls.json).
#   L4  F1 screen (plan §3) on the rerun-2 runs (primary = D's model, E1-V refit; ETH reported), then E16:
#       a short s2 recording in Go2-Start-Unitree-Check-v0 (GPU, 32 envs x 1000 steps; waits for 6 GiB) and
#       the term-by-term check. E16_TRACE=<npz> skips the recording and checks that trace.
# Lanes are independent: a failed lane is reported and the others finish. Exit code 1 if any lane failed.
# Run from the robotic_world_model root in the container:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_w2_rest.sh <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
set -uo pipefail
OUT=${1:?fresh output dir}
[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
PY=${PY:-/isaac-sim/python.sh}
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
ETH=assets/models/pretrain_rnn_ens.pt
RERUN2=${RERUN2:-logs/p2/v6w1_rerun2_1118}
C0OLD=${C0OLD:-logs/p2/c0_1118/c0_report.json}
REPRO_TRACE=${REPRO_TRACE:-logs/p2/transition_fidelity.ELf50z/baseline.npz}
REPRO_JSON=${REPRO_JSON:-logs/p2/paired_fall_replay.cqaEuU/paired_fall_replay.json}
OFFLINE=${OFFLINE:-logs/offline/anymal_d_flat}
ROUND0_SIDECAR=${ROUND0_SIDECAR:-logs/p2/go2_round0_1118/round0_s2_seed101.json}
C10_ENVS=${C10_ENVS:-640}
C10_STEPS=${C10_STEPS:-1000}
E16_TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Start-Unitree-Check-v0
mkdir -p "$OUT"
status() { echo "$(date -u +%FT%TZ) [${LANE:-main}] $*" | tee -a "$OUT/queue.log"; }
fail() { status "FAILED: $*"; exit 1; }

# ---- 0. inputs, provenance, tests
for f in "$ETH" "$C0OLD" "$REPRO_TRACE" "$REPRO_JSON" "$RERUN2/control/c0_windows.npz" "$RERUN2/eth/c0_windows.npz" \
         "$RERUN2/c0_eth/c0_windows.npz" "$AN/c10_real_falls.json"; do
  [ -s "$f" ] || fail "missing $f"
done
$PY -c '
import json, sys
for name, v in json.load(open(sys.argv[1]))["meta"]["inputs"].items():
    print("C0TRACE", name, v["path"], v["sha256"])
' "$C0OLD" | grep '^C0TRACE ' > "$OUT/c0_traces.txt" || fail "cannot read the C0 inputs from $C0OLD"
C0TRACES=()
while read -r _ name path sha; do
  [ -s "$path" ] || fail "C0 trace $name missing: $path"
  [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$sha" ] || fail "C0 trace $name changed since 18 Nov: $path"
  C0TRACES+=(--trace "$name=$path")
done < "$OUT/c0_traces.txt"
[ ${#C0TRACES[@]} -eq 6 ] || fail "expected 3 C0 sources in $C0OLD"
C10_ARGS=()
for spec in 1601:1601 1701_B1:1701 1901:1901 2102_B2:2102 2103_B3:2103 2004_D1:2004 2402_D2:2402 2403_D3:2403; do
  name=${spec%%:*}; run=${spec##*:}
  dirs=( "$OFFLINE"/*_"$run" )
  [ ${#dirs[@]} -eq 1 ] && [ -s "${dirs[0]}/run_args.json" ] && [ -s "${dirs[0]}/policy_2499.pt" ] \
    || fail "C10 run $run: need exactly one $OFFLINE/*_$run with run_args.json and policy_2499.pt (found: ${dirs[*]})"
  C10_ARGS+=("$name=${dirs[0]}")
done
if [ -z "${E16_TRACE:-}" ]; then
  [ -s "$ROUND0_SIDECAR" ] || fail "missing $ROUND0_SIDECAR (s2 checkpoint path); or set E16_TRACE"
  S2=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$ROUND0_SIDECAR" | tail -1)
  [ -s "$S2" ] || fail "s2 checkpoint not found: $S2"
fi
{ git rev-parse HEAD; git status --short | grep -v '^??' ; } > "$OUT/git.txt"
sha256sum "$ETH" "$REPRO_TRACE" "$REPRO_JSON" "$AN/c10_real_falls.json" "$AN"/c0_battery.py "$AN"/policy_loop.py \
  "$AN"/c10_optimism_gap.py "$AN"/f1_screen.py "$AN"/e16_reward_check.py "$MB"/envs/go2_flat.py "$MB"/configs/go2_flat_cfg.py \
  "$0" ${S2:+"$S2"} > "$OUT/inputs.sha256"
printf '%s\n' "${C10_ARGS[@]}" > "$OUT/c10_runs.txt"
$PY -m pytest -q $AN/tests/test_c0_battery.py $AN/tests/test_c0_compare.py $AN/tests/test_c10.py $AN/tests/test_e16.py \
  $AN/tests/test_f1_screen.py > "$OUT/tests.log" 2>&1 || fail "CPU tests failed (tests.log)"
status "tests passed"

# ---- lanes
( LANE=L1; export CUDA_VISIBLE_DEVICES=""
  status "C0 closed loop (ETH, 3 sources, own policies)"
  $PY $AN/c0_battery.py --wm "$ETH" "${C0TRACES[@]}" --repro-trace "$REPRO_TRACE" --repro-json "$REPRO_JSON" \
    --actions policy --out "$OUT/c0_closed_eth" > "$OUT/c0_closed_eth.log" 2>&1 \
    || { status "FAILED C0 closed loop ($(tail -1 "$OUT/c0_closed_eth.log"))"; exit 1; }
  $PY $AN/c0_compare.py --control "recorded=$RERUN2/c0_eth" --treat "closed_loop=$OUT/c0_closed_eth" \
    --sources batchA,batchC,round0_1701 --out "$OUT/table_closed" > "$OUT/table_closed.log" 2>&1 \
    || { status "FAILED closed-loop table"; exit 1; }
  status "done C0 closed loop" ) & P1=$!
( LANE=L2; export CUDA_VISIBLE_DEVICES=""; status "C10 part 1"
  $PY $AN/c10_optimism_gap.py $(printf -- '--policy %s ' "${C10_ARGS[@]:0:4}") --real $AN/c10_real_falls.json \
    --num-envs "$C10_ENVS" --steps "$C10_STEPS" --out "$OUT/c10_part1" > "$OUT/c10_part1.log" 2>&1 \
    || { status "FAILED C10 part 1 ($(tail -1 "$OUT/c10_part1.log"))"; exit 1; }
  status "done C10 part 1" ) & P2=$!
( LANE=L3; export CUDA_VISIBLE_DEVICES=""; status "C10 part 2"
  $PY $AN/c10_optimism_gap.py $(printf -- '--policy %s ' "${C10_ARGS[@]:4:4}") --real $AN/c10_real_falls.json \
    --num-envs "$C10_ENVS" --steps "$C10_STEPS" --out "$OUT/c10_part2" > "$OUT/c10_part2.log" 2>&1 \
    || { status "FAILED C10 part 2 ($(tail -1 "$OUT/c10_part2.log"))"; exit 1; }
  status "done C10 part 2" ) & P3=$!
( LANE=L4; RC=0
  status "F1 screen"
  CUDA_VISIBLE_DEVICES="" $PY $AN/f1_screen.py --run "D_model=$RERUN2/control" --run "ETH=$RERUN2/eth" --primary D_model \
    --out "$OUT/f1_screen" > "$OUT/f1_screen.log" 2>&1 || { status "FAILED F1 screen"; RC=1; }
  TRACE=${E16_TRACE:-}
  if [ -z "$TRACE" ]; then
    export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
    export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
    until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 6144 ]; do
      status "waiting for 6 GiB GPU (E16 recording)"; sleep 300; done
    status "E16 recording: s2 in $E16_TASK, 32 envs x 1000 steps, seed 7"
    TRACE="$OUT/e16_s2_check.npz"
    CUDA_VISIBLE_DEVICES=0 $PY scripts/reinforcement_learning/rsl_rl/record_pre_reset_transitions.py --task $E16_TASK \
      --checkpoint "$S2" --num_envs 32 --num_steps 1000 --seed 7 --cmd_resample_s 10 --headless --out "$TRACE" \
      > "$OUT/e16_record.log" 2>&1 || { status "FAILED E16 recording (e16_record.log)"; exit 1; }
  fi
  status "E16 check on $TRACE"
  CUDA_VISIBLE_DEVICES="" $PY $AN/e16_reward_check.py --trace "$TRACE" --out "$OUT/e16" > "$OUT/e16.log" 2>&1 \
    || { status "FAILED E16 check ($(tail -1 "$OUT/e16.log"))"; exit 1; }
  status "done F1 screen + E16"; exit $RC ) & P4=$!
RC=0; for P in $P1 $P2 $P3 $P4; do wait $P || RC=1; done
status "done (lanes failed: $RC)"

# ---- summary
for f in c0_closed_eth/c0_report.txt; do [ -s "$OUT/$f" ] && { echo "=== C0 closed loop (head)"; awk '/^== /{n++} n<=1' "$OUT/$f" | head -30;
  grep -A8 "^Closed loop" "$OUT/$f"; grep -E "^== |C0 rule" "$OUT/$f"; }; done
[ -s "$OUT/table_closed/compare.txt" ] && { echo "=== closed loop vs recorded actions"; cat "$OUT/table_closed/compare.txt"; }
for f in c10_part1/c10_report.txt c10_part2/c10_report.txt f1_screen/f1_screen.txt e16/e16_report.txt; do
  [ -s "$OUT/$f" ] && { echo "=== $f"; cat "$OUT/$f"; }; done
exit $RC
