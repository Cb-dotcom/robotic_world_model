#!/bin/bash
# A4 E1-R2 (rules fixed 23 Nov 2026 before any run, A4 §1 and §3): E1 on ANYmal with a FRESH data collection (data
# seed 102) and a FRESH model fit (fit seed 2); the E1 procedure otherwise unchanged (e1v_queue.sh / e1r_queue.sh).
# Arms, all lambda 1, paper recipe, PPO seeds 1, 2, 3:
#   B' frozen ETH model    runs 2701, 2702, 2703   (need no new data: they start at once)
#   D' fresh refit model   runs 2801, 2802, 2803   (after record -> prepare -> refit -> model check)
# Main lane: record 1701 + reference (pre-reset recorder, 64 x 5000, seed 102, commands 10 s, pushes off) -> prepare
# (envs 52-63 held out) -> refit with per-head reset (ETH warm start, 500 it, lr 1e-4, fit seed 2, pos-weight 1, ETH
# normalizer; b3 provenance checked) -> model check h1/8/24 with E1's bars (the queue stops on a miss) -> D' lanes.
# Every training is checked against 1701's run_args.json (e1v_match.py --expect-seed) as soon as it starts.
# Evaluations as E1: eval_episode_stats.py, 32 x 10000, std .5, pushes on, seeds 42/43, commands 30 s and 10 s.
# Readout: analysis/e1r2_score.py (descriptive reading fixed in its docstring).
# Before any GPU work, the reader is checked on E1-R's saved cells against the fall counts in the run-log (14 Nov).
# Outcomes are read from output files, not from python.sh exit codes (bug 15q).
# Phases, from the robotic_world_model root in the container:
#   all    preflight + tests + crosscheck, then all lanes, evaluations and the readout
#   score  the readout again from the saved evaluations of an existing OUT (CPU)
# Example:
#   setsid nohup scripts/reinforcement_learning/model_based/v6_dev_e1r2.sh all logs/p2/e1r2_1123 > /tmp/e1r2.out 2>&1 < /dev/null &
set -uo pipefail
PHASE=${1:?phase: all | score}
OUT=${2:?output dir}
DATA_SEED=102; FIT_SEED=2; FT_ITERS=500; FT_LR=1e-4; PEN=-1.0
SEEDS=(1 2 3); B_RUNS=(2701 2702 2703); D_RUNS=(2801 2802 2803)
MB=scripts/reinforcement_learning/model_based
AN=$MB/analysis
EVAL=scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py
RECORDER=scripts/reinforcement_learning/rsl_rl/record_pre_reset_transitions.py
TASK=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
LOGS=logs/offline/anymal_d_flat
ETH=assets/models/pretrain_rnn_ens.pt
ETH_SHA=2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a
CTRL_DIR=$LOGS/2026-10-21_13-59-55_1701
CTRL=$CTRL_DIR/policy_2499.pt
E1R=${E1R:-logs/p2/e1r_1113_1853}
PY=${PY:-/isaac-sim/python.sh}
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
LANE=main
status() { echo "$(date -u +%FT%TZ) [e1r2 $LANE] $*" | tee "$OUT/status_$LANE.txt" | tee -a "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
acquire_gpu() {   # one GPU start at a time; the lock is released 3 min after the start
  until mkdir "$OUT/gpu.lock" 2>/dev/null; do sleep 20; done
  until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12288 ]; do
    status "waiting for 12 GiB GPU before $1"; sleep 300; done; }
release_gpu_later() { ( sleep 180; rmdir "$OUT/gpu.lock" 2>/dev/null ) & }

score() {
  local ARGS=() k
  for k in 0 1 2; do ARGS+=(--arm "B${SEEDS[$k]}=$OUT/eval:B${SEEDS[$k]}" --arm "D${SEEDS[$k]}=$OUT/eval:D${SEEDS[$k]}"); done
  rm -rf "$OUT/score.tmp"
  CUDA_VISIBLE_DEVICES="" $PY $AN/e1r2_score.py "${ARGS[@]}" --out "$OUT/score.tmp" > "$OUT/score.log" 2>&1
  [ -s "$OUT/score.tmp/e1r2_score.txt" ] || { tail -5 "$OUT/score.log"; fail "the readout wrote no result (score.log)"; }
  rm -rf "$OUT/score"; mv "$OUT/score.tmp" "$OUT/score"
  tee -a "$OUT/queue.log" < "$OUT/score/e1r2_score.txt"
  status "done: $(grep '^READING' "$OUT/score/e1r2_score.txt"); $(sha256sum "$OUT/score/e1r2_score.json" | cut -c1-12)"
}

if [ "$PHASE" = score ]; then
  [ -d "$OUT/eval" ] || { echo "no evaluations in $OUT"; exit 2; }
  score; exit 0
fi
[ "$PHASE" = all ] || { echo "phase must be all or score"; exit 2; }
[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT" || exit 1

# ---- 0. preflight, tests and the reader's known-answer check (CPU)
status "preflight"
git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
git merge-base --is-ancestor d9d2a66 HEAD || fail "HEAD lacks the acceleration-reward fix d9d2a66"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; } > "$OUT/git.txt"
[ "$(sha256sum "$ETH" | cut -d' ' -f1)" = "$ETH_SHA" ] || fail "ETH checkpoint hash differs"
[ -s "$CTRL_DIR/run_args.json" ] || fail "no run_args.json for 1701 ($CTRL_DIR)"
cp "$CTRL_DIR/run_args.json" "$OUT/control_run_args.json"
grep -q "policy hash MATCHES" logs/p2/e1A_1111_1650/eval_1701/hash_check.txt 2>/dev/null \
  || fail "1701 policy hash not confirmed (logs/p2/e1A_1111_1650/eval_1701/hash_check.txt)"
for R in "${B_RUNS[@]}" "${D_RUNS[@]}"; do ls -d $LOGS/*_$R >/dev/null 2>&1 && fail "run $R already exists in $LOGS"; done
sha256sum "$ETH" "$CTRL" $MB/e1_train.py $MB/e1_prepare_data.py $MB/e1_model_check.py $MB/e1v_match.py \
  $MB/refit/fit_reset_per_head.py $MB/p15_train.py $MB/train.py $MB/policy_training.py $AN/e1r2_score.py \
  $MB/v6_dev_e1r2.sh $EVAL $RECORDER > "$OUT/inputs.sha256"
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q $AN/tests/test_e1r2_score.py $AN/tests/test_e1v_match.py \
  $AN/tests/test_e1_prepare_data.py $AN/tests/test_fall_rate.py $AN/tests/test_policy_training_log.py \
  > "$OUT/tests.log" 2>&1 || fail "tests (tests.log)"
grep -qE "[0-9]+ skipped" "$OUT/tests.log" && fail "tests were skipped (tests.log)"
status "tests: $(tail -1 "$OUT/tests.log")"
# E1-R's saved cells (run-log 14 Nov: falls / episodes, seeds 42 + 43 pooled)
CUDA_VISIBLE_DEVICES="" $PY $AN/e1r2_score.py --arm "B2=$E1R/eval:B2" --arm "B3=$E1R/eval:B3" --arm "D2=$E1R/eval:D2" \
  --arm "D3=$E1R/eval:D3" --expect B2:30=65 --expect B3:30=26 --expect D2:30=32 --expect D3:30=21 --expect B2:10=238 \
  --expect B3:10=157 --expect D2:10=38 --expect D3:10=41 --out "$OUT/crosscheck" > "$OUT/crosscheck.log" 2>&1
grep -q "^CROSSCHECK: PASS" "$OUT/crosscheck/e1r2_score.txt" 2>/dev/null \
  || { tail -12 "$OUT/crosscheck.log"; fail "the reader does not reproduce E1-R's fall counts (crosscheck.log)"; }
status "crosscheck on E1-R's saved cells: PASS"

train_arm() {   # RUN WM SEED TAG
  local RUN=$1 WM=$2 SEED=$3 TAG=$4 D="" PID i
  ls -d $LOGS/*_$RUN >/dev/null 2>&1 && fail "$TAG: run $RUN already exists"
  acquire_gpu "train $TAG"; status "training $TAG: run $RUN, PPO seed $SEED, penalty weight $PEN, WM $WM"
  CUDA_VISIBLE_DEVICES=0 setsid $PY $MB/e1_train.py --wm-checkpoint "$WM" --penalty-weight "$PEN" \
    paper $RUN anymal_d_flat --seed $SEED > "$OUT/train_$TAG.log" 2>&1 &
  PID=$!; release_gpu_later
  for i in $(seq 1 240); do
    D=$(ls -d $LOGS/*_$RUN 2>/dev/null | tail -1)
    [ -n "$D" ] && [ -s "$D/run_args.json" ] && break
    kill -0 $PID 2>/dev/null || break; sleep 15; done
  if [ -z "$D" ] || [ ! -s "$D/run_args.json" ]; then
    kill -TERM -- -$PID 2>/dev/null; fail "$TAG: training wrote no run_args.json (train_$TAG.log)"; fi
  $PY $MB/e1v_match.py --control "$OUT/control_run_args.json" --candidate "$D/run_args.json" \
    --expect-wm-sha "$(sha256sum "$WM" | cut -d' ' -f1)" --expect-penalty "$PEN" --expect-seed "$SEED" \
    --control-fix-note "966d5bb contains d9d2a66 (checked 13 Nov)" > "$OUT/match_$TAG.txt" 2>&1
  if ! grep -q "^MATCH: True" "$OUT/match_$TAG.txt"; then
    kill -TERM -- -$PID 2>/dev/null; wait $PID 2>/dev/null; fail "$TAG: matching check failed (match_$TAG.txt)"; fi
  status "$TAG: matching check passed; training continues ($D)"
  wait $PID
  [ -s "$D/policy_2499.pt" ] || fail "$TAG: no policy_2499.pt in $D (train_$TAG.log)"
  echo "$D/policy_2499.pt" > "$OUT/policy_$TAG.txt"
}

evaluate() {    # TAG CHECKPOINT
  local TAG=$1 CK=$2 CMD S DIR
  for CMD in 30 10; do for S in 42 43; do
    DIR="$OUT/eval/${TAG}_cmd${CMD}_s${S}"; mkdir -p "$DIR"
    acquire_gpu "eval $TAG cmd$CMD s$S"; status "eval $TAG cmd$CMD seed $S"; release_gpu_later
    CUDA_VISIBLE_DEVICES=0 $PY $EVAL --task $TASK --checkpoint "$CK" --num_envs 32 --num_steps 10000 \
      --seed $S --std .5 --cmd_resample_s $CMD --headless --out_csv "$DIR/episodes.csv" > "$DIR/eval.log" 2>&1
    [ -s "$DIR/episodes_summary.json" ] || fail "eval $TAG cmd$CMD s$S wrote no summary ($DIR/eval.log)"
  done; done
}

lane() {        # TAG RUN WM SEED
  LANE=$1; train_arm "$2" "$3" "$4" "$1"; evaluate "$1" "$(cat "$OUT/policy_$1.txt")"; status "done"
}

# ---- 1. B' lanes: frozen ETH model (no new data needed)
PIDS=()
for k in 0 1 2; do
  ( lane "B${SEEDS[$k]}" "${B_RUNS[$k]}" "$ETH" "${SEEDS[$k]}" ) & PIDS+=($!); sleep 30
done

# ---- 2. main lane: fresh round-0 data, fresh fit, model check
mkdir -p "$OUT/record"
for PAIR in baseline1701:$CTRL reference:$ETH; do
  NAME=${PAIR%%:*}; CK=${PAIR#*:}
  acquire_gpu "record $NAME"; status "recording $NAME (64 x 5000, seed $DATA_SEED, commands 10 s, pushes off)"; release_gpu_later
  CUDA_VISIBLE_DEVICES=0 $PY $RECORDER --task $TASK --checkpoint "$CK" --num_envs 64 --num_steps 5000 \
    --seed $DATA_SEED --cmd_resample_s 10 --headless --out "$OUT/record/$NAME.npz" > "$OUT/record/$NAME.log" 2>&1
  [ -s "$OUT/record/$NAME.npz" ] || fail "recording $NAME wrote no trace (record/$NAME.log)"
done
status "preparing data (envs 52-63 held out)"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1_prepare_data.py --baseline "$OUT/record/baseline1701.npz" \
  --reference "$OUT/record/reference.npz" --baseline-tag baseline1701 --out "$OUT/data" > "$OUT/prepare.log" 2>&1
[ -s "$OUT/data/summary.json" ] && [ -s "$OUT/data/segments/manifest.csv" ] || fail "prepare data (prepare.log)"
acquire_gpu "refit"; status "refit with per-head reset: ETH warm start, $FT_ITERS it, lr $FT_LR, fit seed $FIT_SEED"; release_gpu_later
CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config anymal_d_flat \
  --data "$OUT/data/segments" --output "$OUT/wm/e1r2_wm.pt" --reference_wm "$ETH" --warm_start \
  --norm_stats "$OUT/data/eth_norm.npz" --seed $FIT_SEED --iterations $FT_ITERS --lr $FT_LR \
  --termination_pos_weight 1 --lazy_windows --save_interval 100 --device cuda:0 > "$OUT/refit.log" 2>&1
WM_NEW="$OUT/wm/e1r2_wm.pt"
[ -s "$WM_NEW" ] || fail "no refit checkpoint (refit.log)"
CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
print('B3', ck.get('provenance', {}).get('b3_reset_per_head') is True)" "$WM_NEW" > "$OUT/provenance.txt" 2>&1
grep -q "^B3 True" "$OUT/provenance.txt" || fail "refit checkpoint lacks provenance b3_reset_per_head=True (provenance.txt)"
status "refit model $(sha256sum "$WM_NEW" | cut -d' ' -f1)"
status "model check (horizons 1/8/24; E1 bars)"
CUDA_VISIBLE_DEVICES="" $PY $MB/e1_model_check.py --eth "$ETH" --finetuned "$WM_NEW" \
  --baseline-test "$OUT/data/test_baseline1701.npz" --reference-test "$OUT/data/test_reference.npz" \
  --baseline-tag baseline1701 --horizons 1,8,24 --out "$OUT/model_check" > "$OUT/model_check.log" 2>&1
[ -s "$OUT/model_check/report.txt" ] || fail "model check wrote no report (model_check.log)"
grep -E "^(A|B)|MODEL CHECK PASS" "$OUT/model_check/report.txt" | tee -a "$OUT/queue.log" > /dev/null
grep -q "MODEL CHECK PASS: True" "$OUT/model_check/report.txt" || fail "model check bars not met (model_check/report.txt): D' lanes do not run"

# ---- 3. D' lanes: fresh refit model
for k in 0 1 2; do
  ( lane "D${SEEDS[$k]}" "${D_RUNS[$k]}" "$WM_NEW" "${SEEDS[$k]}" ) & PIDS+=($!); sleep 30
done
RC=0; for P in "${PIDS[@]}"; do wait "$P" || RC=1; done
LANE=main
for T in B1 B2 B3 D1 D2 D3; do grep -q "\] done" "$OUT/status_$T.txt" 2>/dev/null || RC=1; done
[ $RC -eq 0 ] || fail "a lane did not finish (status_*.txt)"

# ---- 4. readout
score
