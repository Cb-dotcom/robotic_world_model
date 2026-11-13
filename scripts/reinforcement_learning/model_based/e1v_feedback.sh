#!/bin/bash
# E1 validation feedback cycle (plan v5.2 §5, step after e1v_queue.sh). Run from the robotic_world_model root:
#   setsid nohup scripts/reinforcement_learning/model_based/e1v_feedback.sh <e1v_out_dir> <fresh_out_dir> > <out>.nohup 2>&1 < /dev/null &
# Waits until the e1v queue is done. For each arm A-D: deploy its policy in simulation and record a small batch (pre-reset recorder,
# 32 envs x 1000 steps, seed 202, pushes off, commands 10 s), then score the whole batch test-first
# on that arm's model (A, B: ETH; C, D: refit) at horizons 1/8/24.
# For C and D: one short update (100 it from the refit model, per-head reset, envs 24-31 held out),
# then re-score the held-out batch envs and the round-0 held-out tests (forgetting check).
# No new policy training. No bars: the cycle checks the feedback path and records the numbers.
set -uo pipefail
E1V=${1:?e1v_queue output dir}
OUT=${2:?fresh output dir}
MB=scripts/reinforcement_learning/model_based
RECORDER=scripts/reinforcement_learning/rsl_rl/record_pre_reset_transitions.py
TASK=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0
ETH=assets/models/pretrain_rnn_ens.pt
WM1="$E1V/wm/e1v_wm.pt"
CTRL=logs/offline/anymal_d_flat/2026-10-21_13-59-55_1701/policy_2499.pt
PY=/isaac-sim/python.sh

[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
status() { echo "$(date -u +%FT%TZ) $*" | tee "$OUT/status.txt" >> "$OUT/queue.log"; }
fail()   { status "FAILED: $*"; exit 1; }
wait_gpu() { until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12288 ]; do
  status "waiting for 12 GiB GPU before $1"; sleep 300; done; }

until grep -q "\] done" "$E1V/status_main.txt" 2>/dev/null; do
  grep -q "FAILED" "$E1V"/status_*.txt 2>/dev/null && fail "e1v queue failed ($E1V/status_*.txt)"
  status "waiting for the e1v queue to finish ($E1V)"; sleep 300; done
git diff --quiet HEAD -- scripts source || fail "uncommitted changes in scripts/ or source/"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; } > "$OUT/git.txt"
[ -s "$WM1" ] || fail "no refit model $WM1"
declare -A POLICY=([A]="$(cat "$E1V/policy_A.txt")" [B]="$CTRL" [C]="$(cat "$E1V/policy_C.txt")" [D]="$(cat "$E1V/policy_D.txt")")
declare -A MODEL=([A]="$ETH" [B]="$ETH" [C]="$WM1" [D]="$WM1")
sha256sum "$ETH" "$WM1" "${POLICY[@]}" $MB/e1v_*.py $MB/e1v_feedback.sh $MB/e1_prepare_data.py > "$OUT/inputs.sha256"

for ARM in A B C D; do
  wait_gpu "record $ARM"; status "recording arm $ARM (32 x 1000, seed 202)"
  CUDA_VISIBLE_DEVICES=0 $PY $RECORDER --task $TASK --checkpoint "${POLICY[$ARM]}" --num_envs 32 --num_steps 1000 \
    --seed 202 --cmd_resample_s 10 --headless --out "$OUT/batch_$ARM.npz" > "$OUT/record_$ARM.log" 2>&1 \
    || fail "recording arm $ARM"
  status "test-first score arm $ARM"
  CUDA_VISIBLE_DEVICES="" $PY $MB/e1v_score.py --model "arm$ARM=${MODEL[$ARM]}" --trace "batch$ARM=$OUT/batch_$ARM.npz" \
    --horizons 1,8,24 --out "$OUT/score_first_$ARM" > "$OUT/score_first_$ARM.log" 2>&1 || fail "score arm $ARM"
done

for ARM in C D; do
  status "short update arm $ARM: prepare"
  CUDA_VISIBLE_DEVICES="" $PY $MB/e1_prepare_data.py --baseline "$OUT/batch_$ARM.npz" --baseline-tag "batch$ARM" \
    --test-fraction 0.25 --out "$OUT/data_$ARM" > "$OUT/prepare_$ARM.log" 2>&1 || fail "prepare arm $ARM"
  wait_gpu "update $ARM"; status "short update arm $ARM: 100 it from the refit model, per-head reset"
  CUDA_VISIBLE_DEVICES=0 $PY $MB/refit/fit_reset_per_head.py --config anymal_d_flat \
    --data "$OUT/data_$ARM/segments" --output "$OUT/wm_$ARM/update.pt" --reference_wm "$WM1" --warm_start \
    --norm_stats "$OUT/data_$ARM/eth_norm.npz" --seed 1 --iterations 100 --lr 1e-4 \
    --termination_pos_weight 1 --lazy_windows --save_interval 50 --device cuda:0 > "$OUT/update_$ARM.log" 2>&1 \
    || fail "update arm $ARM"
  CUDA_VISIBLE_DEVICES="" $PY -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
sys.exit(0 if ck.get('provenance', {}).get('b3_reset_per_head') is True else 1)" "$OUT/wm_$ARM/update.pt" \
    || fail "update $ARM lacks provenance b3_reset_per_head=True"
  status "re-score arm $ARM"
  CUDA_VISIBLE_DEVICES="" $PY $MB/e1v_score.py --model "before=$WM1" --model "after=$OUT/wm_$ARM/update.pt" \
    --trace "heldout_batch$ARM=$OUT/data_$ARM/test_batch$ARM.npz" \
    --trace "round0_baseline1701=$E1V/data/test_baseline1701.npz" --trace "round0_reference=$E1V/data/test_reference.npz" \
    --horizons 1,8,24 --out "$OUT/score_update_$ARM" > "$OUT/score_update_$ARM.log" 2>&1 || fail "re-score arm $ARM"
done

{ for ARM in A B C D; do cat "$OUT/score_first_$ARM/scores.txt"; done
  for ARM in C D; do echo "#### update $ARM"; cat "$OUT/score_update_$ARM/scores.txt"; done; } > "$OUT/summary.txt"
status "done"
