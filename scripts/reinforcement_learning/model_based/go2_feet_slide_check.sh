#!/bin/bash
# Queue the Go2 feet_slide simulator check. Run from the robotic_world_model root in the container:
#   nohup scripts/reinforcement_learning/model_based/go2_feet_slide_check.sh <out_dir> > <out_dir>.nohup 2>&1 &
# Waits for 6 GiB GPU (32 envs, 1000 steps; a few minutes), then writes <out_dir>/report.json.
set -uo pipefail
OUT=${1:?fresh output dir}
CK=logs/rsl_rl/unitree_go2_flat/2026-06-12_09-07-38_pretrain/model_2000.pt   # accepted Go2 walker (4C)
[ -e "$OUT" ] && { echo "OUT exists"; exit 2; }
mkdir -p "$OUT"; [ -s "$CK" ] || { echo "missing $CK" > "$OUT/status.txt"; exit 1; }
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
git rev-parse HEAD > "$OUT/commit.txt"
until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 6144 ]; do
  echo "$(date -u +%FT%TZ) waiting for 6 GiB" > "$OUT/status.txt"; sleep 300; done
echo "$(date -u +%FT%TZ) running" > "$OUT/status.txt"
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/validate_go2_feet_slide.py --checkpoint "$CK" \
  --num_envs 32 --num_steps 1000 --seed 7 --headless --out "$OUT/report.json" > "$OUT/check.log" 2>&1
echo "$(date -u +%FT%TZ) done rc=$?" > "$OUT/status.txt"
