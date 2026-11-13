#!/bin/bash
# Go2 pre-reset recorder smoke test (plan v5.1, needed for E2). Run from the robotic_world_model root:
#   setsid nohup scripts/reinforcement_learning/model_based/go2_record_smoke.sh <fresh_out_dir> > <out_dir>.nohup 2>&1 < /dev/null &
# Records the accepted Go2 walker (4C) for 16 envs x 1000 steps (20 s), pushes off, commands 10 s, then
# validates the trace on CPU (schema, terminal rows checked before reset, no unmarked discontinuity).
# Pass = recorder rc 0 and validate rc 0. Waits for 8 GiB GPU. A few minutes.
set -uo pipefail
OUT=${1:?fresh output dir}
CK=logs/rsl_rl/unitree_go2_flat/2026-06-12_09-07-38_pretrain/model_2000.pt   # accepted Go2 walker (4C)
TASK=Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0
MB=scripts/reinforcement_learning/model_based
PY=/isaac-sim/python.sh
[ -e "$OUT" ] && { echo "OUT exists"; exit 2; }
mkdir -p "$OUT"
status() { echo "$(date -u +%FT%TZ) $*" | tee "$OUT/status.txt" >> "$OUT/queue.log"; }
[ -s "$CK" ] || { status "FAILED: missing $CK"; exit 1; }
export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y WANDB_MODE=offline
export PYTHONPATH="/isaac-sim/kit/extscore/omni.client.lib:$PWD/../rsl_rl_rwm"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; } > "$OUT/git.txt"
until [ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 8192 ]; do
  status "waiting for 8 GiB GPU"; sleep 300; done
status "recording"
CUDA_VISIBLE_DEVICES=0 $PY scripts/reinforcement_learning/rsl_rl/record_pre_reset_transitions.py --task $TASK \
  --checkpoint "$CK" --num_envs 16 --num_steps 1000 --seed 101 --cmd_resample_s 10 --headless \
  --out "$OUT/go2_trace.npz" > "$OUT/record.log" 2>&1
RC=$?; echo "record rc=$RC" > "$OUT/rc.txt"
[ $RC -eq 0 ] || { status "FAILED: recorder rc=$RC (see record.log)"; exit 1; }
status "validating"
CUDA_VISIBLE_DEVICES="" $PY - "$OUT/go2_trace.npz" > "$OUT/validate.log" 2>&1 <<'PYEOF'
import json, sys
import numpy as np
sys.path.insert(0, "scripts/reinforcement_learning/model_based/analysis")
from pre_reset_transitions import validate_trace
with np.load(sys.argv[1]) as t:
    meta = validate_trace(t)
    out = dict(robot=meta["robot"], task=meta["task"], steps=int(t["done"].shape[0]), envs=int(t["done"].shape[1]),
               falls=int((t["terminated"] & ~t["time_out"]).sum()), time_outs=int(t["time_out"].sum()),
               verified_terminal_rows=meta["verified_terminal_rows"], contact_mean=t["contact"].mean(0).mean(0).round(3).tolist(),
               reward_names=meta["reward_names"], foot_bodies=meta["timer_foot_body_names"])
print(json.dumps(out, indent=2))
PYEOF
RC=$?; echo "validate rc=$RC" >> "$OUT/rc.txt"
[ $RC -eq 0 ] && status "done PASS" || status "FAILED: validate rc=$RC (see validate.log)"
