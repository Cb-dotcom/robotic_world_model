#!/bin/bash
# Record the software versions of this container (ETH's README now says Isaac Lab 2.2.1, Isaac Sim 4.5.0).
PY=${PY:-/isaac-sim/python.sh}
echo "date_utc: $(date -u +%FT%TZ)"
echo "isaac_sim_VERSION: $(cat /isaac-sim/VERSION 2>/dev/null || echo unknown)"
$PY -m pip show isaaclab isaaclab_rl isaaclab_tasks rsl-rl-lib torch 2>/dev/null | grep -E "^(Name|Version|Location|Editable project location):"
$PY - <<'EOF' 2>/dev/null
import importlib, os, subprocess
for m in ("isaaclab", "isaaclab_tasks", "rsl_rl"):
    try:
        mod = importlib.import_module(m)
        d = os.path.dirname(os.path.abspath(mod.__file__))
        rev = subprocess.run(["git", "-C", d, "describe", "--always", "--dirty", "--tags"],
                             capture_output=True, text=True).stdout.strip() or "not a git checkout"
        print(f"{m}: file={mod.__file__} version={getattr(mod, '__version__', 'n/a')} git={rev}")
    except Exception as e:
        print(f"{m}: import failed: {e}")
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
EOF
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
