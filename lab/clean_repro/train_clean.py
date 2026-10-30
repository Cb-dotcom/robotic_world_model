#!/usr/bin/env python3
"""Train an offline ANYmal policy with an UNMODIFIED clone of ETH's repo (README Option 2).

The only additions are outside the clone and do not change training:
  * a copy of every wandb.log dict is appended to --metrics_jsonl (imagined metrics);
  * --lambda_ overrides uncertainty_penalty_weight (lambda sweep only; omit for the clean repro);
  * the effective config and import paths are written to --manifest.

Run from anywhere (the script changes into the clone):
  WANDB_MODE=offline /isaac-sim/python.sh train_clean.py --clone /workspace/rwm_clean \
      --run_num 2001 --metrics_jsonl out/metrics.jsonl --manifest out/manifest.json
"""
import argparse
import hashlib
import json
import os
import sys
import types

p = argparse.ArgumentParser()
p.add_argument("--clone", required=True)
p.add_argument("--run_num", type=int, required=True)
p.add_argument("--lambda_", type=float, default=None, help="uncertainty penalty weight lambda (>= 0); default = repo value")
p.add_argument("--metrics_jsonl", required=True)
p.add_argument("--manifest", required=True)
p.add_argument("--dry_run", action="store_true", help="build and print the config only (no torch needed)")
a = p.parse_args()

clone = os.path.abspath(a.clone)
mb = os.path.join(clone, "scripts", "reinforcement_learning", "model_based")
metrics_path, manifest_path = os.path.abspath(a.metrics_jsonl), os.path.abspath(a.manifest)
os.chdir(clone)                      # repo paths ("assets/...", "logs/...") are relative to the clone root
sys.path.insert(0, mb)               # clone modules first, never the fork's
os.environ.setdefault("WANDB_MODE", "offline")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


import configs  # noqa: E402

if not os.path.abspath(configs.__file__).startswith(mb):
    sys.exit(f"configs imported from {configs.__file__}, not from the clone")
config = configs.AnymalDFlatConfig()
repo_lambda = -config.environment_config.uncertainty_penalty_weight
if a.lambda_ is not None:
    config.environment_config.uncertainty_penalty_weight = -abs(a.lambda_)

manifest = {
    "clone": clone, "run_num": a.run_num,
    "lambda_repo": repo_lambda, "lambda_used": -config.environment_config.uncertainty_penalty_weight,
    "config": config.to_dict(),
}
for rel in ("assets/models/pretrain_rnn_ens.pt", "assets/data/state_action_data_0.csv"):
    manifest[f"sha256:{rel}"] = sha256(rel) if os.path.isfile(rel) else None

if a.dry_run:
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2))
    print("env:", json.dumps(config.environment_config.to_dict()))
    print("policy_alg:", json.dumps(config.policy_algorithm_config.to_dict()))
    print("policy_training:", json.dumps(config.policy_training_config.to_dict()))
    sys.exit(0)

try:
    import wandb  # noqa: F401
except ImportError:  # stub only if wandb is absent; it does not change training
    stub = types.ModuleType("wandb")
    stub.run = types.SimpleNamespace(name=None)
    stub.config = types.SimpleNamespace(update=lambda *x, **y: None)
    stub.init = lambda *x, **y: stub.run
    stub.log = lambda *x, **y: None
    sys.modules["wandb"] = stub
import wandb  # noqa: E402

_log = wandb.log


def tee(d, *x, **y):
    row = {}
    for k, v in d.items():
        try:
            row[k] = float(v)
        except (TypeError, ValueError):
            pass
    with open(metrics_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return _log(d, *x, **y)


wandb.log = tee

import torch  # noqa: E402

# torch >= 2.6 changed torch.load's default to weights_only=True, which refuses the released
# checkpoint. Restore the old default only; the loaded tensors are identical.
_tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
manifest["torch_version"] = torch.__version__
manifest["torch_load_weights_only_default_restored"] = _tv >= (2, 6)
if _tv >= (2, 6):
    _load = torch.load
    torch.load = lambda *x, **y: _load(*x, **{"weights_only": False, **y})

import rsl_rl  # noqa: E402
import train as T  # noqa: E402  (clone's train.py; its __main__ block does not run on import)

for mod in (T, configs, sys.modules["envs"], sys.modules["policy_training"]):
    if not os.path.abspath(mod.__file__).startswith(mb):
        sys.exit(f"{mod.__name__} imported from {mod.__file__}, not from the clone")
manifest["imports"] = {m: os.path.abspath(sys.modules[m].__file__) for m in ("train", "configs", "envs", "policy_training")}
manifest["rsl_rl_file"] = os.path.abspath(rsl_rl.__file__)
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=str)

T.args_cli = argparse.Namespace(task="anymal_d_flat", run_num=a.run_num)
T.run(config)
