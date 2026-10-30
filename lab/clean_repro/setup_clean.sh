#!/bin/bash
# Make an unmodified clone of ETH's repo at a fixed commit. Never touches the fork checkout.
# Usage: bash setup_clean.sh <clean_dir> <fork_dir>
set -euo pipefail
CLEAN=${1:-/workspace/rwm_clean}
FORK=${2:-/workspace/Uncertainty-Aware-Robotic-World-Model-Go2/upstream/robotic_world_model}
COMMIT=82147f9170a4f50fc44bcc2351ea925da9561e23   # ETH master on 29 Oct 2026 (README badge Isaac Lab 2.2.1)
MODEL=assets/models/pretrain_rnn_ens.pt
WANT=2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a   # LFS oid of the released model

if [ ! -d "$CLEAN/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/leggedrobotics/robotic_world_model.git "$CLEAN"
fi
cd "$CLEAN"
GIT_LFS_SKIP_SMUDGE=1 git -c advice.detachedHead=false checkout -q "$COMMIT"
echo "clean HEAD: $(git rev-parse HEAD)"

oid=$(git show "HEAD:$MODEL" | sed -n 's/^oid sha256://p')
[ "$oid" = "$WANT" ] || { echo "ERROR: LFS pointer oid $oid != $WANT"; exit 1; }
have=$(sha256sum "$MODEL" | cut -d' ' -f1)
if [ "$have" != "$WANT" ]; then
  echo "model is an LFS pointer here; copying the binary from the fork checkout"
  src=$(sha256sum "$FORK/$MODEL" | cut -d' ' -f1)
  [ "$src" = "$WANT" ] || { echo "ERROR: fork model sha $src != $WANT"; exit 1; }
  cp "$FORK/$MODEL" "$MODEL"
  git update-index --assume-unchanged "$MODEL"
fi
have=$(sha256sum "$MODEL" | cut -d' ' -f1)
[ "$have" = "$WANT" ] || { echo "ERROR: model sha $have != $WANT"; exit 1; }
echo "model sha256 OK: $have"
echo "csv sha256: $(sha256sum assets/data/state_action_data_0.csv | cut -d' ' -f1)"
# Only the copied LFS binary may differ from the commit; anything else is an error.
dirty=$(git status --porcelain --untracked-files=no)
[ -z "$dirty" ] || { echo "ERROR: clean clone has local changes:"; echo "$dirty"; exit 1; }
echo "SETUP OK"
