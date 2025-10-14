#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/src" ]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

path_utils="$REPO_ROOT/local"
model_root="${MODEL_ROOT:-/data/aws}"
save_path="${SAVE_PATH:-$REPO_ROOT/local/output}"

mkdir -p "$save_path"

run_experiment() {
  local experiment_name="$1"
  local exp_label="$2"
  local test_num="$3"

  local path_model="$model_root/$experiment_name"
  mkdir -p "$save_path/$experiment_name/$exp_label"

  python "$path_utils/showMappping.py" \
    --hdf5path "$path_model/sample" \
    --test_num "$test_num" \
    --save_dir "$save_path/$experiment_name" \
    --exp "$exp_label"
}

run_experiment \
  "Nelson_A_recon-srmemult2prim_reflect_train_Fnet_ng32_nd64_poorPrediction_divide" \
  "exp-1" \
  20

run_experiment \
  "Nelson_A_recon-srmemult2prim-mult_reflect_train_Fnet_ng32_nd64_poorPrediction_divide" \
  "exp-2" \
  20

run_experiment \
  "Nelson_A_recon-2prim_reflect_train_Fnet_ng32_nd64_divide" \
  "exp-1" \
  20
