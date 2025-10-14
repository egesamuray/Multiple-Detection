#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/src" ]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

path_script="$REPO_ROOT/src"
path_data="${DATA_PATH:-/data/NelsonData/trainingData}"
path_model="$path_script"

python "$path_script/main.py" --dataset_dir=multiple --phase train --transfer 0 \
  --epoch 400 --epoch_step 50 --batch_size 1 --save_freq 3000 --print_freq 50 --continue_train True \
  --checkpoint_dir "$path_model/checkpoint" --sample_dir "$path_model/sample" --log_dir "$path_model/log" \
  --use_resnet 2 --input_nc 1 --output_nc 1 --ngf 32 --L1_lambda 1500.0 --data_path "$path_data"
