#!/usr/bin/env bash

set -euo pipefail

experiment_name=Nelson_A_recon-srmemult2prim_reflect_train_Fnet_ng32_nd64_poorPrediction_divide

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/src" ]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

path_script="$REPO_ROOT/src"
path_utils="$REPO_ROOT/utilities"
path_model="$HOME/model/$experiment_name"

path_DropboxAWS=GTDropbox:aws

mkdir -p "$path_model"

yes | cp -r "$path_script/." "$path_model"
yes | cp "$REPO_ROOT/RunTraining.sh" "$path_model"
yes | cp "$REPO_ROOT/RunTesting.sh" "$path_model"

CUDA_VISIBLE_DEVICES=0 python "$path_model/main.py" --dataset_dir=multiple --phase test --transfer 0 \
  --batch_size 1 --continue_train True --checkpoint_dir "$path_model/checkpoint" --sample_dir "$path_model/sample" \
  --log_dir "$path_model/log" --use_resnet 2 --input_nc 2 --output_nc 1 --ngf 32 --L1_lambda 1500.0

python "$path_utils/showMappping.py" --hdf5path "$path_model/sample" --test_num 100 --save_dir default

rclone sync -x -v "$path_model" "$path_DropboxAWS/$experiment_name"
