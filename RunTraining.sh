#!/usr/bin/env bash

set -euo pipefail

experiment_name=Nelson_A_recon-srmemult2prim_reflect_train_Fnet_ng32_nd64_poorPrediction_divide

trainA_str=Nelson_A_recon2prim_reflect_train.hdf5
trainB_str=Nelson_B_recon2prim_reflect_train.hdf5
testA_str=Nelson_A_recon2prim_reflect_test.hdf5
testB_str=Nelson_B_recon2prim_reflect_test.hdf5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/src" ]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

path_script="$REPO_ROOT/src"
path_data="$HOME/data"
path_model="$HOME/model/$experiment_name"

path_DropboxAWS=GTDropbox:aws
path_DropboxData=GTDropbox:optimum/scratch/slim/alisk/Map_GAN/DispGAN/data/NelsonData/trainingData_poorPrediction_divide

mkdir -p "$path_data"
mkdir -p "$path_model"
rclone mkdir "$path_DropboxAWS/$experiment_name" || true

yes | cp -r "$path_script/." "$path_model"
yes | cp "$REPO_ROOT/RunTraining.sh" "$path_model"
yes | cp "$REPO_ROOT/RunTesting.sh" "$path_model"

rclone sync -x -v "$path_DropboxData/$trainA_str" "$path_data"
rclone sync -x -v "$path_DropboxData/$trainB_str" "$path_data"
rclone sync -x -v "$path_DropboxData/$testA_str" "$path_data"
rclone sync -x -v "$path_DropboxData/$testB_str" "$path_data"

CUDA_VISIBLE_DEVICES=0 python "$path_script/main.py" --dataset_dir=multiple --phase train --transfer 0 \
  --epoch 400 --epoch_step 50 --batch_size 1 --save_freq 3000 --print_freq 50 --continue_train True \
  --checkpoint_dir "$path_model/checkpoint" --sample_dir "$path_model/sample" --log_dir "$path_model/log" \
  --use_resnet 2 --input_nc 2 --output_nc 1 --ngf 32 --L1_lambda 1500.0

rclone sync -x -v "$path_model" "$path_DropboxAWS/$experiment_name"
