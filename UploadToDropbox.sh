#!/usr/bin/env bash

set -euo pipefail

echo "Uploading to Dropbox"

experiment_name=$1

path_DropboxAWS=GTDropbox:aws
path_model="$HOME/model/$experiment_name"

rclone sync "$path_model" "$path_DropboxAWS/$experiment_name"
