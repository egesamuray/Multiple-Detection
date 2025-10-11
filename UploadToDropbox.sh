#!/bin/bash -l

echo "Uploading to Dropbox"

experiment_name=$1

path_DropboxAWS=GTDropbox:aws
path_model=/home/ec2-user/model/$experiment_name

rclone sync $path_model $path_DropboxAWS/$experiment_name
