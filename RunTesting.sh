# #!/bin/bash -l

experiment_name=Nelson_A_recon-srmemult2prim_reflect_train_Fnet_ng32_nd64_poorPrediction_divide
repo_name=GOMdata-SRME-GAN

path_script=/home/ec2-user/$repo_name/src
path_utls=/home/ec2-user/$repo_name/utilities
path_model=/home/ec2-user/model/$experiment_name

path_DropboxAWS=GTDropbox:aws

yes | cp -r $path_script/. $path_model
yes | cp /home/ec2-user/$repo_name/RunTraining.sh $path_model
yes | cp /home/ec2-user/$repo_name/RunTesting.sh $path_model

CUDA_VISIBLE_DEVICES=0 python $path_model/main.py --dataset_dir=multiple --phase test --transfer 0 \
--batch_size 1 --continue_train True --checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample \
--log_dir $path_model/log --use_resnet 2 --input_nc 2 --output_nc 1 --ngf 32 --L1_lambda 1500.0

python $path_utls/showMappping.py --hdf5path $path_model/sample --test_num 100 --save_dir default

rclone sync -x -v $path_model $path_DropboxAWS/$experiment_name

# command="$path_script/main.py --dataset_dir=dispersion --phase train --which_direction BtoA --batch_size 1 --continue_train True --checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log"

# if python $command ; then
#     echo "Command succeeded"
#     rm -rf path_script=/home/ec2-user/scripts/$experiment_name
# else
#     echo "Command failed"
# fi
