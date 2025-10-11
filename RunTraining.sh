# #!/bin/bash -l

experiment_name=Nelson_A_recon-srmemult2prim_reflect_train_Fnet_ng32_nd64_poorPrediction_divide
repo_name=GOMdata-SRME-GAN

trainA_str=Nelson_A_recon2prim_reflect_train.hdf5
trainB_str=Nelson_B_recon2prim_reflect_train.hdf5
testA_str=Nelson_A_recon2prim_reflect_test.hdf5
testB_str=Nelson_B_recon2prim_reflect_test.hdf5

path_script=/home/ec2-user/$repo_name/src
path_data=/home/ec2-user/data
path_model=/home/ec2-user/model/$experiment_name

path_DropboxAWS=GTDropbox:aws
path_DropboxData=GTDropbox:optimum/scratch/slim/alisk/Map_GAN/DispGAN/data/NelsonData/trainingData_poorPrediction_divide

mkdir $HOME/data
mkdir /home/ec2-user/model/
mkdir $path_model
rclone mkdir $path_DropboxAWS/$experiment_name

yes | cp -r $path_script/. $path_model
yes | cp /home/ec2-user/$repo_name/RunTraining.sh $path_model
yes | cp /home/ec2-user/$repo_name/RunTesting.sh $path_model

rclone sync -x -v $path_DropboxData/$trainA_str $path_data
rclone sync -x -v $path_DropboxData/$trainB_str $path_data
rclone sync -x -v $path_DropboxData/$testA_str $path_data
rclone sync -x -v $path_DropboxData/$testB_str $path_data


CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --dataset_dir=multiple --phase train --transfer 0 \
--epoch 400 --epoch_step 50 --batch_size 1 --save_freq 3000  --print_freq 50 --continue_train True \
--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \
--use_resnet 2 --input_nc 2 --output_nc 1 --ngf 32 --L1_lambda 1500.0

rclone sync -x -v $path_model $path_DropboxAWS/$experiment_name

# command="$path_script/main.py --dataset_dir=dispersion --phase train --which_direction BtoA --batch_size 1 --continue_train True --checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log"

# if python $command ; then
#     echo "Command succeeded"
#     rm -rf path_script=/home/ec2-user/scripts/$experiment_name
# else
#     echo "Command failed"
# fi
