# #!/bin/bash -l

experiment_name=GOMshot_MultipleElimination-inter-mult-2T2_Fnet_ng32_nd64
repo_name=GOMdata-SRME-GAN

path_script=/nethome/asiahkoohi3/Desktop/Ali/$repo_name/src
path_data=/data/NelsonData/trainingData
path_model=$path_script

python $path_script/main.py --dataset_dir=multiple --phase train --transfer 0 \
--epoch 400 --epoch_step 50 --batch_size 1 --save_freq 3000  --print_freq 50 --continue_train True \
--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \
--use_resnet 2 --input_nc 1 --output_nc 1 --ngf 32 --L1_lambda 1500.0 --data_path $path_data


# command="$path_script/main.py --dataset_dir=dispersion --phase train --which_direction BtoA --batch_size 1 --continue_train True --checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log"

# if python $command ; then
#     echo "Command succeeded"
#     rm -rf path_script=/home/ec2-user/scripts/$experiment_name
# else
#     echo "Command failed"
# fi
