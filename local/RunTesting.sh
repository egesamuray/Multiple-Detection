# #!/bin/bash -l

experiment_name=Nelson_A_recon-srmemult2prim_reflect_train_Fnet_ng32_nd64_poorPrediction_divide
repo_name=GOMdata-SRME-GAN
save_path=/nethome/asiahkoohi3/Desktop/SEG19/multiple

path_script=/nethome/asiahkoohi3/Desktop/Ali/$repo_name/src
path_model=/data/aws/$experiment_name
path_utls=/nethome/asiahkoohi3/Desktop/Ali/$repo_name/local

mkdir $save_path/$experiment_name
mkdir $save_path/$experiment_name/exp-1
mkdir $save_path/$experiment_name/exp-2

python $path_utls/showMappping.py --hdf5path $path_model/sample --test_num 20 --save_dir $save_path/$experiment_name --exp exp-1


# command="$path_script/main.py --dataset_dir=dispersion --phase train --which_direction BtoA --batch_size 1 --continue_train True --checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log"

# if python $command ; then
#     echo "Command succeeded"
#     rm -rf path_script=/home/ec2-user/scripts/$experiment_name
# else
#     echo "Command failed"
# fi


experiment_name=Nelson_A_recon-srmemult2prim-mult_reflect_train_Fnet_ng32_nd64_poorPrediction_divide
repo_name=GOMdata-SRME-GAN
save_path=/nethome/asiahkoohi3/Desktop/SEG19/multiple

path_script=/nethome/asiahkoohi3/Desktop/Ali/$repo_name/src
path_model=/data/aws/$experiment_name
path_utls=/nethome/asiahkoohi3/Desktop/Ali/$repo_name/local

mkdir $save_path/$experiment_name
mkdir $save_path/$experiment_name/exp-1
mkdir $save_path/$experiment_name/exp-2

python $path_utls/showMappping.py --hdf5path $path_model/sample --test_num 20 --save_dir $save_path/$experiment_name --exp exp-2



experiment_name=Nelson_A_recon-2prim_reflect_train_Fnet_ng32_nd64_divide
repo_name=GOMdata-SRME-GAN
save_path=/nethome/asiahkoohi3/Desktop/SEG19/multiple

path_script=/nethome/asiahkoohi3/Desktop/Ali/$repo_name/src
path_model=/data/aws/$experiment_name
path_utls=/nethome/asiahkoohi3/Desktop/Ali/$repo_name/local

mkdir $save_path/$experiment_name
mkdir $save_path/$experiment_name/exp-1
mkdir $save_path/$experiment_name/exp-2

python $path_utls/showMappping.py --hdf5path $path_model/sample --test_num 20 --save_dir $save_path/$experiment_name --exp exp-1




