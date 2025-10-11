# MultipleEliminationGAN 

This repository contains codes for reading the GOM dataset, generating training and testing dataset, and training a GAN to remove multiples.

## Prerequisites

* SeismicUnix

* TensorFlow

* `h5py`

* `SeisIO`

### Installation

* Install TensorFlow from https://github.com/JohnWStockwellJr/SeisUnix

* Install TensorFlow from https://github.com/tensorflow/tensorflow

* Install `h5py` with `pip instal h5py`

* Install `SeisIO` from https://github.com/slimgroup/SeisIO.jl



## Script descriptions


`utilities/SU2segy.sh`\: It inputs the GOM data in `su` format and saves it in `segy`\.

`utilities/segy2HDF5.jl`\: It inputs the GOM data in `su` format and saves it in `hsf5`\.

`utilities/VizData.py`\: It inputs the GOM data in `hdf5` format and provides tools to visualize the data

`utilities/genDataset.py`\: It inputs the GOM data in `hdf5` format and generates training and testing dataset for training a GAN

`RunTraining.sh`\: Script for running training on `AWS` cloud. Currently needs permission to access my Dropbox account with `rclone` for downloading the training and testing data set into the `AWS` instance.

`RunTesting.sh`\: Script for testing the trained network on `AWS` cloud and saving the result of the mapping in `hdf5` format. Currently needs permission to access my Dropbox account with `rclone` to upload the results.

`src/main.py`\: The main function to run the training on any machine. The name of training and testing data sets need to be specified in `src/model.py`\.

`show_map.py`\: Plots the interpolated frequency slice after testing stage.


    .
    ├── info                    # Information regrding to the structure of the raw .su GOM dataset
	|
    ├── src                     # scripts for training a GAN
    |
    ├── utilities               # scripts for reading/visualizing the GOM dataset and generating training/testing dataset
    |
    └── ...


### Running the tests

To change the format of `.su` files into `segy`:

```bash
bash utilities/SU2segy.sh
```

To change the format of `segy` files into `hsf5`:

```bash
julia utilities/segy2HDF5.jl
```
To generate training data using the GOM data in `hdf5` format:

```bash
python genDataset.py --data_path /pathToHdf5Dataset
```

To run training on AWS run the following (needs permission to access my Dropbox via `rclone` to download the data sets from my Dropbox). Result of training will be stored in my Dropbox. Path to data set in Dropbox can be set in `RunTraining.sh` using `path_DropboxData` variable and path to save the checkpoints for training can be set in `path_DropboxAWS` variable.

```bash
bash RunTraining.sh
```

To run training on you own machine run the following in `src/` directory. Make sure to enter the name of data sets in `src/model.py`\.

```bash
# Running in CPU
python main.py --data_path /pathToDataset  --dataset_dir=NameOfExperiment --phase train --which_direction BtoA --batch_size 1 --continue_train True

# Running in GPU
CUDA_VISIBLE_DEVICES=0 python main.py --data_path /pathToDataset  --dataset_dir=NameOfExperiment --phase train --which_direction BtoA --batch_size 1 --continue_train True
```

To run the trained network on test data set on `AWS`\:

```bash
bash RunTesting.sh
```

To run the trained network on test data set on you own machine run the following in `src/` directory. Make sure to enter the name of data sets in `src/model.py`\.

```bash
# Running in CPU
python main.py --data_path /pathToDataset  --dataset_dir=NameOfExperiment --phase test --which_direction BtoA --batch_size 1 --continue_train True

# Running in GPU
CUDA_VISIBLE_DEVICES=0 python main.py --data_path /pathToDataset  --dataset_dir=NameOfExperiment --phase test --which_direction BtoA --batch_size 1 --continue_train True
```


To show the result after performing the testing stage run the following:

```bash
python show_map.py
```


## Author

Ali Siahkoohi
