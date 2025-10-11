import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from math import floor

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str, default='/data/gomData/', help='path to GOM data')
parser.add_argument('--save_path', dest='save_path', type=str, default='/data/GOMdata/data/', help='path to save data')
args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path

strName = os.path.join(data_path, 'gomShots.hdf5')
fileShots = h5py.File(strName, 'r')

data_inter = fileShots['inter'][...]
data_mult = fileShots['mult'][...]
data_T2 = fileShots['T2'][...]

font = {'family' : 'sans-serif',
        'size'   : 5}
import matplotlib
matplotlib.rc('font', **font)

a = data_inter[10, :, int(data_inter.shape[2]/2):]
fa = np.absolute(np.fft.fftshift(np.fft.fft2(a)))
fa = fa[floor(np.shape(fa)[0]/2):, :]
plt.figure(dpi=200)
plt.imshow(fa, vmin=0, vmax=55483)
plt.title(r'$f-k$' + ' spectrum of shot record')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.5)
plt.savefig('/nethome/asiahkoohi3/Desktop/Ali/GOMdata-SRME-GAN/report/figs/fk.png', format='png', bbox_inches='tight', dpi=400)

font = {'family' : 'sans-serif',
        'size'   : 10}
import matplotlib
matplotlib.rc('font', **font)


a = data_inter[10, ::2, int(data_inter.shape[2]/2):]
fa = np.absolute(np.fft.fftshift(np.fft.fft2(a)))
fa = fa[floor(np.shape(fa)[0]/2):, :]
plt.figure(dpi=200) 
plt.imshow(fa, vmin=0, vmax=55483/2)
plt.title(r'$f-k$' + ' spectrum of sub-sampled shot record')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.5)
plt.savefig('/nethome/asiahkoohi3/Desktop/Ali/GOMdata-SRME-GAN/report/figs/fk-subsampled.png', format='png', bbox_inches='tight', dpi=400)

traceNum = 2*184


strTrainA = os.path.join(save_path, 'trainingData/inter2T2-DoubleSided', 'GOMshot_A_inter2T2-DoubleSide_train.hdf5')
strTrainB = os.path.join(save_path, 'trainingData/inter2T2-DoubleSided', 'GOMshot_B_inter2T2-DoubleSide_train.hdf5')
strTestA = os.path.join(save_path, 'trainingData/inter2T2-DoubleSided', 'GOMshot_A_inter2T2-DoubleSide_test.hdf5')
strTestB = os.path.join(save_path, 'trainingData/inter2T2-DoubleSided', 'GOMshot_B_inter2T2-DoubleSide_test.hdf5')

dataset_train = "train_dataset"
dataset_test = "test_dataset"

train_size = int(data_inter.shape[0]/2)
test_size = int(data_inter.shape[0]/2)

file_TrainA = h5py.File(strTrainA, 'w-')
file_TrainB  = h5py.File(strTrainB , 'w-')
file_TestA = h5py.File(strTestA, 'w-')
file_TestB  = h5py.File(strTestB , 'w-')

shape = [data_inter[0, ::2, int(data_inter.shape[2]/2):].shape[0], \
			data_inter[0, ::2, int(data_inter.shape[2]/2):].shape[1]]

dataset_TrainA = file_TrainA.create_dataset(dataset_train, (train_size*2, shape[0], shape[1], 1))
dataset_TrainB = file_TrainB.create_dataset(dataset_train, (train_size*2, shape[0], shape[1], 2))
dataset_TestA = file_TestA.create_dataset(dataset_test, (test_size, shape[0], shape[1], 1))
dataset_TestB = file_TestB.create_dataset(dataset_test, (test_size, shape[0], shape[1], 2))


dataset_TrainA[:train_size, :, :, 0] =  data_T2[::2, ::2, \
	int(data_T2.shape[2]/2):]/np.linalg.norm(data_T2[::2, ::2, \
	int(data_T2.shape[2]/2):].reshape(-1), np.inf)

dataset_TrainB[:train_size, :, :, 0] =  data_inter[::2, ::2, \
	int(data_inter.shape[2]/2):]/np.linalg.norm(data_inter[::2, ::2, \
	int(data_inter.shape[2]/2):].reshape(-1), np.inf)

dataset_TrainB[:train_size, :, :, 1] =  data_mult[::2, ::2, \
	int(data_mult.shape[2]/2):]/np.linalg.norm(data_mult[::2, ::2, \
	int(data_mult.shape[2]/2):].reshape(-1), np.inf)


dataset_TrainA[train_size:, :, :, 0] =  data_T2[::2, ::2, \
	:int(data_T2.shape[2]/2)]/np.linalg.norm(data_T2[::2, ::2, \
	:int(data_T2.shape[2]/2)].reshape(-1), np.inf)

dataset_TrainB[train_size:, :, :, 0] =  data_inter[::2, ::2, \
	:int(data_inter.shape[2]/2)]/np.linalg.norm(data_inter[::2, ::2, \
	:int(data_inter.shape[2]/2)].reshape(-1), np.inf)

dataset_TrainB[train_size:, :, :, 1] =  data_mult[::2, ::2, \
	:int(data_mult.shape[2]/2)]/np.linalg.norm(data_mult[::2, ::2, \
	:int(data_mult.shape[2]/2)].reshape(-1), np.inf)


dataset_TestA[:, :, :, 0] =  data_T2[1::2, ::2, \
	int(data_T2.shape[2]/2):]/np.linalg.norm(data_T2[1::2, ::2, \
	int(data_T2.shape[2]/2):].reshape(-1), np.inf)

dataset_TestB[:, :, :, 0] =  data_inter[1::2, ::2, \
	int(data_inter.shape[2]/2):]/np.linalg.norm(data_inter[1::2, ::2, \
	int(data_inter.shape[2]/2):].reshape(-1), np.inf)

dataset_TestB[:, :, :, 1] =  data_mult[1::2, ::2, \
	int(data_mult.shape[2]/2):]/np.linalg.norm(data_mult[1::2, ::2, \
	int(data_mult.shape[2]/2):].reshape(-1), np.inf)

file_TrainA.close()
file_TrainB.close()
file_TestA.close()
file_TestB.close()