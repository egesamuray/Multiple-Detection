import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from math import floor

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str, default='/data/NelsonData/hdf5', help='path to Nelson data')
parser.add_argument('--save_path', dest='save_path', type=str, default='/data/NelsonData/trainingData_poorPrediction', help='path to save data')
args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path

strName = os.path.join(data_path, 'NelsonData.hdf5')
fileShots = h5py.File(strName, 'r')

recon = fileShots["recon"][...]
mul = fileShots["mul"][...]
prim = fileShots["prim"][...]
srmemult = fileShots["srmemult"][...]
srmemult_poor = fileShots["srmemult_poor"][...]

font = {'family' : 'sans-serif',
        'size'   : 5}
import matplotlib
matplotlib.rc('font', **font)

a = recon[10, :, int(recon.shape[2]/2):]
fa = np.absolute(np.fft.fftshift(np.fft.fft2(a)))
fa = fa[floor(np.shape(fa)[0]/2):, :]
plt.figure(dpi=200)
plt.imshow(fa, vmin=0, vmax=55483)
plt.title(r'$f-k$' + ' spectrum of shot record')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.5)
plt.savefig('/nethome/asiahkoohi3/Desktop/Ali/GOMdata-SRME-GAN/report/Nelson/figs/fk.png', format='png', bbox_inches='tight', dpi=400)

font = {'family' : 'sans-serif',
        'size'   : 10}
import matplotlib
matplotlib.rc('font', **font)


traceNum = 401


strTrainA = os.path.join(save_path,  'Nelson_A_recon2prim_reflect_train.hdf5')
strTrainB = os.path.join(save_path,  'Nelson_B_recon2prim_reflect_train.hdf5')
strTestA = os.path.join(save_path,  'Nelson_A_recon2prim_reflect_test.hdf5')
strTestB = os.path.join(save_path,  'Nelson_B_recon2prim_reflect_test.hdf5')

dataset_train = "train_dataset"
dataset_test = "test_dataset"

train_size = int(recon.shape[0]/2)+1
test_size = int(recon.shape[0]/2)

file_TrainA = h5py.File(strTrainA, 'w-')
file_TrainB  = h5py.File(strTrainB , 'w-')
file_TestA = h5py.File(strTestA, 'w-')
file_TestB  = h5py.File(strTestB , 'w-')

shape = [recon[0, :, :].shape[0], recon[0, :, :].shape[1]]

dataset_TrainA = file_TrainA.create_dataset(dataset_train, (train_size*2, shape[0], shape[1], 2))
dataset_TrainB = file_TrainB.create_dataset(dataset_train, (train_size*2, shape[0], shape[1], 2))
dataset_TestA = file_TestA.create_dataset(dataset_test, (test_size, shape[0], shape[1], 2))
dataset_TestB = file_TestB.create_dataset(dataset_test, (test_size, shape[0], shape[1], 2))


k = 0
for i in range(0, prim.shape[0], 2):
	dataset_TrainA[k, :, :, 0] = prim[i, :, :]/np.linalg.norm(prim[i, :, :].reshape(-1), np.inf)
	dataset_TrainA[k, :, :, 1] = mul[i, :, :]/np.linalg.norm(mul[i, :, :].reshape(-1), np.inf)
	dataset_TrainB[k, :, :, 0] = recon[i, :, :]/np.linalg.norm(recon[i, :, :].reshape(-1), np.inf)
	dataset_TrainB[k, :, :, 1] = srmemult_poor[i, :, :]/np.linalg.norm(srmemult_poor[i, :, :].reshape(-1), np.inf)

	dataset_TrainA[k + train_size, :, :, 0] = np.flip(dataset_TrainA[k, :, :, 0], axis=1)
	dataset_TrainA[k + train_size, :, :, 1] = np.flip(dataset_TrainA[k, :, :, 1], axis=1)
	dataset_TrainB[k + train_size, :, :, 0] = np.flip(dataset_TrainB[k, :, :, 0], axis=1)
	dataset_TrainB[k + train_size, :, :, 1] = np.flip(dataset_TrainB[k, :, :, 1], axis=1)
	k = k + 1

k = 0
for i in range(1, prim.shape[0], 2):
	dataset_TestA[k, :, :, 0] = prim[i, :, :]/np.linalg.norm(prim[i, :, :].reshape(-1), np.inf)
	dataset_TestA[k, :, :, 1] = mul[i, :, :]/np.linalg.norm(mul[i, :, :].reshape(-1), np.inf)
	dataset_TestB[k, :, :, 0] = recon[i, :, :]/np.linalg.norm(recon[i, :, :].reshape(-1), np.inf)
	dataset_TestB[k, :, :, 1] = srmemult_poor[i, :, :]/np.linalg.norm(srmemult_poor[i, :, :].reshape(-1), np.inf)
	k = k +1


file_TrainA.close()
file_TrainB.close()
file_TestA.close()
file_TestB.close()



# font = {'family' : 'sans-serif',
#         'size'   : 4}
# import matplotlib
# matplotlib.rc('font', **font)


# plt.figure(dpi=200, figsize=(3, 7)); 
# plt.imshow(dataset_TrainA[0, :, :, 0], vmin=-.1, vmax=.1, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('dataset_TrainA - 0', fontweight='bold')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.tight_layout()

# plt.figure(dpi=200, figsize=(3, 7)); 
# plt.imshow(dataset_TrainA[0, :, :, 1], vmin=-.1, vmax=.1, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('dataset_TrainA - 1', fontweight='bold')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.tight_layout()

# plt.figure(dpi=200, figsize=(3, 7)); 
# plt.imshow(dataset_TrainB[0, :, :, 0], vmin=-.1, vmax=.1, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('dataset_TrainB - 0', fontweight='bold')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.tight_layout()

# plt.figure(dpi=200, figsize=(3, 7)); 
# plt.imshow(dataset_TrainB[0, :, :, 1], vmin=-.1, vmax=.1, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('dataset_TrainB - 1', fontweight='bold')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.tight_layout()

# plt.figure(dpi=200, figsize=(3, 7)); 
# plt.imshow(dataset_TestA[0, :, :, 0], vmin=-.1, vmax=.1, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('dataset_TestA - 0', fontweight='bold')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.tight_layout()

# plt.figure(dpi=200, figsize=(3, 7)); 
# plt.imshow(dataset_TestA[0, :, :, 1], vmin=-.1, vmax=.1, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('dataset_TestA - 1', fontweight='bold')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.tight_layout()

# plt.figure(dpi=200, figsize=(3, 7)); 
# plt.imshow(dataset_TestB[0, :, :, 0], vmin=-.1, vmax=.1, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('dataset_TestB - 0', fontweight='bold')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.tight_layout()

# plt.figure(dpi=200, figsize=(3, 7)); 
# plt.imshow(dataset_TestB[0, :, :, 1], vmin=-.1, vmax=.1, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('dataset_TestB - 1', fontweight='bold')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.tight_layout()