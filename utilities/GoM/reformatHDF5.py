import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str, default='/data/gomData/test', help='path to GOM data')
parser.add_argument('--shotNum', dest='shotNum', type=int, default=0, help='shot number')
args = parser.parse_args()
data_path = args.data_path
shotNum = args.shotNum

file_name = os.path.join(data_path, 'GOMshots.hdf5')
hdf5file = h5py.File(file_name, 'r')

inter_w1 = hdf5file["shots.inter.group1.window1.segy"][...]
inter_w2 = hdf5file["shots.inter.group1.window2.segy"][...]
inter_w3 = hdf5file["shots.inter.group1.window3.segy"][...]

mult_w1 = hdf5file["shots.mult.group1.window1.segy"][...]
mult_w2 = hdf5file["shots.mult.group1.window2.segy"][...]
mult_w3 = hdf5file["shots.mult.group1.window3.segy"][...]

srme_w1 = hdf5file["shots.srme.group1.window1.segy"][...]
srme_w2 = hdf5file["shots.srme.group1.window2.segy"][...]
srme_w3 = hdf5file["shots.srme.group1.window3.segy"][...]

T15_w1 = hdf5file["shots.T1.5.group1.window1.segy"][...]
T15_w2 = hdf5file["shots.T1.5.group1.window2.segy"][...]
T15_w3 = hdf5file["shots.T1.5.group1.window3.segy"][...]

T2_w1 = hdf5file["shots.T2.group1.window1.segy"][...]
T2_w2 = hdf5file["shots.T2.group1.window2.segy"][...]
T2_w3 = hdf5file["shots.T2.group1.window3.segy"][...]

traceNum = 2*184
shotNum = 280



strName = os.path.join(data_path, 'gomShots.hdf5')

fileShots = h5py.File(strName, 'w-')
data_inter = fileShots.create_dataset('inter', (shotNum, 1476, traceNum))
data_mult = fileShots.create_dataset('mult', (shotNum, 1476, traceNum))
data_srme = fileShots.create_dataset('srme', (shotNum, 1476, traceNum))
data_T15 = fileShots.create_dataset('T15', (shotNum, 1476, traceNum))
data_T2 = fileShots.create_dataset('T2', (shotNum, 1476, traceNum))

for j in range(shotNum):
	data_inter[j, :, :] = np.concatenate((inter_w1[0:500, traceNum*j:traceNum*(j+1)], \
		inter_w2[18:500, traceNum*j:traceNum*(j+1)], inter_w3[18:, traceNum*j:traceNum*(j+1)]), axis=0)
	data_mult[j, :, :] = np.concatenate((mult_w1[0:500, traceNum*j:traceNum*(j+1)], \
		mult_w2[18:500, traceNum*j:traceNum*(j+1)], mult_w3[18:, traceNum*j:traceNum*(j+1)]), axis=0)
	data_srme[j, :, :] = np.concatenate((srme_w1[0:500, traceNum*j:traceNum*(j+1)], \
		srme_w2[18:500, traceNum*j:traceNum*(j+1)], srme_w3[18:, traceNum*j:traceNum*(j+1)]), axis=0)
	data_T15[j, :, :] = np.concatenate((T15_w1[0:500, traceNum*j:traceNum*(j+1)], \
		T15_w2[18:500, traceNum*j:traceNum*(j+1)], T15_w3[18:, traceNum*j:traceNum*(j+1)]), axis=0)
	data_T2[j, :, :] = np.concatenate((T2_w1[0:500, traceNum*j:traceNum*(j+1)], \
		T2_w2[18:500, traceNum*j:traceNum*(j+1)], T2_w3[18:, traceNum*j:traceNum*(j+1)]), axis=0)

fileShots.close()

# plt.figure(); plt.imshow(inter_w1[:, traceNum*k:traceNum*(k+1)], vmin=-.5, vmax=.5, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('shots_w1', fontweight='bold')
# plt.xlabel('Offset (m)', fontweight='bold', fontsize=8)
# plt.ylabel('Time (s)', fontweight='bold', fontsize=8)

# plt.figure(); plt.imshow(inter_w2[:, traceNum*k:traceNum*(k+1)], vmin=-.5, vmax=.5, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('inter_w2', fontweight='bold')
# plt.xlabel('Offset (m)', fontweight='bold', fontsize=8)
# plt.ylabel('Time (s)', fontweight='bold', fontsize=8)

# plt.figure(); plt.imshow(T15_w1[:, traceNum*k:traceNum*(k+1)], vmin=-.5, vmax=.5, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('T15_w1', fontweight='bold')
# plt.xlabel('Offset (m)', fontweight='bold', fontsize=8)
# plt.ylabel('Time (s)', fontweight='bold', fontsize=8)

# plt.figure(); plt.imshow(T15_w2[:, traceNum*k:traceNum*(k+1)], vmin=-.5, vmax=.5, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('T15_w2', fontweight='bold')
# plt.xlabel('Offset (m)', fontweight='bold', fontsize=8)
# plt.ylabel('Time (s)', fontweight='bold', fontsize=8)
