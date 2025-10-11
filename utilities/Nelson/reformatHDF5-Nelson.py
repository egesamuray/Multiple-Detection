import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str, default='/data/NelsonData/hdf5', help='path to Nelson data')
parser.add_argument('--shotNum', dest='shotNum', type=int, default=0, help='shot number')
args = parser.parse_args()
data_path = args.data_path
shotNum = args.shotNum

file_name = os.path.join(data_path, 'NelsonShots.hdf5')
hdf5file = h5py.File(file_name, 'r')

recon = hdf5file["shots.fixed.1-401.recon.segy"][...]
mul = hdf5file["shots.fixed.1-401.mul.NO.segy"][...]
prim = hdf5file["shots.fixed.1-401.prim.segy"][...]
srmemult = hdf5file["shots.fixed.1-401.srmemult.segy"][...]
srmemult_poor = hdf5file["shots.fixed.1-401.srmemult-poor.segy"][...]

traceNum = 401
shotNum = 401
timeSamples = recon.shape[0]


strName = os.path.join(data_path, 'NelsonData.hdf5')

fileShots = h5py.File(strName, 'w-')
data_recon = fileShots.create_dataset('recon', (shotNum, timeSamples, traceNum))
data_mul = fileShots.create_dataset('mul', (shotNum, timeSamples, traceNum))
data_prim = fileShots.create_dataset('prim', (shotNum, timeSamples, traceNum))
data_srmemult = fileShots.create_dataset('srmemult', (shotNum, timeSamples, traceNum))
data_srmemult_poor = fileShots.create_dataset('srmemult_poor', (shotNum, timeSamples, traceNum))

for j in range(shotNum):
	data_recon[j, :, :] = recon[:, traceNum*j:traceNum*(j+1)]
	data_mul[j, :, :] = mul[:, traceNum*j:traceNum*(j+1)]
	data_prim[j, :, :] = prim[:, traceNum*j:traceNum*(j+1)]
	data_srmemult[j, :, :] = srmemult[:, traceNum*j:traceNum*(j+1)]
	data_srmemult_poor[j, :, :] = srmemult_poor[:, traceNum*j:traceNum*(j+1)]
fileShots.close()

# k = 0
# plt.figure(); plt.imshow(recon[0:recon.shape[0]:2, traceNum*k:traceNum*(k+1)], vmin=-50, vmax=50, cmap="Greys", aspect='auto', interpolation="lanczos")
# plt.title('recon', fontweight='bold')
# plt.xlabel('Offset (m)', fontweight='bold', fontsize=8)
# plt.ylabel('Time (s)', fontweight='bold', fontsize=8)

# # plt.figure(); plt.imshow(inter_w2[:, traceNum*k:traceNum*(k+1)], vmin=-.5, vmax=.5, cmap="Greys", aspect='auto', interpolation="lanczos")
# # plt.title('inter_w2', fontweight='bold')
# # plt.xlabel('Offset (m)', fontweight='bold', fontsize=8)
# # plt.ylabel('Time (s)', fontweight='bold', fontsize=8)

# # plt.figure(); plt.imshow(T15_w1[:, traceNum*k:traceNum*(k+1)], vmin=-.5, vmax=.5, cmap="Greys", aspect='auto', interpolation="lanczos")
# # plt.title('T15_w1', fontweight='bold')
# # plt.xlabel('Offset (m)', fontweight='bold', fontsize=8)
# # plt.ylabel('Time (s)', fontweight='bold', fontsize=8)

# # plt.figure(); plt.imshow(T15_w2[:, traceNum*k:traceNum*(k+1)], vmin=-.5, vmax=.5, cmap="Greys", aspect='auto', interpolation="lanczos")
# # plt.title('T15_w2', fontweight='bold')
# # plt.xlabel('Offset (m)', fontweight='bold', fontsize=8)
# # plt.ylabel('Time (s)', fontweight='bold', fontsize=8)

# a = recon[:, traceNum*k:traceNum*(k+1)]
# a = a[0:a.shape[0]:2, :]
# fa = np.absolute(np.fft.fftshift(np.fft.fft2(a)))
# fa = fa[floor(np.shape(fa)[0]/2):, :]
# plt.figure(dpi=200)
# plt.imshow(fa, vmin=0, vmax=55483)
# plt.title(r'$f-k$' + ' spectrum of shot record')
# plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
#     right='False', left='False', labelleft='False')
# plt.axes().set_aspect(.5)