import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str, default='/data/NelsonData/hdf5', help='path to GOM data')
parser.add_argument('--shotNum', dest='shotNum', type=int, default=0, help='shot number')
args = parser.parse_args()
data_path = args.data_path
shotNum = args.shotNum

strName = os.path.join(data_path, 'NelsonData.hdf5')
fileShots = h5py.File(strName, 'r')

recon = fileShots["recon"][...]
mul = fileShots["mul"][...]
prim = fileShots["prim"][...]
srmemult = fileShots["srmemult"][...]
srmemult_poor = fileShots["srmemult_poor"][...]


traceNum = 401
timeSamples = recon.shape[0]

# from IPython import embed; embed()

font = {'family' : 'sans-serif',
        'size'   : 4}
import matplotlib
matplotlib.rc('font', **font)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(recon[shotNum, :, :], vmin=-50, vmax=50, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('recon', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(mul[shotNum, :, :], vmin=-50, vmax=50, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('mul', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
# plt.axes().set_aspect(.7)

plt.figure(dpi=200, figsize=(3, 7));
plt.imshow(prim[shotNum, :, :], vmin=-50, vmax=50, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('prim', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
# plt.axes().set_aspect(.7)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(srmemult[shotNum, :, :], vmin=-50, vmax=50, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('srmemult', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
# plt.axes().set_aspect(.7)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(srmemult_poor[shotNum, :, :], vmin=-50, vmax=50, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('srmemult_poor', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
# plt.axes().set_aspect(.7)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(recon[shotNum, :, :] - prim[shotNum, :, :], vmin=-50, vmax=50, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('removed events', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()

plt.show()
