import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str, default='/data/gomData/', help='path to GOM data')
parser.add_argument('--shotNum', dest='shotNum', type=int, default=0, help='shot number')
args = parser.parse_args()
data_path = args.data_path
shotNum = args.shotNum

strName = os.path.join(data_path, 'gomShots.hdf5')
fileShots = h5py.File(strName, 'r')

data_inter = fileShots['inter'][...]
data_mult = fileShots['mult'][...]
data_srme = fileShots['srme'][...]
data_T15 = fileShots['T15'][...]
data_T2 = fileShots['T2'][...]

traceNum = 2*184

from IPython import embed; embed()

font = {'family' : 'sans-serif',
        'size'   : 8}
import matplotlib
matplotlib.rc('font', **font)

plt.figure(dpi=100, figsize=(3, 7)); 
plt.imshow(data_inter[0, :, :], vmin=-100, vmax=100, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('inter', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
plt.axes().set_aspect(.7)

plt.figure(dpi=100, figsize=(3, 7)); 
plt.imshow(data_mult[0, :, :], vmin=-50, vmax=50, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('mult', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
plt.axes().set_aspect(.7)

plt.figure(dpi=100, figsize=(3, 7));
plt.imshow(data_srme[0, :, :], vmin=-100, vmax=100, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('srme', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
plt.axes().set_aspect(.7)

plt.figure(dpi=100, figsize=(3, 7)); 
plt.imshow(data_T15[0, :, :], vmin=-100, vmax=100, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('T15', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
plt.axes().set_aspect(.7)

plt.figure(dpi=100, figsize=(3, 7)); 
plt.imshow(data_T2[0, :, :], vmin=-100, vmax=100, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('T2', fontweight='bold')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.tight_layout()
plt.axes().set_aspect(.7)

plt.show()
