import argparse
import os
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(seed=19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--hdf5path', dest='hdf5path', type=str, default='/home/ec2-user/model', help='path')
parser.add_argument('--test_num', dest='test_num', type=int, default=10, help='test number')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='default', help='saving directory')
parser.add_argument('--exp', dest='exp', type=str, default='exp-1', help='saving directory')
args = parser.parse_args()

hdf5path  = args.hdf5path
hdf5name = os.path.join(hdf5path, 'mapping_result.hdf5')
test_num = args.test_num
save_dir = args.save_dir
exp = args.exp

strName = hdf5name
dataset_name = "result"

fileName = h5py.File(strName, 'r')
data_num = fileName[dataset_name].shape[0]
data_numA = fileName[dataset_name + 'A'].shape[0]
data_numB = fileName[dataset_name + 'B'].shape[0]

if data_num == 0:
    raise ValueError('Dataset "{0}" in {1} is empty.'.format(dataset_name, hdf5name))

if test_num < 0 or test_num >= data_num:
    new_index = test_num % data_num
    warnings.warn(
        'Requested test_num {0} is outside the available range [0, {1}). '
        'Wrapping it to {2}.'.format(test_num, data_num, new_index)
    )
    test_num = new_index

if not (data_num == data_numA == data_numB):
    raise ValueError('Inconsistent dataset sizes found in {0}.'.format(hdf5name))

data_train = fileName[dataset_name][...]
data_trainA = fileName[dataset_name + 'A']
data_trainB = fileName[dataset_name + 'B']

image = data_train[test_num,:,:,0]/np.linalg.norm(data_train[test_num,:,:,0].reshape(-1), 2)
if data_train.shape[-1] > 1:
    image_mult = data_train[test_num,:,:,1]/np.linalg.norm(data_train[test_num,:,:,1].reshape(-1), 2)
else:
    image_mult = np.zeros(image.shape)

imageA = data_trainA[test_num,:,:,0]/np.linalg.norm(data_trainA[test_num,:,:,0].reshape(-1), 2)
imageA_mult = data_trainA[test_num,:,:,1]/np.linalg.norm(data_trainA[test_num,:,:,1].reshape(-1), 2)
imageB = data_trainB[test_num,:,:, 0]/np.linalg.norm(data_trainB[test_num,:,:, 0].reshape(-1), 2)
imageB_mult = data_trainB[test_num,:,:, 1]/np.linalg.norm(data_trainB[test_num,:,:, 1].reshape(-1), 2)

# image = data_train[test_num,:,:]
# imageA = data_trainA[test_num,:,:]
# imageB = data_trainB[test_num,:,:]

Rec_SNR = -20.0* np.log(np.linalg.norm(np.absolute(image-imageA), 'fro')/np.linalg.norm(imageA, 'fro'))/np.log(10.0)
print('multiple elimination quality:\n')
print(Rec_SNR)

if data_train.shape[-1] > 1:
    Rec_SNR = -20.0* np.log(np.linalg.norm(np.absolute(image_mult-imageA_mult), 'fro')/np.linalg.norm(imageA, 'fro'))/np.log(10.0)
    print('multiple prediction quality:\n')
    print(Rec_SNR)

font = {'family' : 'sans-serif',
        'size'   : 7}
import matplotlib
matplotlib.rc('font', **font)


if save_dir == 'default':
    save_dir = os.path.join(hdf5path, 'figs')
    if os.path.isdir(save_dir)==False:
        os.mkdir(save_dir)
        print(save_dir)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(imageA, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('Without multiples')
plt.xlabel('Offset')
plt.ylabel('Time')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.7)
plt.savefig(os.path.join(save_dir, 'MutipleElimination-A.png'), format='png', bbox_inches='tight', dpi=400)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(imageB - imageA, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('Removed events - EPSI')
plt.xlabel('Offset')
plt.ylabel('Time')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.7)
plt.savefig(os.path.join(save_dir, 'MutipleElimination-error-conv.png'), format='png', bbox_inches='tight', dpi=400)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(imageB, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('With multiples')
plt.xlabel('Offset')
plt.ylabel('Time')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.7)
plt.savefig(os.path.join(save_dir, 'MutipleElimination-B.png'), format='png', bbox_inches='tight', dpi=400)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(image, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('Multiple elimination - ' + str(exp))
plt.xlabel('Offset')
plt.ylabel('Time')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.7)
plt.savefig(os.path.join(save_dir, 'MutipleElimination-result-NoTF.png'), format='png', bbox_inches='tight', dpi=400)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(imageB_mult, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('Unadapted multiples')
plt.xlabel('Offset')
plt.ylabel('Time')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.7)
plt.savefig(os.path.join(save_dir, 'PredictedMultiples-B.png'), format='png', bbox_inches='tight', dpi=400)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(imageA_mult, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('Adapted multiples - EPSI')
plt.xlabel('Offset')
plt.ylabel('Time')
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.axes().set_aspect(.7)
plt.savefig(os.path.join(save_dir, 'PredictedMultiples-A.png'), format='png', bbox_inches='tight', dpi=400)

plt.figure(dpi=200, figsize=(3, 7)); 
plt.imshow(imageB - image, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
plt.title('Removed events - ' + str(exp))
plt.xlabel('Offset')
plt.ylabel('Time')
plt.axes().set_aspect(.7)
plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.savefig(os.path.join(save_dir, 'MutipleElimination-error-NoTF.png'), format='png', bbox_inches='tight', dpi=400)

if data_train.shape[-1] > 1:

    plt.figure(dpi=200, figsize=(3, 7)); 
    plt.imshow(image_mult, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
    plt.title('Adapted multiples - CNN')
    plt.xlabel('Offset')
    plt.ylabel('Time')
    plt.axes().set_aspect(.7)
    plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
        right='False', left='False', labelleft='False')
    plt.savefig(os.path.join(save_dir, 'PredictedMultiples-CNN-B.png'), format='png', bbox_inches='tight', dpi=400)

    plt.figure(dpi=200, figsize=(3, 7)); 
    plt.imshow(imageA_mult - image_mult, vmin=-.005, vmax=.005, cmap="Greys", aspect='auto', interpolation="lanczos")
    plt.title('Error in adapting multiples')
    plt.xlabel('Offset')
    plt.ylabel('Time')
    plt.axes().set_aspect(.7)
    plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
        right='False', left='False', labelleft='False')
    plt.savefig(os.path.join(save_dir, 'Error-PredictedMultiples-CNN-B.png'), format='png', bbox_inches='tight', dpi=400)



font = {'family' : 'sans-serif',
        'size'   : 10}
import matplotlib
matplotlib.rc('font', **font)

plt.figure(dpi=200, figsize=(12, 4)); 
plt.plot(np.arange(175, 175+250, 1), imageA[175:175+250, test_num],linewidth=1.5, \
	label="w/o multiple", color='#00c292')
plt.plot(np.arange(175, 175+250, 1), image[175:175+250, test_num], linewidth=1.5, \
	label="Multiple elimination, " + str(exp), color='#f68818')
# plt.title('Multiple elimination w/ CNN vs multiple subtraction with Curvelet')
plt.xlabel('Time sample number')
plt.legend(loc='upper left')
plt.ylim(-.015, .015)
plt.tight_layout()
plt.axes().set_aspect(2000)
plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, 'MutipleElimination-result-trace.png'), format='png', bbox_inches='tight', dpi=400)


plt.figure(dpi=200, figsize=(12, 4)); 
plt.plot(np.arange(175, 175+250, 1), imageB[175:175+250, test_num], linewidth=1.5, \
	label="w/ multiples", color='#03a9f3')
plt.xlabel('Time sample number')
plt.legend(loc='upper left')
plt.ylim(-.015, .015)
plt.tight_layout()
plt.axes().set_aspect(2000)
plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, 'utipleElimination-B-trace.png'), format='png', bbox_inches='tight', dpi=400)


plt.figure(dpi=200, figsize=(12, 4)); 
plt.plot(np.arange(175, 175+250, 1), imageA_mult[175:175+250, test_num], linewidth=1.5, \
    label="Adapted multiples - EPSI", color='#d12121')
if data_train.shape[-1] > 1:
    plt.plot(np.arange(175, 175+250, 1), image_mult[175:175+250, test_num], linewidth=1.5, \
        label="Adapted multiples - CNN", color='#16cca7')
plt.xlabel('Time sample number')
plt.legend(loc='upper left')
plt.ylim(-.015, .015)
plt.tight_layout()
plt.axes().set_aspect(2000)
plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, 'PredictedMultiples-CNN-trace.png'), format='png', bbox_inches='tight', dpi=400)


plt.figure(dpi=200, figsize=(12, 4)); 
plt.plot(np.arange(175, 175+250, 1), imageB_mult[175:175+250, test_num], linewidth=1.5, \
    label="Unadapted multiples", color='#A569BD')
# plt.title('Predicted multiples')
plt.xlabel('Time sample number')
plt.legend(loc='upper left')
plt.ylim(-.006*2, .006*2)
plt.tight_layout()
plt.axes().set_aspect(2000)
plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, 'PredictedMultiples-B-trace.png'), format='png', bbox_inches='tight', dpi=400)




plt.figure(dpi=200, figsize=(12, 4)); 
plt.plot(np.arange(1024-250, 1024, 1), imageA[-250:, test_num],linewidth=1.5, \
    label="w/o multiple", color='#00c292')
plt.plot(np.arange(1024-250, 1024, 1), image[-250:, test_num], linewidth=1.5, \
    label="Multiple elimination, " + str(exp), color='#f68818')
# plt.title('Multiple elimination w/ CNN vs multiple subtraction with Curvelet')
plt.xlabel('Time sample number')
plt.legend(loc='upper left')
plt.ylim(-.0015, .0015)
plt.tight_layout()
plt.axes().set_aspect(20000)
plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, 'MutipleElimination-result-trace-end.png'), format='png', bbox_inches='tight', dpi=400)


plt.figure(dpi=200, figsize=(12, 4)); 
plt.plot(np.arange(1024-250, 1024, 1), imageB[-250:, test_num], linewidth=1.5, \
    label="w/ multiples", color='#03a9f3')
# plt.title('Data with multiples')
plt.xlabel('Time sample number')
plt.legend(loc='upper left')
plt.ylim(-.0015, .0015)
plt.tight_layout()
plt.axes().set_aspect(20000)
plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, 'utipleElimination-B-trace-end.png'), format='png', bbox_inches='tight', dpi=400)

plt.figure(dpi=200, figsize=(12, 4)); 
plt.plot(np.arange(1024-250, 1024, 1), imageA_mult[-250:, test_num], linewidth=1.5, \
    label="Adapted multiples - EPSI", color='#d12121')
if data_train.shape[-1] > 1:
    plt.plot(np.arange(1024-250, 1024, 1), image_mult[-250:, test_num], linewidth=1.5, \
        label="Adapted multiples - CNN", color='#16cca7')
# plt.title('Predicted multiples')
plt.xlabel('Time sample number')
plt.legend(loc='upper left')
plt.ylim(-.003, .003)
plt.tight_layout()
plt.axes().set_aspect(10000)
plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, 'PredictedMultiples-CNN-trace-end.png'), format='png', bbox_inches='tight', dpi=400)

plt.figure(dpi=200, figsize=(12, 4)); 
plt.plot(np.arange(1024-250, 1024, 1), imageB_mult[-250:, test_num], linewidth=1.5, \
    label="Unadapted multiples", color='#A569BD')
# plt.title('Predicted multiples')
plt.xlabel('Time sample number')
plt.legend(loc='upper left')
plt.ylim(-.003, .003)
plt.tight_layout()
plt.axes().set_aspect(4*2000)
plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    right='False', left='False', labelleft='False')
plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, 'PredictedMultiples-B-trace-end.png'), format='png', bbox_inches='tight', dpi=400)




# plt.show()

######################################################

