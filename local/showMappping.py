import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse
np.random.seed(seed=19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--hdf5path', dest='hdf5path', type=str, default='/home/ec2-user/model', help='path')
parser.add_argument('--test_num', dest='test_num', type=int, default=10, help='test number')
parser.add_argument('--data_path', dest='data_path', type=str, default='/data/NelsonData/hdf5', help='path')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='default', help='saving directory')
parser.add_argument('--exp', dest='exp', type=str, default='exp-1', help='saving directory')
args = parser.parse_args()

hdf5path  = args.hdf5path
hdf5name = os.path.join(hdf5path, 'mapping_result.hdf5')
test_num = args.test_num
save_dir = args.save_dir
exp = args.exp
data_path = args.data_path


strReal = os.path.join(data_path, 'NelsonData.hdf5')
fileShots = h5py.File(strReal, 'r')

recon = fileShots["recon"]
mul = fileShots["mul"]
prim = fileShots["prim"]
srmemult = fileShots["srmemult_poor"]

strName = hdf5name
dataset_name = "result"
fileName = h5py.File(strName, 'r')
data_num = fileName[dataset_name].shape[0]
data_numA = fileName[dataset_name + 'A'].shape[0]
data_numB = fileName[dataset_name + 'B'].shape[0]

data_train = fileName[dataset_name][...]
data_trainA = fileName[dataset_name + 'A']
data_trainB = fileName[dataset_name + 'B']

image = data_train[test_num,:,:,0]*np.linalg.norm(prim[test_num+201, :, :].reshape(-1), np.inf)
if data_train.shape[-1] > 1:
    image_mult = data_train[test_num,:,:,1]*np.linalg.norm(mul[test_num+201, :, :].reshape(-1), np.inf)
else:
    image_mult = np.zeros(image.shape)

imageA = data_trainA[test_num,:,:,0]*np.linalg.norm(prim[test_num+201, :, :].reshape(-1), np.inf)
imageA_mult = data_trainA[test_num,:,:,1]*np.linalg.norm(mul[test_num+201, :, :].reshape(-1), np.inf)
imageB = data_trainB[test_num,:,:, 0]*np.linalg.norm(recon[test_num+201, :, :].reshape(-1), np.inf)
imageB_mult = data_trainB[test_num,:,:, 1]*np.linalg.norm(srmemult[test_num+201, :, :].reshape(-1), np.inf)

if data_train.shape[-1] > 1:
    image = imageB - image_mult
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

# from IPython import embed; embed()

dt = 4e-3
Fm = 1/(2*dt)

dx = 12.5
Km = 1/(2*dx)


Xstart= (imageA.shape[1]-(test_num+201)) * dx
X0 = (-(test_num+201) + 1) * dx



Tend= imageA.shape[0]*dt


if exp=='exp-2':

    plt.figure(dpi=150, figsize=(3, 7)); 
    plt.imshow(imageA, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
    plt.title('Multipile elimination - EPSI')
    plt.xlabel('Offset (m)')
    plt.ylabel('Time (s)')
    # plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    plt.axes().set_aspect(.7*25/4e-3)
    plt.savefig(os.path.join(save_dir, 'MutipleElimination-A.png'), format='png', bbox_inches='tight', dpi=150)

    plt.figure(dpi=150, figsize=(3, 7)); 
    plt.imshow(imageB - imageA, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
    plt.title('Predicted multiples - EPSI')
    plt.xlabel('Offset (m)')
    plt.ylabel('Time (s)')
    # plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    plt.axes().set_aspect(.7*25/4e-3)
    plt.savefig(os.path.join(save_dir, 'MutipleElimination-error-conv.png'), format='png', bbox_inches='tight', dpi=150)

    plt.figure(dpi=150, figsize=(3, 7)); 
    plt.imshow(imageB, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
    plt.title('Shot record with multiples')
    plt.xlabel('Offset (m)')
    plt.ylabel('Time (s)')
    # plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    plt.axes().set_aspect(.7*25/4e-3)
    plt.savefig(os.path.join(save_dir, 'MutipleElimination-B.png'), format='png', bbox_inches='tight', dpi=150)

    plt.figure(dpi=150, figsize=(3, 7)); 
    plt.imshow(imageB_mult, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
    plt.title('Unadapted multiples')
    plt.xlabel('Offset (m)')
    plt.ylabel('Time (s)')
    # plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    plt.axes().set_aspect(.7*25/4e-3)
    plt.savefig(os.path.join(save_dir, 'PredictedMultiples-B.png'), format='png', bbox_inches='tight', dpi=150)

    plt.figure(dpi=150, figsize=(3, 7)); 
    plt.imshow(imageA_mult, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
    plt.title('Predicted multiples - EPSI')
    plt.xlabel('Offset (m)')
    plt.ylabel('Time (s)')
    # plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    plt.axes().set_aspect(.7*25/4e-3)
    plt.savefig(os.path.join(save_dir, 'PredictedMultiples-A.png'), format='png', bbox_inches='tight', dpi=150)


plt.figure(dpi=150, figsize=(3, 7)); 
plt.imshow(imageB - image, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
plt.title('Predicted multiples - ' + str(exp))
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.axes().set_aspect(.7*25/4e-3)
# plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
#     right='False', left='False', labelleft='False')
plt.savefig(os.path.join(save_dir, exp, 'MutipleElimination-error-NoTF.png'), format='png', bbox_inches='tight', dpi=150)

plt.figure(dpi=150, figsize=(3, 7)); 
plt.imshow(image, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
plt.title('Multiple elimination - ' + str(exp))
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
# plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
#     right='False', left='False', labelleft='False')
plt.axes().set_aspect(.7*25/4e-3)
plt.savefig(os.path.join(save_dir, exp, 'MutipleElimination-result-NoTF.png'), format='png', bbox_inches='tight', dpi=150)


if data_train.shape[-1] > 1:

    plt.figure(dpi=150, figsize=(3, 7)); 
    plt.imshow(image_mult, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
    plt.title('Predicted multiples - exp-2')
    plt.xlabel('Offset (m)')
    plt.ylabel('Time (s)')
    plt.axes().set_aspect(.7*25/4e-3)
    # plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    plt.savefig(os.path.join(save_dir, exp, 'PredictedMultiples-CNN-B.png'), format='png', bbox_inches='tight', dpi=150)

    plt.figure(dpi=150, figsize=(3, 7)); 
    plt.imshow(imageA_mult - image_mult, vmin=-35, vmax=35, cmap="Greys", aspect='auto', extent=[X0,Xstart,Tend,0], interpolation="lanczos")
    plt.title('Error in adapting multiples')
    plt.xlabel('Offset (m)')
    plt.ylabel('Time (s)')
    plt.axes().set_aspect(.7*25/4e-3)
    # plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    plt.savefig(os.path.join(save_dir, exp, 'Error-PredictedMultiples-CNN-B.png'), format='png', bbox_inches='tight', dpi=150)



font = {'family' : 'sans-serif',
        'size'   : 12}
import matplotlib
matplotlib.rc('font', **font)

plt.figure(dpi=150, figsize=(12, 4)); 
plt.plot(np.arange(175, 175+250, 1)*dt, imageA[175:175+250, test_num+201 + 12],linewidth=1.5, \
    label="Multiple elimination - EPSI", color='#00c292')
plt.plot(np.arange(175, 175+250, 1)*dt, image[175:175+250, test_num+201 + 12], linewidth=1.5, \
    label="Multiple elimination, " + str(exp), color='#f68818')
# plt.title('Multiple elimination w/ CNN vs multiple subtraction with Curvelet')
plt.xlabel('Time (s)')
plt.legend(loc='upper right')
plt.ylim(-80/2.5, 80/2.5)
plt.tight_layout()
plt.axes().set_aspect(.0012*2.5)
# plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
#     right='False', left='False', labelleft='False')
# plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, exp, 'MutipleElimination-result-trace.png'), format='png', bbox_inches='tight', dpi=150)

if exp=='exp-2':

    # plt.figure(dpi=150, figsize=(12, 4)); 
    # plt.plot(np.arange(175, 175+250, 1)*dt, imageB[175:175+250, test_num+201 + 12], linewidth=1.5, \
    #     label="w/ multiples", color='#03a9f3')
    # plt.xlabel('Time (s)')
    # plt.legend(loc='upper right')
    # plt.ylim(-80/2.5, 80/2.5)
    # plt.tight_layout()
    # plt.axes().set_aspect(.0012*2.5)
    # # plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    # #     right='False', left='False', labelleft='False')
    # # plt.grid(axis='y', linestyle='-')
    # plt.savefig(os.path.join(save_dir, 'utipleElimination-B-trace.png'), format='png', bbox_inches='tight', dpi=150)

    plt.figure(dpi=150, figsize=(12, 4)); 
    plt.plot(np.arange(175, 175+250, 1)*dt, imageB_mult[175:175+250, test_num+201 + 12], linewidth=1.5, \
        label="Unadapted multiples", color='#A569BD')
    # plt.title('Predicted multiples')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right')
    plt.ylim(-80/1.45/2, 80/1.45/2)
    plt.tight_layout()
    plt.axes().set_aspect(.0012*1.45*2)
    # plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    # plt.grid(axis='y', linestyle='-')
    plt.savefig(os.path.join(save_dir,'PredictedMultiples-B-trace.png'), format='png', bbox_inches='tight', dpi=150)


plt.figure(dpi=150, figsize=(12, 4)); 
plt.plot(np.arange(175, 175+250, 1)*dt, imageA_mult[175:175+250, test_num+201 + 12], linewidth=1.5, \
    label="Predicted multiples - EPSI", color='#3f3f3f')
if data_train.shape[-1] > 1:
    plt.plot(np.arange(175, 175+250, 1)*dt, image_mult[175:175+250, test_num+201 + 12], linewidth=1.5, \
        label="Adapted multiples - CNN", color='#d1c70e')
plt.xlabel('Time (s)')
plt.legend(loc='upper right')
plt.ylim(-80/1.45/2, 80/1.45/2)
plt.tight_layout()
plt.axes().set_aspect(.0012*1.45*2)
# plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
#     right='False', left='False', labelleft='False')
# plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, exp, 'PredictedMultiples-CNN-trace.png'), format='png', bbox_inches='tight', dpi=150)



# from IPython import embed; embed()



plt.figure(dpi=150, figsize=(12, 4)); 
plt.plot(np.arange(1024-250, 1024, 1)*dt, imageA[-250:, test_num+201 + 12],linewidth=1.5, \
    label="Multiple elimination - EPSI", color='#4057e8')
plt.plot(np.arange(1024-250, 1024, 1)*dt, image[-250:, test_num+201 + 12], linewidth=1.5, \
    label="Multiple elimination, " + str(exp), color='#c6132e')
# plt.title('Multiple elimination w/ CNN vs multiple subtraction with Curvelet')
plt.xlabel('Time (s)')
plt.legend(loc='upper right')
plt.ylim(-16, 16)
plt.tight_layout()
plt.axes().set_aspect(.012/2)
# plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
#     right='False', left='False', labelleft='False')
# plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, exp, 'MutipleElimination-result-trace-end.png'), format='png', bbox_inches='tight', dpi=150)


if exp=='exp-2':
    # plt.figure(dpi=150, figsize=(12, 4)); 
    # plt.plot(np.arange(1024-250, 1024, 1)*dt, imageB[-250:, test_num+201 + 12], linewidth=1.5, \
    #     label="w/ multiples", color='#03a9f3')
    # # plt.title('Data with multiples')
    # plt.xlabel('Time (s)')
    # plt.legend(loc='upper right')
    # plt.ylim(-16, 16)
    # plt.tight_layout()
    # plt.axes().set_aspect(.012/2)
    # # plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    # #     right='False', left='False', labelleft='False')
    # # plt.grid(axis='y', linestyle='-')
    # plt.savefig(os.path.join(save_dir, 'utipleElimination-B-trace-end.png'), format='png', bbox_inches='tight', dpi=150)


    plt.figure(dpi=150, figsize=(12, 4)); 
    plt.plot(np.arange(1024-250, 1024, 1)*dt, imageB_mult[-250:, test_num+201 + 12], linewidth=1.5, \
        label="Unadapted multiples", color='#A569BD')
    # plt.title('Predicted multiples')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right')
    plt.ylim(-16/.7, 16/.7)
    plt.tight_layout()
    plt.axes().set_aspect(.012/2*0.7)
    # plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
    #     right='False', left='False', labelleft='False')
    # plt.grid(axis='y', linestyle='-')
    plt.savefig(os.path.join(save_dir, 'PredictedMultiples-B-trace-end.png'), format='png', bbox_inches='tight', dpi=150)

plt.figure(dpi=150, figsize=(12, 4)); 
plt.plot(np.arange(1024-250, 1024, 1)*dt, imageA_mult[-250:, test_num+201 + 12], linewidth=1.5, \
    label="Adapted multiples - EPSI", color='#3f3f3f')
if data_train.shape[-1] > 1:
    plt.plot(np.arange(1024-250, 1024, 1)*dt, image_mult[-250:, test_num+201 + 12], linewidth=1.5, \
        label="Adapted multiples - CNN", color='#d1c70e')
# plt.title('Predicted multiples')
plt.xlabel('Time (s)')
plt.legend(loc='upper right')
plt.ylim(-16/.7, 16/.7)
plt.tight_layout()
plt.axes().set_aspect(.012/2*0.7)
# plt.tick_params(axis='y', which='both', bottom='False', top='False', labelbottom='False', \
#     right='False', left='False', labelleft='False')
# plt.grid(axis='y', linestyle='-')
plt.savefig(os.path.join(save_dir, exp, 'PredictedMultiples-CNN-trace-end.png'), format='png', bbox_inches='tight', dpi=150)






