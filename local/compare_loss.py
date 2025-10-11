import torch 
import numpy as np 
import time 
import matplotlib 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import os 
import h5py 
from SeisModel import ricker_wavelet, absorbing_boundaries, stencil 
from NeuralNet import NeuralNet
from module import *
from utils import * 
from velocity_model import * 
from math import floor 
from torch import autograd 
import torchvision.utils as vutils 
from scipy import ndimage 
from tensorboardX import SummaryWriter 
torch.set_num_threads(8) 
np.random.seed(19) 
torch.manual_seed(19) 
 
 
net = NeuralNet() 

checkpoint_dir = './checkpoint/fit_model_noScheduler.pth'
checkpoint = torch.load(checkpoint_dir, map_location='cpu')
loss_log = checkpoint['loss_log']

plt.figure(figsize=(10, 4)); plt.semilogy(loss_log, linewidth=1.5, label="fit_model_noScheduler", color='#A569BD')


checkpoint_dir = './checkpoint/fit_model_mildScheduler.pth'
checkpoint = torch.load(checkpoint_dir, map_location='cpu')
loss_log = checkpoint['loss_log']

plt.semilogy(loss_log, linewidth=1.5, label="fit_model_mildScheduler", color='#3f3f3f')


checkpoint_dir = './checkpoint/fit_model.pth'
checkpoint = torch.load(checkpoint_dir, map_location='cpu')
loss_log = checkpoint['loss_log']

plt.figure(figsize=(10, 4)); plt.semilogy(loss_log, linewidth=1.5, label="fit_model", color='#d1c70e')
plt.title('training loss'); plt.xlabel('itr'); plt.ylabel('loss'); plt.grid(True) 
plt.show()