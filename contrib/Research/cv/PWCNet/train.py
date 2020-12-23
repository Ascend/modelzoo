from __future__ import absolute_import, division, print_function
import cv2
import tensorflow as tf
import numpy as np
import tensorflow as tf
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


import moxing as mox
import os
import shutil
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--iterations', type=int, default=200000, 
                    help='the training iterations')
parser.add_argument('--display', type=int, default=1,
                    help='the interval steps to display loss')
        

args = parser.parse_args()



# shutil.rmtree('../../FlyingChairs_subset')
os.makedirs('./FlyingChairs/')
mox.file.copy_parallel('obs://pwcnet-lxm/FlyingChairs', './FlyingChairs')


"""
pwcnet_train.ipynb

PWC-Net model training.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Tensorboard:
    [win] tensorboard --logdir=E:\\repos\\tf-optflow\\tfoptflow\\pwcnet-lg-6-2-cyclic-chairsthingsmix
    [ubu] tensorboard --logdir=/media/EDrive/repos/tf-optflow/tfoptflow/pwcnet-lg-6-2-cyclic-chairsthingsmix
"""

import sys
from copy import deepcopy

from dataset_base import _DEFAULT_DS_TRAIN_OPTIONS
from dataset_flyingchairs import FlyingChairsDataset
from dataset_flyingthings3d import FlyingThings3DHalfResDataset
from dataset_mixer import MixedDataset
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TRAIN_OPTIONS

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# TODO: You MUST set dataset_root to the correct path on your machine!
if sys.platform.startswith("win"):
    _DATASET_ROOT = 'E:/datasets/'
else:
    _DATASET_ROOT = './'
_FLYINGCHAIRS_ROOT = _DATASET_ROOT + 'FlyingChairs'
_FLYINGTHINGS3DHALFRES_ROOT = _DATASET_ROOT + 'FlyingThings3D_HalfRes'
    
# TODO: You MUST adjust the settings below based on the number of GPU(s) used for training
# Set controller device and devices
# A one-gpu setup would be something like controller='/device:GPU:0' and gpu_devices=['/device:GPU:0']
# Here, we use a dual-GPU setup, as shown below
gpu_devices = ['/device:CPU:0']
controller = '/device:CPU:0'

# TODO: You MUST adjust this setting below based on the amount of memory on your GPU(s)
# Batch size
batch_size = 8


# TODO: You MUST set the batch size based on the capabilities of your GPU(s) 
#  Load train dataset
ds_opts = deepcopy(_DEFAULT_DS_TRAIN_OPTIONS)
ds_opts['in_memory'] = False                          # Too many samples to keep in memory at once, so don't preload them
ds_opts['aug_type'] = 'heavy'                         # Apply all supported augmentations
ds_opts['batch_size'] = batch_size * len(gpu_devices) # Use a multiple of 8; here, 16 for dual-GPU mode (Titan X & 1080 Ti)
ds_opts['crop_preproc'] = (256, 448)                  # Crop to a smaller input size
ds1 = FlyingChairsDataset(mode='train_with_val', ds_root=_FLYINGCHAIRS_ROOT, options=ds_opts)
ds_opts['type'] = 'into_future'
# ds2 = FlyingThings3DHalfResDataset(mode='train_with_val', ds_root=_FLYINGTHINGS3DHALFRES_ROOT, options=ds_opts)
ds = MixedDataset(mode='train_with_val', datasets=[ds1], options=ds_opts)


# Display dataset configuration
ds.print_config()


# Start from the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_TRAIN_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_dir'] = './pwcnet-lg-6-2-cyclic-chairsthingsmix/'
nn_opts['batch_size'] = ds_opts['batch_size']
nn_opts['x_shape'] = [2, ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 3]
nn_opts['y_shape'] = [ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 2]
nn_opts['use_tf_data'] = True # Use tf.data reader
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# Use the PWC-Net-large model in quarter-resolution mode
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2


# Set the learning rate schedule. This schedule is for a single GPU using a batch size of 8.
# Below,we adjust the schedule to the size of the batch and the number of GPUs.
nn_opts['lr_policy'] = 'cyclic'
nn_opts['cyclic_lr_max'] = 5e-04 # Anything higher will generate NaNs
nn_opts['cyclic_lr_base'] = 1e-05
nn_opts['cyclic_lr_stepsize'] = 20000
nn_opts['max_steps'] = args.iterations
nn_opts['display_step'] = 1
# Below,we adjust the schedule to the size of the batch and our number of GPUs (2).
nn_opts['cyclic_lr_stepsize'] /= len(gpu_devices)
nn_opts['max_steps'] /= len(gpu_devices)
nn_opts['cyclic_lr_stepsize'] = int(nn_opts['cyclic_lr_stepsize'] / (float(ds_opts['batch_size']) / 8))
nn_opts['max_steps'] = int(nn_opts['max_steps'] / (float(ds_opts['batch_size']) / 8))


# Instantiate the model and display the model configuration
nn = ModelPWCNet(mode='train_with_val', options=nn_opts, dataset=ds)
nn.print_config()


# Train the model
nn.train()

