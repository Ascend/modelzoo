#!/usr/bin/env python
# coding: utf-8

"""
pwcnet_finetune.ipynb

PWC-Net model finetuning.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""
from __future__ import absolute_import, division, print_function
import sys
from copy import deepcopy

from dataset_base import _DEFAULT_DS_TUNE_OPTIONS
from dataset_flyingchairs import FlyingChairsDataset
from dataset_flyingthings3d import FlyingThings3DHalfResDataset
from dataset_mpisintel import MPISintelDataset

from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_FINETUNE_OPTIONS

import moxing as mox
import os
import shutil

if sys.platform.startswith("win"):
    _DATASET_ROOT = 'E:/datasets/'
else:
    _DATASET_ROOT = '/cache/'
_MPISINTEL_ROOT = _DATASET_ROOT + 'MPI-Sintel-complete'    

os.makedirs(_MPISINTEL_ROOT)
mox.file.copy_parallel('obs://pwcnet-lxm/MPI-Sintel-complete', _MPISINTEL_ROOT)
mox.file.copy_parallel('obs://pwcnet-lxm/pretrained', './pretrained')

gpu_devices = ['/device:CPU:0']
controller = '/device:CPU:0'

batch_size = 4

ds_opts = deepcopy(_DEFAULT_DS_TUNE_OPTIONS)

ds_opts['in_memory'] = False                          # Too many samples to keep in memory at once, so don't preload them
ds_opts['aug_type'] = 'heavy'                         # Apply all supported augmentations
ds_opts['batch_size'] = batch_size * len(gpu_devices) # Use a multiple of 8; here, 16 for dual-GPU mode (Titan X & 1080 Ti)
ds_opts['crop_preproc'] = (384, 768)                  # Crop to a smaller input size


ds_opts['type'] = 'clean'
ds = MPISintelDataset(mode='train_with_val', ds_root=_MPISINTEL_ROOT, options=ds_opts)

# Display dataset configuration
ds.print_config()

# Start from the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_FINETUNE_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = './pretrained/pwcnet.ckpt-595000'
nn_opts['ckpt_dir'] = './pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/'
nn_opts['batch_size'] = ds_opts['batch_size']
nn_opts['x_shape'] = [2, ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 3]
nn_opts['y_shape'] = [ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 2]
nn_opts['use_tf_data'] = True  # Use tf.data reader
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# Use the PWC-Net-small model in quarter-resolution mode
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

nn_opts['loss_fn'] = 'loss_multiscale'
nn_opts['q'] = 1.
nn_opts['epsilon'] = 0.

# Set the learning rate schedule. This schedule is for a single GPU using a batch size of 8.
# Below,we adjust the schedule to the size of the batch and the number of GPUs.
nn_opts['lr_policy'] = 'multisteps'
nn_opts['lr_boundaries'] = [40000, 60000, 80000, 100000]
nn_opts['lr_values'] = [1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07]
nn_opts['max_steps'] = 100000
nn_opts['display_step'] = 1
nn_opts['snapshot_step'] = 10
nn_opts['val_steps'] = 10

# Below,we adjust the schedule to the size of the batch and our number of GPUs (2).
nn_opts['max_steps'] = int(nn_opts['max_steps'] * 4 / ds_opts['batch_size'])
nn_opts['lr_boundaries'] = [int(boundary * 4 / ds_opts['batch_size']) for boundary in nn_opts['lr_boundaries']]

# Instantiate the model and display the model configuration
nn = ModelPWCNet(mode='train_with_val', options=nn_opts, dataset=ds)
nn.print_config()

# Train the model
nn.train()
