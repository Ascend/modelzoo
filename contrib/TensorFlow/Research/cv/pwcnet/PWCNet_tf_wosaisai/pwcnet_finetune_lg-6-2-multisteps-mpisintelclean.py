# Copyright 2017 Phil Ferriere. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import os
import shutil

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--iterations', type=int, default=100000,
                    help='the training iterations')
parser.add_argument('--display', type=int, default=100,
                    help='the interval steps to display loss')
parser.add_argument('--save_path', type=str, default='./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/',
                    help='the path to save checkpoint')
parser.add_argument('--batch_size', type=int, default=4,
                    help='the batch size')
parser.add_argument('--dataset', type=str, default='./dataset/',
                    help='the path of dataset')
parser.add_argument('--pretrained', type=str, default='./pretrained/pwcnet.ckpt-176000',
                    help='the path of pretrained model')
parser.add_argument('--robust', type=bool, default=False,
                    help='use robust loss')
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_url', type=str)

args = parser.parse_args()

_DATASET_ROOT = args.dataset
_MPISINTEL_ROOT = _DATASET_ROOT + 'MPI-Sintel-complete'

gpu_devices = ['/device:CPU:0']
controller = '/device:CPU:0'

batch_size = args.batch_size

ds_opts = deepcopy(_DEFAULT_DS_TUNE_OPTIONS)

ds_opts['in_memory'] = False  # Too many samples to keep in memory at once, so don't preload them
ds_opts['aug_type'] = 'heavy'  # Apply all supported augmentations
ds_opts['batch_size'] = batch_size * len(gpu_devices)  # Use a multiple of 8; here, 16 for dual-GPU mode (Titan X & 1080 Ti)
ds_opts['crop_preproc'] = (384, 768)  # Crop to a smaller input size

ds_opts['type'] = 'clean'
ds = MPISintelDataset(mode='train_noval', ds_root=_MPISINTEL_ROOT, options=ds_opts)

# Display dataset configuration
ds.print_config()

# Start from the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_FINETUNE_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = args.pretrained
nn_opts['ckpt_dir'] = args.save_path
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

if args.robust:
    nn_opts['loss_fn'] = 'loss_robust'
    nn_opts['q'] = 0.4
    nn_opts['epsilon'] = 0.01
else:
    nn_opts['loss_fn'] = 'loss_multiscale'
    nn_opts['q'] = 1.
    nn_opts['epsilon'] = 0.

# Set the learning rate schedule. This schedule is for a single GPU using a batch size of 8.
# Below,we adjust the schedule to the size of the batch and the number of GPUs.
nn_opts['lr_policy'] = 'multisteps'
nn_opts['lr_boundaries'] = [60000, 60000, 80000, 100000]
# nn_opts['lr_values'] = [1e-4, 5e-05, 2.5e-05, 1.25e-05, 6.25e-06]
nn_opts['lr_values'] = [1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07]
nn_opts['max_steps'] = args.iterations
nn_opts['display_step'] = args.display
nn_opts['val_step'] = 1000
nn_opts['tb_val_imgs'] = None

# Below,we adjust the schedule to the size of the batch and our number of GPUs (2).
nn_opts['max_steps'] = int(nn_opts['max_steps'] * 4 / ds_opts['batch_size'])
nn_opts['lr_boundaries'] = [int(boundary * 4 / ds_opts['batch_size']) for boundary in nn_opts['lr_boundaries']]

# Instantiate the model and display the model configuration
nn = ModelPWCNet(mode='train_noval', options=nn_opts, dataset=ds)
nn.print_config()

# Train the model
nn.train()
