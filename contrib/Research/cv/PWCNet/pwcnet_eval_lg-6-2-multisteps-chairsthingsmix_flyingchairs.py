#!/usr/bin/env python
# coding: utf-8
"""
pwcnet_eval.ipynb

PWC-Net model evaluation.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""
from __future__ import absolute_import, division, print_function
import sys
from copy import deepcopy
import pandas as pd

from dataset_base import _DEFAULT_DS_VAL_OPTIONS
from dataset_flyingchairs import FlyingChairsDataset
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_VAL_OPTIONS
from visualize import display_img_pairs_w_flows
import moxing as mox
import os
import shutil

if sys.platform.startswith("win"):
    _DATASET_ROOT = 'E:/datasets/'
else:
    _DATASET_ROOT = '/cache/'
_FLYINGCHAIRS_ROOT = _DATASET_ROOT + 'FlyingChairs'

os.makedirs(_FLYINGCHAIRS_ROOT)
mox.file.copy_parallel('obs://pwcnet-lxm/FlyingChairs', _FLYINGCHAIRS_ROOT)
mox.file.copy_parallel('obs://pwcnet-lxm/pretrained', './pretrained')

gpu_devices = ['/device:GPU:0']  # We're doing the evaluation on a single GPU
controller = '/device:GPU:0'

# More options...
mode = 'val'  # We're doing the evaluation on the validation split of the dataset
num_samples = 10  # Number of samples for error analysis
ckpt_path = './pretrained/pwcnet.ckpt-595000'  # Model to eval

# Load the dataset in evaluation mode, starting with the default evaluation options
ds_opts = deepcopy(_DEFAULT_DS_VAL_OPTIONS)
ds = FlyingChairsDataset(mode=mode, ds_root=_FLYINGCHAIRS_ROOT, options=ds_opts)

# Display dataset configuration
ds.print_config()

# Configure the model for evaluation, starting with the default evaluation options
nn_opts = deepcopy(_DEFAULT_PWCNET_VAL_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = ckpt_path
nn_opts['batch_size'] = 1  # Setting this to 1 leads to more accurate evaluations of the processing time
nn_opts['use_tf_data'] = False  # Don't use tf.data reader for this simple task
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller  # Evaluate on CPU or GPU?

nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# Instantiate the model in evaluation mode and display the model configuration
nn = ModelPWCNet(mode=mode, options=nn_opts, dataset=ds)
nn.print_config()

avg_metric, avg_duration, df = nn.eval(metric_name='EPE', save_preds=True)
print(f'Average EPE={avg_metric:.2f}, mean inference time={avg_duration * 1000.:.2f}ms')
