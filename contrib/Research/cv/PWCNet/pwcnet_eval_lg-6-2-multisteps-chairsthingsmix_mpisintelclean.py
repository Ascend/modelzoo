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
from dataset_mpisintel import MPISintelDataset
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_VAL_OPTIONS
from visualize import display_img_pairs_w_flows
import moxing as mox
import os
import shutil

if sys.platform.startswith("win"):
    _DATASET_ROOT = 'E:/datasets/'
else:
    _DATASET_ROOT = '/cache/'
_MPISINTEL_ROOT = _DATASET_ROOT + 'MPI-Sintel-complete'

os.makedirs(_MPISINTEL_ROOT)
mox.file.copy_parallel('obs://pwcnet-final/MPI-Sintel-complete', _MPISINTEL_ROOT)
mox.file.copy_parallel('obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned',
                       './pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned')

gpu_devices = ['/device:CPU:0']
controller = '/device:CPU:0'

# More options...
mode = 'val_notrain'  # 'val_notrain'            # We're doing evaluation using the entire dataset for evaluation
num_samples = 10  # Number of samples for error analysis
ckpt_path = './pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000'  # Model to eval

# Load the dataset in evaluation mode, starting with the default evaluation options
ds_opts = deepcopy(_DEFAULT_DS_VAL_OPTIONS)
ds_opts['type'] = 'clean'
ds = MPISintelDataset(mode=mode, ds_root=_MPISINTEL_ROOT, options=ds_opts)

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

# We're evaluating the PWC-Net-large model in quarter-resolution mode
# That is, with a 6 level pyramid, and uspampling of level 2 by 4 in each dimension as the final flow prediction
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
nn_opts['adapt_info'] = (1, 436, 1024, 2)

# Instantiate the model in evaluation mode and display the model configuration
nn = ModelPWCNet(mode=mode, options=nn_opts, dataset=ds)
nn.print_config()

# ## Evaluate the model
avg_metric, avg_duration, df = nn.eval(metric_name='EPE', save_preds=True)


print(f'Average EPE={avg_metric:.2f}, mean inference time={avg_duration * 1000.:.2f}ms')
