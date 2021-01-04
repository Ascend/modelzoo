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
import sys
sys.path.append('..')

from dataset_base import _DEFAULT_DS_VAL_OPTIONS
from dataset_mpisintel import MPISintelDataset
from copy import deepcopy
import os
import argparse
import numpy as np
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default='./datasets/')
parser.add_argument('-o', '--output', default='offline_infer/Bin', help='output path.')

args = parser.parse_args()


def adapt_x(x, pyr_lvls=6):
    """Preprocess the input samples to adapt them to the network's requirements
    Here, x, is the actual data, not the x TF tensor.
    Args:
        x: input samples in list[(2,H,W,3)] or (N,2,H,W,3) np array form
    Returns:
        Samples ready to be given to the network (w. same shape as x)
        Also, return adaptation info in (N,2,H,W,3) format
    """
    # Ensure we're dealing with RGB image pairs
    assert (isinstance(x, np.ndarray) or isinstance(x, list))
    if isinstance(x, np.ndarray):
        assert (len(x.shape) == 5)
        assert (x.shape[1] == 2 and x.shape[4] == 3)
    else:
        assert (len(x[0].shape) == 4)
        assert (x[0].shape[0] == 2 or x[0].shape[3] == 3)

    # Bring image range from 0..255 to 0..1 and use floats (also, list[(2,H,W,3)] -> (batch_size,2,H,W,3))
    x_adapt = np.array(x, dtype=np.float32) if isinstance(x, list) else x.astype(np.float32)
    x_adapt /= 255.

    # Make sure the image dimensions are multiples of 2**pyramid_levels, pad them if they're not
    _, pad_h = divmod(x_adapt.shape[2], 2 ** pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(x_adapt.shape[3], 2 ** pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    x_adapt_info = None
    if pad_h != 0 or pad_w != 0:
        padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)]
        x_adapt_info = x_adapt.shape  # Save original shape
        x_adapt = np.pad(x_adapt, padding, mode='constant', constant_values=0.)

    return x_adapt, x_adapt_info


def adapt_y(y, pyr_lvls=6):
    """Preprocess the labels to adapt them to the loss computation requirements of the network
    Here, y, is the actual data, not the y TF tensor.
    Args:
        y: labels in list[(H,W,2)] or (N,H,W,2) np array form
    Returns:
        Labels ready to be used by the network's loss function (w. same shape as y)
        Also, return adaptation info in (N,H,W,2) format
    """
    # Ensure we're dealing with u,v flows
    assert (isinstance(y, np.ndarray) or isinstance(y, list))
    if isinstance(y, np.ndarray):
        assert (len(y.shape) == 4)
        assert (y.shape[3] == 2)
    else:
        assert (len(y[0].shape) == 3)
        assert (y[0].shape[2] == 2)

    y_adapt = np.array(y, dtype=np.float32) if isinstance(y, list) else y  # list[(H,W,2)] -> (batch_size,H,W,2)

    # Make sure the flow dimensions are multiples of 2**pyramid_levels, pad them if they're not
    _, pad_h = divmod(y.shape[1], 2 ** pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(y.shape[2], 2 ** pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    y_adapt_info = None
    if pad_h != 0 or pad_w != 0:
        padding = [(0, 0), (0, pad_h), (0, pad_w), (0, 0)]
        y_adapt_info = y_adapt.shape  # Save original shape
        y_adapt = np.pad(y_adapt, padding, mode='constant', constant_values=0.)

    return y_adapt, y_adapt_info


if __name__ == '__main__':

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    _DATASET_ROOT = args.dataset
    _MPISINTEL_ROOT = os.path.join(_DATASET_ROOT, 'MPI-Sintel-complete')

    mode = 'val_notrain'  # 'val_notrain'
    # Load the dataset in evaluation mode, starting with the default evaluation options
    ds_opts = deepcopy(_DEFAULT_DS_VAL_OPTIONS)
    ds_opts['type'] = 'clean'
    ds = MPISintelDataset(mode=mode, ds_root=_MPISINTEL_ROOT, options=ds_opts)

    for idx, sample in enumerate(tqdm(ds._img_val_path)):
        idx = np.array([idx])
        image, gt_label, ID = ds.get_samples(idx=idx, split='val', as_list=False, simple_IDs=True)
        image, _ = adapt_x(image)
        gt_label, _ = adapt_y(gt_label)

        out_path_image = os.path.join(args.output, 'image', '{0}-{1}.bin'.format(os.path.split(ID[0])[0], os.path.split(ID[0])[1]))
        out_path_gt = os.path.join(args.output,  'gt', '{0}-{1}.bin'.format(os.path.split(ID[0])[0], os.path.split(ID[0])[1]))

        if not os.path.exists(os.path.join(args.output, 'image')):
            os.makedirs(os.path.join(args.output, 'image'))
        if not os.path.exists(os.path.join(args.output, 'gt')):
            os.makedirs(os.path.join(args.output, 'gt'))
        image.tofile(out_path_image)
        gt_label.tofile(out_path_gt)
