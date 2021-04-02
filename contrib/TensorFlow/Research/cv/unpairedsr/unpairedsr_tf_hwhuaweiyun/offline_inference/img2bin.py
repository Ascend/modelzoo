# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse
import glob
import os

import numpy as np


def main():
    """

    Convert the data set to bin files to adapt to msame.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/test', help='Directory for original test data')
    parser.add_argument('--output', type=str, default='./bin_data', help='Directory for bin data')
    args = parser.parse_args()

    npy_paths = glob.glob(os.path.join(args.input, '*.npy'))
    for subdir in ['lr', 'hr']:
        if not os.path.isdir(os.path.join(args.output, subdir)):
            os.makedirs(os.path.join(args.output, subdir))
    sample_id = 0
    for npy_path in npy_paths:
        data = np.load(npy_path, allow_pickle=True).item()
        lr_data = np.clip(data['sample'].astype(np.float32) / 255, 0., 1.)
        hr_data = np.clip(data['label'].astype(np.float32) / 255, 0., 1.)
        for lr_img, hr_img in zip(lr_data, hr_data):
            sample_id += 1
            lr_img.tofile(os.path.join(args.output, 'lr', '%04d.bin' % sample_id))
            hr_img.tofile(os.path.join(args.output, 'hr', '%04d.bin' % sample_id))


if __name__ == '__main__':
    main()
