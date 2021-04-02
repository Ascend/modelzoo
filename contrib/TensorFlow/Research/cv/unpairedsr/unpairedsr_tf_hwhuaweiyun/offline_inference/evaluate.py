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
import math
import os

import numpy as np


def get_psnr(img1, img2):
    """ Calculate PSNR for a pair of images

    Args:
        img1: An `ndarray` of shape [H, W, C], and its values has been normalized to the interval [0, 1]
        img2: An `ndarray` with the same shape as img1, and its values has been normalized to the interval [0, 1]

    Returns:
        PSNR of the given image pair.
    """
    mse = np.mean((img1 - img2) ** 2)
    psnr = 10 * math.log10(1. / mse)
    return psnr


def main():
    """

    Calculate the PSNR on the data set.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_hr', type=str, default='bin_data/hr', help='Directory for real hr data')
    parser.add_argument('--fake_hr', type=str, default='bin_data/fake_hr', help='Directory for predicted hr data')
    args = parser.parse_args()

    cnt, psnr_sum = 0, 0
    for real_hr_file in os.listdir(args.real_hr):
        file_id = real_hr_file.split('.')[0]
        fake_hr_file = '%s_output_0.bin' % file_id
        real_hr = np.fromfile(os.path.join(args.real_hr, real_hr_file), dtype=np.float32).reshape([64, 64, 3])
        fake_hr = np.fromfile(os.path.join(args.fake_hr, fake_hr_file), dtype=np.float32).reshape([64, 64, 3])
        psnr_sum += get_psnr(real_hr, fake_hr)
        cnt += 1

    print(psnr_sum / cnt)


if __name__ == '__main__':
    main()
