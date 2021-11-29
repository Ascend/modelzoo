#!/usr/bin/env python
# coding=utf-8

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
"""LeNet pre process"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import argparse

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def get_config(args):
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument('--data_path', default='MNIST',
                        help='training input data path.')
    parser.add_argument('--output_path', default='input_files',
                        help='prepocess result path.')

    parsed_args, unknown_args = parser.parse_known_args(args)

    return parsed_args


def input_data_get(data_path, out_path):
    if (not os.path.exists(out_path)):
        os.mkdir(out_path)

    image_path = os.path.join(out_path, 'images')
    if (not os.path.exists(image_path)):
        os.mkdir(image_path)

    label_path = os.path.join(out_path, 'labels')
    if (not os.path.exists(label_path)):
        os.mkdir(label_path)

    mnist = input_data.read_data_sets(data_path, one_hot=True)
    test_nums = mnist.test.num_examples
    test_datas = mnist.test.images
    test_labels = mnist.test.labels
    for index in range(int(test_nums)):
        image_list = np.array([test_datas[index]]).reshape([784])
        image_list.tofile(image_path + "/input_" + str(index) + ".bin")

        label_list = np.array([test_labels[index]]).reshape([10])
        label_list.tofile(label_path + "/input_" + str(index) + ".bin")


def main():
    args = get_config(sys.argv[1:])
    input_data_get(args.data_path, args.out_path)


if __name__ == "__main__":
    main()
