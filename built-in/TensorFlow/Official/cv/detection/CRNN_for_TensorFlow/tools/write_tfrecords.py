#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
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
# @Time    : 19-3-13 涓嬪崍1:31
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : write_tfrecords.py
# @IDE: PyCharm
"""
Write tfrecords tools
"""
import argparse
import os
import os.path as ops
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)

from data_provider import shadownet_data_feed_pipline
#import shadownet_data_feed_pipline


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='The origin synth90k dataset_dir')
    parser.add_argument('-s', '--save_dir', type=str, help='The generated tfrecords save dir')
    parser.add_argument('-c', '--char_dict_path', type=str, default=None,
                        help='The char dict file path. If it is None the char dict will be'
                             'generated automatically in folder data/char_dict')
    parser.add_argument('-o', '--ord_map_dict_path', type=str, default=None,
                        help='The ord map dict file path. If it is None the ord map dict will be'
                             'generated automatically in folder data/char_dict')

    return parser.parse_args()


def write_tfrecords(dataset_dir, char_dict_path, ord_map_dict_path, save_dir):
    """
    Write tensorflow records for training , testing and validation
    :param dataset_dir: the root dir of crnn dataset
    :param char_dict_path: json file path which contains the map relation
    between ord value and single character
    :param ord_map_dict_path: json file path which contains the map relation
    between int index value and char ord value
    :param save_dir: the root dir of tensorflow records to write into
    :return:
    """
    assert ops.exists(dataset_dir), '{:s} not exist'.format(dataset_dir)

    os.makedirs(save_dir, exist_ok=True)

    # test crnn data producer
    producer = shadownet_data_feed_pipline.CrnnDataProducer(
        dataset_dir=dataset_dir,
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
        writer_process_nums=8
    )

    producer.generate_tfrecords(
        save_dir=save_dir
    )


if __name__ == '__main__':
    """
    generate tfrecords
    """
    args = init_args()

    write_tfrecords(
        dataset_dir=args.dataset_dir,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        save_dir=args.save_dir
    )
