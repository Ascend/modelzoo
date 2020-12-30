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
data is a set of functions to save data
"""
from __future__ import absolute_import, division, print_function

import pickle

import h5py


def init_file(name):
    """
    :param name:
    :return:
    """
    try:
        model_file = h5py.File(name, 'r+')
        print("File is exsited")
    except BaseException:
        datatype = h5py.special_dtype(vlen=str)
        model_file = h5py.File(name, 'w')
        model_file.create_dataset("data_0", (5000, ), dtype=datatype)
        length = [0, 0]
        model_file.create_dataset("len", data=length)
        #print("create file")
    return model_file


def save_data(model_file, data):
    """
    :param model_file:
    :param data:
    :return:
    """
    index = model_file['len'][0]
    dataset_index = model_file['len'][1]

    if index >= (dataset_index + 1) * 5000:
        dataset_index += 1
        datatype = h5py.special_dtype(vlen=str)
        model_file.create_dataset("data_" + str(dataset_index), (5000, ), dtype=datatype)
        model_file['len'][1] = dataset_index

    model_file['data_' + str(dataset_index)][index % 5000] = pickle.dumps(data)
    index += 1
    model_file['len'][0] = index


def reset_data(model_file, datalen, dataset_index):
    """
    :param model_file:
    :param datalen:
    :param dataset_index:
    :return:
    """
    model_file['len'][0] = datalen
    model_file['len'][1] = dataset_index


def get_datalen(model_file):
    """
    :param model_file:
    :return:
    """
    return model_file['len'][0]


def get_data(model_file, index):
    """
    :param model_file:
    :param index:
    :return:
    """
    dataset_index = int(index / 5000)
    data = model_file['data_' + str(dataset_index)][index % 5000]
    return data.tostring()


def close_file(model_file):
    """
    :param model_file:
    :return:
    """
    model_file.close()
