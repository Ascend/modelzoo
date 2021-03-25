# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
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
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
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

# -------------------------------------------------------------------------------
from preprocessing.siam_mask_dataset import DataSets
import numpy as np


def generator(data_set, batch_size=8, is_training=True):
    # get train data length and init train indices
    template_size = 127
    search_size = 255
    cls_size = 25
    dataset_size = data_set.__len__()
    if is_training:
        data_set.shuffle()
    current_idx = 0
    while True:
        template = np.zeros([batch_size, template_size, template_size, 3], dtype=np.float32)
        search = np.zeros([batch_size, search_size, search_size, 3], dtype=np.float32)
        label_cls = np.zeros([batch_size, cls_size, cls_size, 5], dtype=np.float32)
        label_loc = np.zeros([batch_size, cls_size, cls_size, 4, 5], dtype=np.float32)
        label_loc_weight = np.zeros([batch_size, cls_size, cls_size, 5], dtype=np.float32)
        label_mask = np.zeros([batch_size, search_size, search_size, 1], dtype=np.float32)
        label_mask_weight = np.zeros([batch_size, cls_size, cls_size, 1], dtype=np.float32)

        for i in range(batch_size):
            template_each, search_each, cls_each, delta_each, delta_weight_each, bbox_each \
                , mask_each, mask_weight_each = data_set.__getitem__(current_idx)
            template[i] = np.transpose(template_each, (1, 2, 0))
            search[i] = np.transpose(search_each, (1, 2, 0))
            label_cls[i] = np.transpose(cls_each, (1, 2, 0))
            label_loc[i] = np.transpose(delta_each, (2, 3, 0, 1))
            label_loc_weight[i] = np.transpose(delta_weight_each, (1, 2, 0))
            label_mask[i] = np.transpose(mask_each, (1, 2, 0))
            label_mask_weight[i] = np.transpose(mask_weight_each, (1, 2, 0))

            current_idx += 1
            if current_idx >= dataset_size:
                if is_training:
                    data_set.shuffle()
                current_idx = 0
        yield [template, search, label_cls, label_loc, label_loc_weight, label_mask, label_mask_weight]


def build_data_loader(cfg, epochs, batch_size):
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'], epochs)
    train_set.shuffle()
    return generator(train_set, batch_size, True)
