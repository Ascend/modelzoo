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

import os
import numpy as np


class DataSampler(object):
    def __init__(self):
        self.shape = [64, 64, 3]
        self.name = "lsun"
        self.db_path = "/media/data/data/lsun/bedroom"
        self.db_files = os.listdir(self.db_path)
        self.cur_batch_ptr = 0
        self.cur_batch = self.load_new_data()
        self.train_batch_ptr = 0
        self.train_size = len(self.db_files) * 10000
        self.test_size = self.train_size

    def load_new_data(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                self.db_path, self.db_files[self.cur_batch_ptr])
        self.cur_batch_ptr += 1
        if self.cur_batch_ptr == len(self.db_files):
            self.cur_batch_ptr = 0
        return np.load(filename) * 2.0 - 1.0

    def __call__(self, batch_size):
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.cur_batch.shape[0]:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
            self.cur_batch = self.load_new_data()
        x = self.cur_batch[prev_batch_ptr:self.train_batch_ptr, :, :, :]
        return np.reshape(x, [batch_size, -1])

    def data2img(self, data):
        rescaled = np.divide(data + 1.0, 2.0)
        return np.reshape(np.clip(rescaled, 0.0, 1.0), [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])