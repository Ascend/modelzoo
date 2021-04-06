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

class DataSampler(object):
    def __init__(self):
        self.shape = [64, 64, 3]
        self.name = "lsun"
        # self.db_path = "/media/data/data/lsun/bedroom"
        self.db_path = "/home/student_3/iranb/yiqing/02-wgan/wgan/lsun/lsun_bedrooms_64_64_300000.npy"
        self.cur_batch_ptr = 0
        self.train_batch_ptr = 0
        self.train_size = 300000
        self.test_size = self.train_size
        self.all_data = np.load(self.db_path)
        # print(self.all_data.shape) # (300000, 64, 64, 3)
        self.batch_size = 64

    def __call__(self, batch_size):
        self.batch_size = batch_size
        m = self.all_data.shape[0]
        if self.train_batch_ptr + batch_size > self.train_size:
            self.train_batch_ptr = batch_size
            data = self.all_data[0:self.train_batch_ptr,:,:]
        else:
            data = self.all_data[self.train_batch_ptr:self.train_batch_ptr + batch_size,:,:,:]
        data = data.astype(np.float64)
        data *= 255.0/ data.max() 
        return np.reshape(data, [batch_size, -1])


    def data2img(self, data):
        rescaled = np.divide(data + 1.0, 2.0)
        return np.reshape(np.clip(rescaled, 0.0, 1.0), [data.shape[0]] + self.shape)

class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])
