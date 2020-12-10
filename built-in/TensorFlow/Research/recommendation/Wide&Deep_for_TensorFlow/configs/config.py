# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
BASE_DIR = './'

num_gpu = 8
num_inputs = 39
num_features = 200000


batch_size = 16000
multi_hot_flags = [False]
multi_hot_len = 1
###
#n_epoches =50
#iterations_per_loop = 10
n_epoches = 1
iterations_per_loop = 1
#one_step = 50/iterations_per_loop # for one step debug
one_step = 0
line_per_sample = 1000

#record_path = '/data/tf_record'
record_path = '/autotest/CI_daily/ModelZoo_WideDeep_TF/data/tf_record'
train_tag = 'train_part'
test_tag = 'test_part'
writer_path = './model/'
graph_path = './model/'

train_size = 41257636
test_size = 4582981


