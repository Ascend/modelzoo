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

#!/usr/bin/env python
# -*- coding:utf-8 -*-

# dataset params.
min_area_not_validate = 20
train_data_path = ['./train_data.txt', ]
max_crop_tries = 50
min_crop_side_ratio = 0.6

# label params.
shrink_ratio = 0.3
min_text_size = 8

# model params.
checkpoint_path = './ckpts/'
pretrained_model_path = './pretrained_model/resnet_v1_50.ckpt' # None
restore = False # whether to restore from checkpoint
save_checkpoint_steps = 1000
save_summary_steps = 100

# training params.
num_readers = 16
batch_size_per_gpu = 16
learning_rate = 0.001
max_steps = 1000
moving_average_decay = 0.997
gpu_list = '1'
input_size = 512

# testing params.
text_rescale = 512


