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
import numpy as np 
import os 
import cv2


def calc_mad(output, mask):
    mad = np.mean(np.abs(mask - output))
    return mad

output_shape = [1, 256, 256, 2]
gt_shape = [1, 256, 256, 1]
output_dir = 'Bin/test/outputs'
gt_dir = 'Bin/test/masks'


names_output = sorted(os.listdir(output_dir))
names_gt = sorted(os.listdir(gt_dir))

total_mad = 0

for i in range(len(names_gt)):
    filename_gt = os.path.join(gt_dir, names_gt[i])
    filename_output = os.path.join(output_dir, names_gt[i].split('.')[0] + '_output_0.bin')
    # print(filename_output, filename_gt)
    output = np.fromfile(filename_output, dtype=np.float32).reshape(output_shape)[:, :, :, -1:]
    gt = np.fromfile(filename_gt, dtype=np.float32).reshape(gt_shape)
    total_mad += calc_mad(output, gt)

print(len(names_gt))
print(total_mad / len(names_gt))

