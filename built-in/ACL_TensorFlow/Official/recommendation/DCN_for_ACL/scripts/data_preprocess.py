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
from numpy.lib.stride_tricks import _broadcast_shape
from data_process.criteo import create_criteo_dataset
import numpy as np
import sys
import os

file = sys.argv[1]                #'./dataset/Criteo/train.txt'
os.makedirs('input_x', exist_ok=True)
os.makedirs('labels', exist_ok=True)

read_part = True
sample_num = 5000000
test_size = 0.2
embed_dim = 8

batch_size = 4000
# ========================== Create dataset =======================
feature_columns, train, test = create_criteo_dataset(file=file,
                                                        embed_dim=embed_dim,
                                                        read_part=read_part,
                                                        sample_num=sample_num,
                                                        test_size=test_size)
test_X, test_Y = test
length = len(test_X)
print(length)
for i in range(length // batch_size):
    index = i * batch_size
    batch_X = list()
    batch_Y = list()
    for j in range(batch_size):
        batch_X.append(test_X[index + j])
        batch_Y.append(test_Y[index + j])
    batch_X = np.array(batch_X)
    batch_Y = np.array(batch_Y)
    print(len(batch_X))
    print(batch_X.dtype)
    batch_X.tofile("{}/batch{}_X.bin".format(os.path.join('./','input_x'),i))
    batch_Y.tofile("{}/batch{}_Y.bin".format(os.path.join('./','labels'),i))

