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
import os
import numpy as np
img_path = '/home/HwHiAiUser/ImageNet12/val'
label_path = '../ground_truth/label.txt'
output = './label_output'

f = open(label_path,'r')
labels = f.readlines()
print(labels)
files = os.listdir(img_path)
files.sort()
k = 0
batch_labels = []
for img_name, label in zip(files, labels):
    k += 1
    label = int(label.split('\n')[0])
    batch_labels.append(label)
    path = output+'/'+img_name + '.bin'
    if k == 96:
        batch_labels = np.array(batch_labels)
        batch_labels.tofile(path)
        print(batch_labels,batch_labels.dtype)
        k = 0
        batch_labels = []
