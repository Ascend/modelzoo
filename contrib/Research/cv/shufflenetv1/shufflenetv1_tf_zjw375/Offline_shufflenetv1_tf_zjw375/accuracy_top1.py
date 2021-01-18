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
import sys
import os
import numpy as np

if __name__ == "__main__":
    output_path = sys.argv[1]
    label_path = sys.argv[2]
    offset = 1
    output_num = 0
    check_num = 0
    files = os.listdir(output_path)
    files.sort()
    labels = os.listdir(label_path)
    labels.sort()
    #print(files)
    for file,label in zip(files, labels):
        if file.endswith(".bin"):
            output_num += 96 #batch_size
            tmp = np.fromfile(output_path+'/'+file, dtype='int32')
            inf_label = tmp + offset
            true_label = np.fromfile(label_path+'/'+label, dtype='int64')
            #print(inf_label.shape,true_label)
            res = np.equal(inf_label,true_label).astype(int)
            int_res = res.sum()
            check_num += int_res
    top1_accuarcy = check_num/output_num
    print("Totol pic num: %d, Top1 accuarcy: %.4f"%(output_num,top1_accuarcy))





