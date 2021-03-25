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

def read_label(label_path):
    if label_path.endswith(".txt"):
        dict={}
        for line in open(label_path):
            line = line.replace("\r","").replace("\n","")
            pic_name= str(line.split(' ')[0])
            gt_label= int(line.split(' ')[1])
            dict[pic_name] = gt_label
        return dict
    else:
        print("Label file should be endswith *.txt, Please check your input!")
        return 0

if __name__ == "__main__":
    output_path = sys.argv[1]
    label_path = sys.argv[2]
    label_dict = read_label(label_path)
    offset = 0
    output_num = 0
    check_num = 0
    files = os.listdir(output_path)
    files.sort()
    for file in files:
        if file.endswith(".bin"):
            output_num += 1
            tmp = np.fromfile(output_path+'/'+file, dtype='float32')
            inf_label = int(np.argmax(tmp)) + offset
            try:
                pic_name = str(file.split("_output")[0])
                print("%s, inference label:%d, gt_label:%d"%(pic_name,inf_label,label_dict[pic_name]))
                if inf_label == label_dict[pic_name]:
                    #print("%s inference result Ok!"%pic_name)
                    check_num += 1
            except:
                print("Can't find %s in the label file: %s"%(pic_name,label_path))
    top1_accuarcy = check_num/output_num
    print("Totol pic num: %d, Top1 accuarcy: %.4f"%(output_num,top1_accuarcy))





