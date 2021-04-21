# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
from collections import defaultdict

"""
Create txt files from Kaggle micorcontroller detection dataset


IMG_20181228_101826.jpg,800,600,Arduino_Nano,317,265,556,342

==> train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2

"""
class_names = "Arduino Nano,ESP8266,Raspberry Pi 3,Heltec ESP32 Lora".split(',')
class_map = dict(zip(class_names, range(1, len(class_names)+1)))
print(class_map)


def create_txt(lbl_file, data_type='train'):
    img_obj_mapping = defaultdict(list)
    with open(lbl_file) as f:
        for line in f:
            if not line.startswith("IMG"):
                continue
            img_name, _, _, class_name, *bbox = line.split(',')
            bbox = [cord.strip() for cord in bbox]
            file_name = os.path.join(data_type, img_name)
            class_id = class_map.get(class_name.replace('_', ' '))
            obj = ','.join(bbox) + f',{class_id}'
            img_obj_mapping[file_name].append(obj)

    txt_file_name = lbl_file.replace('.csv', '.txt')
    txt_lines = []
    for file in img_obj_mapping:
        txt_lines.append(' '.join([file, *img_obj_mapping.get(file)]))

    with open(txt_file_name, 'w') as f:
        f.write('\n'.join(txt_lines))


if __name__ == '__main__':
    create_txt('/data/dataset/MicrocontrollerDetection/train_labels.csv')
    create_txt('/data/dataset/MicrocontrollerDetection/test_labels.csv', 'test')
