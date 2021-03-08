# -*- coding: UTF-8 -*-
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
import argparse


def getLabel(filename, label_map_path):
    label_list = []
    with open(label_map_path) as jaf:
        while True:
            data = jaf.readline()
            if not data:
                break
            data = data.strip('\n')
            label_list.append(data)
    # print(label_list, len(label_list))
    # print('filename:', filename)
    image_label = filename.split('_')[1]
    # print(image_label)
    if image_label in label_list:
        i_label = label_list.index(image_label)
        # print(i_label)
    return i_label


def createRGBFileList(label_map_path, rgb_path, rgb_save_path):
    print(rgb_save_path)
    fw = open(rgb_save_path, "w")
    files_list = os.listdir(rgb_path)
    for filename in files_list:
        label = getLabel(filename, label_map_path)
#         print('filename:', filename, label)
        imagelist_path = os.path.join(rgb_path, filename)
        images_list = os.listdir(imagelist_path)
        num = 0
        for eachname in images_list:
            if os.path.splitext(eachname)[1] == '.jpg':
                num += 1
#         print('filename:', filename, imagelist_path, '=======', label)
        fw.writelines(filename + ' ' + imagelist_path + ' ' + str(num) + ' ' + str(label) + '\n')
    fw.close()
    
def createFLOWFileList(label_map_path, flow_path, flow_save_path):
    fw = open(flow_save_path, "w")
    file_real_path = os.path.join(flow_path, 'v')
    files_list = os.listdir(file_real_path)
    for filename in files_list:
        label = getLabel(filename, label_map_path)
        flowlist_path = os.path.join(file_real_path, filename)
        images_list = os.listdir(flowlist_path)
        num = 0
        for eachname in images_list:
            if os.path.splitext(eachname)[1] == '.jpg':
                num += 1
        path = flow_path + '/{:s}/' + filename
        fw.writelines(filename + ' ' + path + ' ' + str(num) + ' ' + str(label) + '\n')
    fw.close()

def main(rgb_path,rgb_save_path,flow_path,flow_save_path,label_map_path):
    # rgb.txt
    createRGBFileList(label_map_path, rgb_path, rgb_save_path)
    # flow.txt
    createFLOWFileList(label_map_path, flow_path, flow_save_path)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rgb_path', type=str, help="name of rgb_path")
    p.add_argument('--rgb_save_path', type=str, help="name of rgb save path")
    p.add_argument('--flow_path', type=str, help="name of flow_path,")
    p.add_argument('--flow_save_path', type=str, help="name of flow save path")
    p.add_argument('--label_map_path', type=str, help="name of label map path")
    # label_map_path = 'data/ucf101/label_map.txt'
    # rgb_path = 'E://hao//huawei//data//ucf101//jpegs'
    # rgb_save_path = 'data/ucf101/rgb1.txt'
    #
    # flow_path = 'E://hao//huawei//data//ucf101//flow//tvl1_flow'
    # flow_save_path = 'data/ucf101/flow1.txt'
    main(**vars(p.parse_args()))

