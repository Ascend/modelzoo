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
import sys
import os
import numpy as np
import xml.etree.ElementTree as ET
import pickle
from os import listdir,getcwd
from os.path import join

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter',  'bench',  'bird',  'cat',
                     'dog',  'horse',  'sheep',  'cow',  'elephant',
                     'bear',  'zebra',  'giraffe',  'backpack',  'umbrella',
                     'handbag',  'tie',  'suitcase',  'frisbee',  'skis',
                     'snowboard',  'sports ball',  'kite',  'baseball bat',
                     'baseball glove',  'skateboard',  'surfboard',  'tennis racket',  'bottle',
                     'wine glass',  'cup',  'fork',  'knife',  'spoon',  'bowl',
                     'banana',  'apple',  'sandwich',  'orange',  'broccoli',
                     'carrot',  'hot dog',  'pizza',  'donut',  'cake',
                     'chair',  'sofa',  'pottedplant',  'bed',  'diningtable',
                     'toilet',  'tvmonitor',  'laptop',  'mouse',  'remote',
                     'keyboard',  'cell phone',  'microwave',  'oven',
                     'toaster',  'sink',  'refrigerator',  'book',  'clock',
                     'vase',  'scissors',  'teddy bear',  'hair drier',  'toothbrush']
def convert_annotation(xml_dir,txt_dir,image_id):
    in_file = open(join(xml_dir,image_id+'.xml'))
    out_file = open(join(txt_dir,image_id+'.txt'),'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        b = (str(xmlbox.find('xmin').text), str(xmlbox.find('ymin').text), str(xmlbox.find('xmax').text), str(xmlbox.find('ymax').text))
        out_file.write(str(cls) + ' ' + ' '.join([str(a) for a in b]) + '\n')

if __name__ == "__main__":
    xml_dir = sys.argv[1]
    txt_dir = sys.argv[2]
    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir)
    xmls = os.listdir(xml_dir)
    xmls.sort()
    for xml in xmls:
        if xml.endswith('xml'):
            image_id = xml.split('.xml')[0]
            convert_annotation(xml_dir, txt_dir, image_id)


