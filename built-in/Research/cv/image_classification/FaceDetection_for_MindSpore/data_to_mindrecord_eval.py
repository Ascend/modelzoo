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
"""Convert VOC format dataset to mindrecord for evaluating Face detection."""
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from mindspore import log as logger
from mindspore.mindrecord import FileWriter, FileReader, MindPage, SUCCESS

dataset_root_list = ["Your_VOC_dataset_path1",
                     "Your_VOC_dataset_path2",
                     "Your_VOC_dataset_pathN",
                     ]

mindrecord_file_name = "Your_output_path/data.mindrecord"

mindrecord_num = 8
is_train = False
class_indexing = {'face': 0}


def prepare_file_paths():
    image_files = []
    anno_files = []
    image_names = []
    for dataset_root in dataset_root_list:
        if not os.path.isdir(dataset_root):
            raise ValueError("dataset root is unvalid!")
        anno_dir = os.path.join(dataset_root, "Annotations")
        image_dir = os.path.join(dataset_root, "JPEGImages")
        if is_train:
            valid_txt = os.path.join(dataset_root, "ImageSets/Main/train.txt")
        else:
            valid_txt = os.path.join(dataset_root, "ImageSets/Main/test.txt")

        ret_image_files, ret_anno_files, ret_image_names = filter_valid_files_by_txt(image_dir, anno_dir, valid_txt)
        image_files.extend(ret_image_files)
        anno_files.extend(ret_anno_files)
        image_names.extend(ret_image_names)
    return image_files, anno_files, image_names


def filter_valid_files_by_txt(image_dir, anno_dir, valid_txt):
    with open(valid_txt, "r") as txt:
        valid_names = txt.readlines()
    image_files = []
    anno_files = []
    image_names = []
    for name in valid_names:
        strip_name = name.strip("\n")
        anno_joint_path = os.path.join(anno_dir, strip_name + ".xml")
        if os.path.isfile(anno_joint_path):
            image_joint_path = os.path.join(image_dir, strip_name + ".jpg")
            image_name = image_joint_path.split('/')[-1].replace('.jpg', '')
            if os.path.isfile(image_joint_path):
                image_files.append(image_joint_path)
                anno_files.append(anno_joint_path)
                image_names.append(image_name)
                continue
            image_joint_path = os.path.join(image_dir, strip_name + ".png")
            image_name = image_joint_path.split('/')[-1].replace('.png', '')
            if os.path.isfile(image_joint_path):
                image_files.append(image_joint_path)
                anno_files.append(anno_joint_path)
                image_names.append(image_name)
    return image_files, anno_files, image_names


def deserialize(member, class_indexing):
    class_name = member[0].text
    if class_name in class_indexing:
        class_num = class_indexing[class_name]
    else:
        return None
    bnx = member.find('bndbox')
    box_x_min = float(bnx.find('xmin').text)
    box_y_min = float(bnx.find('ymin').text)
    box_x_max = float(bnx.find('xmax').text)
    box_y_max = float(bnx.find('ymax').text)
    width = float(box_x_max - box_x_min + 1)
    height = float(box_y_max - box_y_min + 1)
 
    try:
        ignore = float(member.find('ignore').text)
    except:
        ignore = 0.0
    return [class_num, box_x_min, box_y_min, width, height, ignore]


def get_data(image_file, anno_file, image_name):
    count = 0
    annotation = []
    tree = ET.parse(anno_file)
    root = tree.getroot()

    with Image.open(image_file) as fd:
        orig_width, orig_height = fd.size

    with open(image_file, 'rb') as f:
        img = f.read()

    for member in root.findall('object'):
        anno = deserialize(member, class_indexing)
        if anno is not None:
            annotation.extend(anno)
            count += 1

    for member in root.findall('Object'):
        anno = deserialize(member, class_indexing)
        if anno is not None:
            annotation.extend(anno)
            count += 1

    if count == 0:
        annotation = np.array([[-1, -1, -1, -1, -1, -1]], dtype='float64')
        count = 1

    data = {
        "image": img,
        "annotation": np.array(annotation, dtype='float64'),
        "image_name": image_name,
        "image_size": np.array([orig_width, orig_height], dtype='int32')
        }
    return data


def convert_yolo_data_to_mindrecord():

    writer = FileWriter(mindrecord_file_name, mindrecord_num)
    yolo_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "float64", "shape": [-1, 6]},
        "image_name": {"type": "string"},
        "image_size": {"type": "int32", "shape": [-1, 2]}
        }

    print('Loading eval data...')
    image_files, anno_files, image_names = prepare_file_paths()
    dataset_size = len(anno_files)
    assert dataset_size == len(image_files)
    assert dataset_size == len(image_names)
    logger.info("#size of dataset: {}".format(dataset_size))
    data = []
    for i in range(dataset_size):
        data.append(get_data(image_files[i], anno_files[i], image_names[i]))

    print('Writing eval data to mindrecord...')
    writer.add_schema(yolo_json, "yolo_json")
    if data is None:
        raise ValueError("None needs writing to mindrecord.")
    writer.write_raw_data(data)
    writer.commit()


convert_yolo_data_to_mindrecord()

