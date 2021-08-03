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
import random
import time, datetime
import os, sys
import argparse
import cv2, math
import numpy as np
from utils import load_divided

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lashan', help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=224, help='width of cropped input image to network')
parser.add_argument('--output_path', type=str, default='../datasets', help='eval image path.')
args = parser.parse_args()

def prepare_data(dataset_dir=args.dataset):
    val_input_names, val_output_names, test_input_names, test_output_names = load_divided(dataset_dir)
    print('validation data length: {}'.format(len(val_input_names)))
    val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return val_input_names, val_output_names, test_input_names, test_output_names

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    h, w = args.crop_height, args.crop_width
    image = cv2.resize(image, (h, w))
    return image

print("Loading the data ...")
val_input_names, val_output_names, test_input_names, test_output_names = prepare_data()

for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d" % (ind + 1, len(test_input_names)))
    sys.stdout.flush()

    input_image = np.expand_dims(np.uint8(load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]), axis=0)

    st = time.time()
    print("input_image shape:",input_image.shape)
    input_image.tofile(args.output_path + "/" + str(ind) + ".bin")
