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
import argparse
import array
import collections
import json
import os
import sys
import threading
import time
from queue import Queue
#import env
import cv2
import numpy as np
import re
import pdb

# import converter.converter as converter
#from backend.backend_acl import AclBackend



last_timing = []
last_device_timing = []


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./datasets/imagenet_10", help="path to the dataset")
    parser.add_argument("--backend", default="acl", help="runtime to use")
    parser.add_argument("--model", required=True, help="model file path")
    parser.add_argument("--image_size",default='224,224,3',help="model inputs imagesize")
    parser.add_argument("--inputs", help="model inputs nodes eg: data1:0 ")
    parser.add_argument("--outputs", help="model outputs nodes list eg:fc1:0,fc2:0,fc3:0 ")

    # below will override DNMetis rules compliant settings - don't use for official submission
    parser.add_argument("--count", default=1000, type=int, help="dataset items to infer")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8", "uint8"],
                         help="precision mode, one of " + str(["fp32", "fp16", "int8", "uint8"]))
    parser.add_argument("--feed", default=[], help="feed")
    parser.add_argument("--image_list", default=[], help="image_list")
    parser.add_argument("--label_list", default=[], help="label_list")
    parser.add_argument("--cfg_path",default="./backend_cfg/built-in_config.txt")
    args = parser.parse_args()

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give


    #if args.image_size is None:
    #    args.image_size = SUPPORTED_DATASETS[args.dataset][3]['image_size']

    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")
    if args.image_size:
        args.image_size = list(map(int, args.image_size.split(",")))

    return args

def get_backend(backend):
    if backend == "acl":
        from backend.backend_acl import AclBackend
        backend = AclBackend()
    return backend

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def center_crop(img, out_height, out_width):
    height, width = img.shape[:2]
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def pre_process_nasnet(img, dims=None, precision="fp32"):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_CUBIC
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    STDDEV_RGB = [1.0 * 255, 1.0 * 255, 1.0 * 255]

    if precision=="fp32":
        img = np.asarray(img, dtype='float32')
    if precision=="fp16":
        img = np.asarray(img, dtype='float16')

    stddev = np.array(STDDEV_RGB, dtype=np.float32)
    img /= stddev

    return img

def preprocess_dataset(args,offset=0):
    with open(args.dataset_path + '/val_map.txt', 'r') as f:
        for s in f:
            image_name, label = re.split(r"\s+", s.strip())
            src = os.path.join(args.dataset_path, image_name)
            img_org = cv2.imread(src)
            processed_img = pre_process_nasnet(img_org, dims=args.image_size, precision = args.precision)
            args.feed.append(processed_img)
            args.image_list.append(image_name)
            args.label_list.append(int(label)+offset)


def main():
    good = 0
    total = 0
    #args
    args = get_args()

    # find backend
    backend = get_backend(args.backend)

    # load model to backend
    model = backend.load(args)
    #
    # preprocess_dataset
    preprocess_dataset(args,offset=1)

    #start inference:
    for i in range(len(args.feed)):
        predictions = backend.predict(args.feed[i])
        print('img_orig:',args.image_list[i],'label:',args.label_list[i],'predictions:',np.argmax(predictions),'\n')
        if args.label_list[i] == np.argmax(predictions):
            good +=1
        total +=1
    print('Predict total jpeg:',len(args.image_list),' Accuracy: ',good / total)
    
    backend.unload()

if __name__ == "__main__":
    main()
