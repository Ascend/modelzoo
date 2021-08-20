# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
import os
import sys
import numpy as np
import shutil
import cv2
from PIL import Image
import tensorflow as tf

def convert_jpg2rgb(img_name):
    image = Image.open(img_name).convert('RGB')
    return image

def preprocess(src_path,dst_path):
    in_files = os.listdir(src_path)
    in_files.sort()
    resize_shape = [768,1280,3]
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)
    else:
        shutil.rmtree(dst_path)
        os.makedirs(dst_path)
    for file in in_files:
        x = np.fromfile(os.path.join(src_path, file), np.float32).reshape(resize_shape)
        sqz_mean = np.array([123,117,104], np.float32)
        x = x + sqz_mean
        x = x.astype(np.uint8, copy=False)
        x.tofile(os.path.join(dst_path,file))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("Usage: python3 xxx.py [src_path] [dst_path]")
    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    preprocess(src_path,dst_path)