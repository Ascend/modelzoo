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

import tensorflow as tf
from PIL import Image

def convert_jpg2rgb(img_name):
    image = Image.open(img_name).convert('RGB')
    return image

def normalize_image(x):
    x = x.astype(np.float32)
    x /= 127.5
    x -= 1.
    return x

def preprocess(src_path,save_path):
    in_files = os.listdir(src_path)
    in_files.sort()

    resize_shape = [299, 299, 3]
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    for file in in_files:
        if not os.path.isdir(file):
            print(file)
            img = convert_jpg2rgb(os.path.join(src_path,file))
            img = np.array(img)

            with tf.Session().as_default():
                img_tf = tf.image.central_crop(img, 0.875).eval()
                img_tf = tf.expand_dims(img_tf, 0)
                img_tf = tf.image.resize_bilinear(img_tf, resize_shape[:-1], align_corners=False)
                img_tf = tf.squeeze(img_tf, [0])
                img_tf = img_tf.eval()
                img = normalize_image(img_tf)
                tf.reset_default_graph()

            img.tofile(os.path.join(save_path, file+'.bin'))

if __name__  == "__main__":
    if len(sys.argv) <3:
        raise Exception("Usage: python3 xception_preprocessing.py [src_path] [save_path]")

    src_path = sys.argv[1]
    save_path = sys.argv[2]

    preprocess(src_path, save_path)
