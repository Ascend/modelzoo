#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import PIL
parser = argparse.ArgumentParser(description="SSD Series Data Preprocessing.")
parser.add_argument("--input_file_path", type=str, default="../coco_minival2014",
                    help="The path of image.")
parser.add_argument("--output_file_path", type=str, default="../datasets/",
                    help="The path of inference image.")
parser.add_argument("--crop_width", type=int, default="",
                    help="Width of the image cropping.")
parser.add_argument("--crop_height", type=int, default="",
                    help="height of the image cropping.")
parser.add_argument("--save_conf_path", type=str, default="./img_info",
                    help="The path of img config path, include the image name ,width and height.")                    
args = parser.parse_args()
def get_images(image_file_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(image_file_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    files.sort()
    return files

def inference(input_file_path, output_file_path, crop_width, crop_heigth, img_info_path):
    image_filename = tf.placeholder(dtype=tf.string)

    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)

    img.set_shape([None, None, 3])

    resize_img_batch =tf.image.resize_images(img, [crop_width,crop_heigth],method=0)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        im_fn_list = get_images(input_file_path)
        f = open(img_info_path, "w+")
        i = 0
        for im_fn in im_fn_list:
            image_name = im_fn.split(".jpg")[0].split("/")[-1]
            resized_img_batch = sess.run(resize_img_batch, feed_dict={image_filename: im_fn})
            resized_img = np.squeeze(resized_img_batch, None)
            resized_img = np.asarray(resized_img, dtype='uint8')
            bin_name = output_file_path + image_name + ".bin"
            resized_img.tofile(bin_name)

            img_src =  Image.open(im_fn)
            im_width ,im_height = img_src.size
            f.write(str(i) + " " + image_name + " " + str(im_width) + " " + str(im_height) )
            f.write('\n')
            i = i + 1
        f.close()



if __name__ == '__main__':
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    new_width = args.crop_width
    new_height = args.crop_height
    img_info_path = args.save_conf_path

    inference(input_file_path, output_file_path, new_width, new_height, img_info_path)
