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
from __future__ import print_function
import argparse
import os
import sys
import time
from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import trange

def get_arguments():
    parser = argparse.ArgumentParser(description="date prepare PSPNet")
    parser.add_argument("--img_num", type=int, default=500, help="image number")
    parser.add_argument("--crop_width", type=int, default=720, help="crop width")
    parser.add_argument("--crop_height", type=int, default=720, help="crop height")
    parser.add_argument("--data_dir", type=str, default='./cityscapes', help="image path")
    parser.add_argument("--val_list", type=str, default='./cityscapes/List/cityscapes_val_list.txt',help="Validation List File")
    parser.add_argument("--output_path", type=str, default='./datasets', help="eval image")
    parser.add_argument("--flipped_eval", action="store_true", help="whether to evaluate with flipped img")
    parser.add_argument("--flipped_output_path", type=str, default='./flipped_datasets', help="flipped eval image")
    return parser.parse_args()
def preprocess(img, h, w):
    #Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean
    pad_img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
    pad_img = tf.expand_dims(pad_img, dim=0)
    return pad_img

def main():
    args = get_arguments()
    # load parameters
    crop_size = [args.crop_height, args.crop_width]
    num_steps = args.img_num
    data_dir = args.data_dir

    image_filename = tf.placeholder(dtype=tf.string)
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    img.set_shape([None, None, 3])

    shape = tf.shape(img)
    h, w =(tf.maximum(crop_size[0], shape[0]), tf.maximum(crop_size[1], shape[1]))
    img = preprocess(img, h, w)

    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)

    sess = tf.Session()

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    file = open(args.val_list, 'r')
    for step in trange(num_steps, desc='evaluation', leave=True):
        f1, f2 = file.readline().split('\n')[0].split(' ')
        f1 = os.path.join(data_dir, f1)
        image_name = f1.split("/")[-1].split(".")[0]
        resized_img_batch = sess.run(img, feed_dict={image_filename: f1})
        resized_img = np.squeeze(resized_img_batch, None)
        resized_img = np.asarray(resized_img, dtype='uint8')
        bin_img = args.output_path + "/" + image_name + ".bin"
        resized_img.tofile(bin_img)
        if args.flipped_eval:
            flipped_img_batch = sess.run(flipped_img, feed_dict={image_filename: f1})
            flipped_image = np.squeeze(flipped_img_batch, None)
            filpped_image = np.asarray(flipped_image, dtype='uint8')
            bin_flipped_img = args.flipped_output_path + "/" + image_name + ".bin"
            filpped_image.tofile(bin_flipped_img)
if __name__ == '__main__':
    main()
