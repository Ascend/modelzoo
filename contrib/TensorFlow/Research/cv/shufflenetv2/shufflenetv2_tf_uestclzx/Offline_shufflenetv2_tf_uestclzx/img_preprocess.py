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
import os
import cv2
import numpy as np
import sys
import argparse
import tensorflow as tf

RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR

def pre_process_img(img, dims=None, precision="fp32", MIN_DIMENSION = 256):
    
    dst_height, dst_width, channel = dims
    img = (1.0 / 255.0) * tf.to_float(img)
    
    img = resize_keeping_aspect_ratio(img, MIN_DIMENSION)
    img = central_crop(img, crop_height=dst_height, crop_width=dst_width)
    img.set_shape([dst_height, dst_width, 3])
    
    return img
    
    
def resize_keeping_aspect_ratio(image, min_dimension):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        min_dimension: an int tensor with shape [].
    Returns:
        a float tensor with shape [new_height, new_width, 3],
            where min_dimension = min(new_height, new_width).
    """
    image_shape = tf.shape(image)
    height = tf.to_float(image_shape[0])
    width = tf.to_float(image_shape[1])
    original_min_dim = tf.minimum(height, width)
    scale_factor = tf.to_float(min_dimension) / original_min_dim
    new_height = tf.round(height * scale_factor)
    new_width = tf.round(width * scale_factor)
    new_size = [tf.to_int32(new_height), tf.to_int32(new_width)]
    image = tf.image.resize_images(image, new_size, method=RESIZE_METHOD)
    
    return image

def central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    
    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2

    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", default="/home/HwHiAiUser/ImageNet12/val", help="path of original pictures")
    parser.add_argument("--dst_path", default="./input", help="path of output bin files")
    parser.add_argument("--pic_num", default=-1, help="picture number")
    args = parser.parse_args()
     
    src_path = args.src_path
    dst_path = args.dst_path
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    pic_num  = args.pic_num
    files = os.listdir(src_path)
    files.sort()
    img_size = '224,224,3'
    n = 0
    image_size = [224,224,3]
    
    for file in files:
        if file.endswith('.JPEG'):
            tf.reset_default_graph()
            with tf.Session() as sess:
                src = src_path + "/" + file
                print("start to process %s"%src)
                img = tf.read_file(src)
                img_org = tf.image.decode_jpeg(img, channels=3)
                print(img_org)
                op = pre_process_img(img_org, dims=image_size, precision="fp32", MIN_DIMENSION = 256)
                res = sess.run(op)
                print(res.dtype)
            tf.get_default_graph().finalize()
            res.tofile(dst_path+"/" + file+".bin")
            n += 1
            if int(pic_num) == n:
                break
