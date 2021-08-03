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
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import  trange
def get_arguments():
    parser = argparse.ArgumentParser(description="date prepare PSPNet")
    parser.add_argument("--img_num", type=int, default=500, help="image number")
    parser.add_argument("--crop_width", type=int, default=720, help="crop width")
    parser.add_argument("--crop_height", type=int, default=720, help="crop height")
    parser.add_argument("--data_dir", type=str, default='./cityscapes', help="image path")
    parser.add_argument("--val_list", type=str, default='./cityscapes/list/cityscapes_val_list.txt', help='Validation List File')
    parser.add_argument("--result_path", type=str, default='../results', help="eval image")
    parser.add_argument("--flipped_eval", action="store_true", help="whether to evaluate with flipped img")
    parser.add_argument("--flipped_result_path", type=str, default='../results', help="flipped eval image")
    return parser.parse_args()

def main():
    args = get_arguments()
    num_steps = args.img_num
    data_dir = args.data_dir
    crop_size = [args.crop_height, args.crop_width]
    ignore_label = 255
    num_classes = 19

    anno_filename = tf.placeholder(dtype=tf.string)
    image_filename = tf.placeholder(dtype=tf.string)
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    img.set_shape([None, None, 3])
    anno.set_shape([None, None, 1])

    shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], shape[0]), tf.maximum(crop_size[1], shape[1]))

    raw_output_npu = tf.placeholder(tf.float32, shape=[None, None, None, 19])
    flipped_raw_output = tf.placeholder(tf.float32, shape=[None, None, None, 19])
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(flipped_raw_output))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output_npu, flipped_output])
    else:
        raw_output = raw_output_npu

    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up,0, 0, shape[0], shape[1])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    pred_flatten = tf.reshape(pred, [-1,])
    raw_gt = tf.reshape(anno, [-1,])
    indices = tf.squeeze(tf.where(tf.not_equal(raw_gt, ignore_label)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes)

    sess = tf.Session()
    global_init = tf.global_variables_initializer()
    local_init =  tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    file = open(args.val_list, 'r')
    for step in trange(num_steps, desc='evaluation', leave=True):
        f1, f2 = file.readline().split('\n')[0].split(' ')
        f1 = os.path.join(data_dir, f1)
        f2 = os.path.join(data_dir, f2)
        image_name =f1.split("/")[-1].split(".")[0]
        out_name = args.result_path + "/davinci_" + image_name + "_output0.bin"
        out_file = np.fromfile(out_name, dtype="float32").reshape(1,128,256,19)

        if args.flipped_eval:
            flipped_out_name = args.flipped_result_path + "/davinci_" + image_name + "_output0.bin"
            flipped_out_file = np.fromfile(flipped_out_name, dtype="float32").reshape(1,128,256,19)
            _ = sess.run(update_op, feed_dict={image_filename: f1, anno_filename: f2, raw_output_npu: out_file, flipped_raw_output: flipped_out_file})
        else:
            _ = sess.run(update_op, feed_dict={image_filename:f1, anno_filename: f2, raw_output:out_file})
    print('mIoU: {:04f}'.format(sess.run(mIoU)))

if __name__ == '__main__':
    main()
    
