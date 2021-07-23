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

#prepare data for mobilenetv3large
import numpy as np
import time
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
#import npu_bridge
import glob
import os
import argparse

input_shape = (224, 224, 3)  # (height, width, channel)
CHANNEL_MEANS = [123.68, 116.78, 103.94]  # (R, G, B)



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path', default = './image-50000/',
                        help = """the data path""")
    parser.add_argument('--out_path', default = "./mobilenetv3_aipp",
                        help = """the path for out image""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args

def read_file(image_name, path):
    with open(path, 'r') as cs:
        rs_list = cs.readlines()
        for name in rs_list:
            if image_name in str(name):
                num = str(name).split(" ")[1]
                break
    return int(num) + 1


def image_resize(image):
    '''
    可以先采样到256
    '''
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]
    resize_min = tf.cast(256, tf.float32)
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)
    return tf.compat.v1.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR,align_corners=False)


def mean_normalize(image):
    '''
    对单张图片做减均值
    :param image:
    :return:
    '''
    return image - CHANNEL_MEANS


def central_crop(image):
    '''
    从目标图像中心crop 224*224的图像
    :param image:
    :return:
    '''
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - 224)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - 224)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
      image, [crop_top, crop_left, 0], [224, 224, -1])


def image_process(image_path,out_path):
    ###处理图片预处理
    imagelist = []
    labellist = []
    images_count = 0
    imagename_list = []
    for file in os.listdir(image_path):
        with tf.Session().as_default():
            image_dict = {}
            image_file = os.path.join(image_path, file)
            name = image_file.split('/')[-1]
            image_name = image_file.split('/')[-1].split('.')[0]
            image= tf.gfile.FastGFile(image_file, 'rb').read()
            img = tf.image.decode_jpeg(image)
            img = image_resize(img)
            img = central_crop(img)
            means = tf.broadcast_to(CHANNEL_MEANS, tf.shape(img))
            #img = img - means
            images_count = images_count + 1
            if tf.shape(img)[2].eval() == 1:
                img = tf.image.grayscale_to_rgb(img)
            img = img.eval()
            imagelist.append(img)
            tf.reset_default_graph()
            ###保存bin文件
            img.astype(np.uint8).tofile(out_path +'/{}.bin'.format(name))
            ###处理labels
            #lable = read_file(image_name, label_file)
            #labellist.append(lable)
            imagename_list.append(image_name)
    return np.array(imagelist), images_count,imagename_list


def main():
    args = parse_args()

    ###数据预处理
    tf.reset_default_graph()
    print("########NOW Start Preprocess!!!#########")
    #images, labels, images_count,imagename_list = image_process(args.image_path, args.label_file)
    images, images_count,imagename_list = image_process(args.image_path, args.out_path)
    #print('imagename_list:',imagename_list)



if __name__ == '__main__':
    main()


