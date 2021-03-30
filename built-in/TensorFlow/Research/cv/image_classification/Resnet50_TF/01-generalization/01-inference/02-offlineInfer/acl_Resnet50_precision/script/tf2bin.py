# -*- coding: utf-8 -*-
"""
Text detection inference
"""
# pylint: disable=C0411,W0622,C0103,E1129
from __future__ import absolute_import, division, print_function, unicode_literals
#import cv2  # pylint: disable=import-error
import numpy as np
import tensorflow as tf  # pylint: disable=import-error
import json  # pylint: disable=import-error
from collections import OrderedDict
import time
import os
import sys
#from matplotlib.pyplot import *


#from matplotlib.pyplot import *

#MODEL_NAME = "resnet_gpu_8p"
#MODEL_NAME = "sub_graph_hc"
MODEL_NAME = "resnet50_910to310"


#image_root = r'/home/w00501968/accuracy/resnet50_inference/image'
#image_root = r'/home/w00501968/accuracy/resnet50_inference/image-1024'
#image_root = r'/home/xwx5322041/train/image'
image_root = r'/home/xwx5322041/train/image'
# label_out = os.path.basename(image_root) + '_label_result'
def test_images():
    #image_root = r'/home/xwx5322041/train/images-' + arg1
    folders = os.listdir()
    #folders = ["test_data_1024"]
    folder_list = []
    for folder in folders:
        folder_path = os.path.join(image_root, folder)
        if os.path.isdir(folder_path):
            print(folder)
            folder_list.append(folder_path)
    #print("一共有%s个文件夹" % len(folder_list))
    #params = TextDetectionParams()
    #infer = TextDetectionInference(params)

    # def proc_image(folder):
        # try:
    print("begin to proc the image...............")
    folder_name = os.path.basename(folder)
    images = os.listdir(image_root)
    print("has %s images" % len(images))
    precision_correct_count = 0
    t0 = time.time()
    num = 0
    predict_time = 0
    for image_name in images:
        if image_name.endswith("txt"):
            continue
        # image_name = "20180522135150.jpg"
        print("the image name is {}....".format(image_name))
        image_path = os.path.join(image_root, image_name)
        #image = read_image(image_path, 1)
        #img = Image.open(image_path)
        name = image_name.split('/')[-1].split('.')[0]
        print("name:****************************", name)
        image= tf.gfile.FastGFile(image_path, 'rb').read()
        with tf.Session() as sess:
            img = tf.image.decode_jpeg(image)
            print(img)
        print ("img===================",img.shape)
        if len(img.shape) != 3:
            
            continue   
       # img_rgb = img.convert('RGB')
        img = image_resize(img)
        print("resize***********", img)
        img = central_crop(img)
        print("crop***********", img)
        means = tf.broadcast_to([123.68, 116.78, 103.94], tf.shape(img))
        img = img - means
        print("mean**********", img)
        image = img
        #img.tofile('./tf2bin/{}.bin'.format(image_name))
        with tf.Session() as sess:
           img = img.eval()
           aim_path="/home/xwx5322041/tfbin-50000/" + image_name + ".bin"
           img.tofile(aim_path)
           #np.save(aim_path, img)
        #Q = np.array(iekf.Q.detach().numpy())  # tensor转换成array
        #np.savetxt(file_name, Q)

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

if __name__ == '__main__':
    test_images()