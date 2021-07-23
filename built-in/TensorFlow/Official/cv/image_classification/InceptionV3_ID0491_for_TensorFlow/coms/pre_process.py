#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by pre_process on 19-1-30

import tensorflow as tf
import os
import numpy as np
import coms.utils as utils
import coms.tfrecords as tfrecords

tf.app.flags.DEFINE_string('dataset_dir', './data', '')
FLAGS = tf.app.flags.FLAGS

def get_cifar10_batch(is_train, batch_size, num_cls,img_prob):
    train_dir = r''
    test_dir = r''
    aim_dir = r''
    if utils.isLinuxSys():
        train_dir = os.path.join(FLAGS.dataset_dir, "train.tfrecords")
        #train_dir = r'data/train.tfrecords'
        test_dir = os.path.join(FLAGS.dataset_dir, "test.tfrecords")
        #test_dir  = r'data/test.tfrecords'
    else:
        train_dir = r'D:\DataSets\cifar\cifar\tfrecords\train.tfrecords'
        test_dir = r'D:\DataSets\cifar\cifar\tfrecords\test.tfrecords'
    '''
    if is_train:
        aim_dir = train_dir
        print(aim_dir)
        train_img_tfrecords, train_label_tfrecords = tfrecords.get_tfrecords(aim_dir,img_prob=img_prob)
        train_img_batch, train_label_batch = get_batch_tfrecords(train_img_tfrecords,train_label_tfrecords,img_prob[0],img_prob[1],batch_size,10)
        train_label_batch = tf.one_hot(train_label_batch,depth=num_cls)
        return train_img_batch,train_label_batch
        #yield train_img_batch,train_label_batch
    else:
        aim_dir = test_dir
        test_img_tfrecords, test_label_tfrecords = tfrecords.get_tfrecords(aim_dir,img_prob=img_prob)
        test_img_batch, test_label_batch = get_batch_tfrecords(test_img_tfrecords,test_label_tfrecords,img_prob[0],img_prob[1],batch_size,1,False)
        test_label_batch = tf.one_hot(test_label_batch,depth=num_cls)
        return test_img_batch,test_label_batch
        #yield test_img_batch,test_label_batch
    '''
    # for npu
    if is_train:
        aim_dir = train_dir
        print(aim_dir)
        ds = tfrecords.get_tfrecords_npu(aim_dir, img_prob=img_prob, batch_size=batch_size, num_cls=num_cls)
        return ds
    else:
        aim_dir = test_dir
        print(aim_dir)
        ds = tfrecords.get_tfrecords_npu(aim_dir, img_prob=img_prob, batch_size=batch_size, num_cls=num_cls)
        return ds
    # for npu



def get_dogcat_img(file_dir):
    cls_list = ['dog','cat']

    cls_img_path , cls_img_label = [],[]

    for file in os.listdir(file_dir):
        for index , name in enumerate(cls_list):
            if name in file:
                cls_img_path.append(file_dir + '/' + file)
                cls_img_label.append(index)
    temp = np.array([cls_img_path,cls_img_label])
    temp = temp.transpose()

    np.random.shuffle(temp)
    img_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int (i) for i in label_list]

    return img_list, label_list


def get_cifar10_img(file_dir):
    cls_list = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    cls_img_path = []
    cls_img_label = []
    # for index , name in enumerate(cls_list):
    #     print(index,  name)
    #     print(cls_list.index(name))
    cout = 0
    for file in os.listdir(file_dir):
        for index , name in enumerate(cls_list):
            if name in file:
                cls_img_path.append(file_dir+'/'+file)
                cls_img_label.append(index)
                break
        # if cout == 10:
        #     break
        # else:
        #     cout = cout + 1
    temp = np.array([cls_img_path,cls_img_label])
    temp = temp.transpose()

    np.random.shuffle(temp)

    img_list = list(temp[:,0])
    label_list = list(temp[:,1])

    label_list = [int (i) for i in label_list]

    return img_list, label_list

def get_batch_tfrecords(imgs,label,img_w,img_h,batch_size,num_threads,shuffle=True):
    imgs = tf.image.resize_images(images=imgs,size=[img_w,img_h],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # capacity=5+batch_size*3, min_after_dequeue=5
    if shuffle:
        img_batch, label_batch = tf.train.shuffle_batch(
            [imgs, label]
            , batch_size=batch_size
            , capacity=5+batch_size*3
            , num_threads=num_threads
            , min_after_dequeue=10
        )
    else:
        # num_thread = 1时，每次去除的样本顺序固定不变
        img_batch, label_batch = tf.train.batch(
            [imgs, label]
            , batch_size=batch_size
            , capacity=5+batch_size*3
            , num_threads=num_threads
        )
    img_batch = tf.cast(img_batch, tf.float32)

    return img_batch,label_batch





'''
生成相同大小的批次,使用此函数将图片分批次，原因为一次性将大量图片读入内存可能会存在内存不足，同时也是性能浪费
@:param img get_cat_and_dog_files()返回的img_list
@:param label get_cat_and_dog_files()返回的label_list
@:param img_w, img_h  设置好固定的宽和高
@:param batch_size 每个batch的大小
@:param capacity 一个队列最大容量
@:return 包含图像和标签的batch
'''
def get_batch(img, label, img_w, img_h, batch_size, capacity):

    # 格式化为tf需要的格式
    img = tf.cast(img, tf.string)
    label = tf.cast(label, tf.int32)

    # 生产队列
    input_queue = tf.train.slice_input_producer([img,label])

    # 从队列中读取图
    img_contents = tf.read_file(input_queue[0])
    label = input_queue[1]

    # 图像解码,不同类型图像不要混在一起
    img = tf.image.decode_jpeg(img_contents, channels=3)

    # 图像统一预处理,缩放，旋转，裁剪，归一化等
    img = tf.image.resize_images(images=img,size=[img_h,img_w],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    img = tf.cast(img, tf.float32) / 255.   # 转换数据类型并归一化

    # 图片标准化
    # img = tf.image.per_image_standardization(img)

    img_batch, label_batch = tf.train.batch(
        [img,label],
        batch_size= batch_size,
        num_threads= 64,
        capacity=capacity
    )

    # label_batch = tf.reshape(label_batch,[batch_size])

    img_batch = tf.cast(img_batch, tf.float32)

    return img_batch, label_batch



if __name__ == '__main__':
    get_cifar10_img('1')
