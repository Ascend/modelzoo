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
import tensorflow as tf
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import fcn_pretrain
import tensorflow.contrib.slim as slim
import argparse


from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



def read_data(tf_file,is_training):
    def _parse_read(tfrecord_file):
        features = {
            'image':
                tf.io.FixedLenFeature((),tf.string),
            "label":
                tf.io.FixedLenFeature((),tf.string),
            'height':
                tf.io.FixedLenFeature((),tf.int64),
            'width':
                tf.io.FixedLenFeature((),tf.int64),
            'channels':
                tf.io.FixedLenFeature((),tf.int64)
        }
        parsed = tf.io.parse_single_example(tfrecord_file, features)
        image = tf.decode_raw(parsed['image'], tf.uint8)
        image = tf.reshape(image, [parsed['height'], parsed['width'], parsed['channels']])
        image = tf.cast(image, tf.float32)
        label = tf.decode_raw(parsed['label'], tf.uint8)
        label = tf.reshape(label, [parsed['height'], parsed['width']])
        h_pad = 512 - parsed['height']
        w_pad = 512 - parsed['width']
        image_padding = ((h_pad//2, h_pad-h_pad//2), (w_pad//2, w_pad-w_pad//2),(0,0))
        label_padding = ((h_pad//2, h_pad-h_pad//2), (w_pad//2, w_pad-w_pad//2))
        image = tf.pad(image, image_padding, mode='constant', constant_values=0)
        label = tf.pad(label, label_padding, mode='constant', constant_values=0)
        image = image - [122.67891434,116.66876762,104.00698793]
        image = image / 255.
        return image, label, parsed['height'],parsed['width']

    dataset = tf.data.TFRecordDataset(tf_file)
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    if is_training:
        dataset = dataset.shuffle(batch_size * 100)
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch, hs, ws = iterator.get_next()
    print(images_batch, labels_batch)
    return images_batch, labels_batch, hs, ws

def check_data( images_batch, labels_batch, gpu):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            for i in range(100):
                x_in,y_in = sess.run([images_batch,labels_batch])
                print(x_in.shape, y_in.shape)
                print(np.max(x_in),np.min(x_in))
                print(np.unique(y_in))

def get_matrix(log, label, num_classes=21):
    predict = np.argmax(log, axis=-1)
    mask = (label >= 0) & (label < num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix

def get_result(confusion_matrix):
    # pixel accuracy
    Pixel_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    # mean iou
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    print(MIoU)
    MIoU = np.nanmean(MIoU)
    # mean pixel accuracy
    Mean_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    Mean_acc = np.nanmean(Mean_acc)
    # frequncey iou
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
        np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return Pixel_acc, Mean_acc, MIoU, FWIoU


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "model_path", None,
    "The config json file corresponding to the pre-trained RESNET model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "data_path", None,
    "The config json file corresponding to the pre-trained RESNET model. ")


flags.DEFINE_integer(
    "image_num", 736,
    "The config json file corresponding to the pre-trained RESNET model. ")

train_tf=  FLAGS.data_path
batch_size =1
img_N=FLAGS.image_num
img_H=512
img_W=512
is_training = False
gpu=1
model = FLAGS.model_path

images_batch,labels_batch,hs,ws = read_data(train_tf,is_training)
#check_data( images_batch, labels_batch, gpu)

inputx = tf.placeholder(tf.float32, shape=[batch_size, img_H, img_W, 3], name="inputx")
inputy = tf.placeholder(tf.int64,shape=[batch_size, img_H, img_W],  name="inputy")
config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
with tf.Session(config=config) as sess:
    with slim.arg_scope(fcn_pretrain.vgg_arg_scope()):
        out = fcn_pretrain.fcn8s(inputx, is_training=is_training)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model)
    print ("Test start....")
    c_matrix=np.zeros((21,21))
    try:
        for step in range(1000):
            #print(step)
            x_in,y_in,h,w = sess.run([images_batch,labels_batch,hs,ws])
            prediction= sess.run(out ,feed_dict={inputx: x_in})
            prediction =prediction[0, (512-h[0])//2 : (512-h[0])//2+h[0] , (512-w[0])//2 : (512-w[0])//2+w[0]  ]
            y_in = y_in[0, (512-h[0])//2 : (512-h[0])//2+h[0] , (512-w[0])//2 : (512-w[0])//2+w[0]  ]
            pre = np.argmax(prediction, axis=-1)
            c_matrix+= get_matrix(prediction, y_in, num_classes=21)
    except tf.errors.OutOfRangeError:
        print('epoch limit reached')
    finally:
        p_acc, m_acc, miou, fmiou= get_result(c_matrix)
        print("miou-----",miou)
