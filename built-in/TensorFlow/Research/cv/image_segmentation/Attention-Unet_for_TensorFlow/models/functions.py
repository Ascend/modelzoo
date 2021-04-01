
from npu_bridge.npu_init import *
import os, time, cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.layers import instance_norm
from tensorflow.contrib.slim import conv2d, conv2d_transpose, pool
from tensorflow.python.client import device_lib

def elu(inputs):
    return tf.nn.elu(inputs)

def relu(inputs):
    return tf.nn.relu(inputs)

def leaky(inputs):
    return tf.nn.leaky_relu(inputs)

def shortcut(input, res, n_filters, equal=True):
    if (not equal):
        net = conv2d(input, n_filters, kernel_size=[1, 1], activation_fn=None)
        net = add(res, net)
    else:
        net = add(input, res)
    return net

def desconv(inputs, n_filters, kernel_size=[2, 2], stride=[2, 2]):
    net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[2, 2], stride=stride, activation_fn=None)
    net = elu(net)
    return net

def Upsample(net, rate=2):
    return tf.image.resize_bilinear(net, ((tf.shape(net)[1] * rate), (tf.shape(net)[2] * rate)))

def add(tensor_a, tensor_b):
    return tf.add(tensor_a, tensor_b)

def concat(tensor_a, tensor_b):
    return tf.concat([tensor_a, tensor_b], axis=3)

def resConv(inputs, n_filters, kernel_size=[3, 3], reluAct=False):
    net = convBlock(inputs, n_filters)
    net = convBlock(net, n_filters)
    net_1 = shortcut(inputs, net, n_filters, equal=False)
    net = convBlock(net_1, n_filters)
    net = convBlock(net, n_filters)
    net_2 = shortcut(net_1, net, n_filters)
    net = convBlock(net_2, n_filters)
    net = convBlock(net, n_filters)
    net_3 = shortcut(net_2, net, n_filters)
    return net_3

def resBlock(inputs, n_filters, first=1, btk=3):
    if (first == 0):
        inputs = conv2d(inputs, n_filters[(- 1)], kernel_size=[1, 1], activation_fn=None)
    if (btk != 1):
        net = convBlock(inputs, n_filters[0])
        net = convBlock(inputs, n_filters[1])
        net = convBlock(inputs, n_filters[1])
    else:
        net = convBlock(inputs, n_filters[0], kernel_size=[1, 1])
        net = convBlock(net, n_filters[1])
        net = conv2d(net, n_filters[2], kernel_size=[1, 1], activation_fn=None)
    net = add(inputs, net)
    return net

def decoderBlock(inputs, n_filters, desconv=False):
    net = UpsampleNear(inputs)
    net = convBlock(net, n_filters)
    return net

def Conv(inputs, n_filters, kernel_size=[3, 3], stride=[1, 1]):
    return conv2d(inputs, n_filters, kernel_size, stride, activation_fn=None)

def IncRes2(inputs, n_filters):
    act = Conv(inputs, n_filters, kernel_size=[1, 1])
    b1 = Conv(act, n_filters, kernel_size=[1, 1])
    b2 = Conv(act, n_filters, kernel_size=[1, 1])
    b3 = Conv(act, n_filters, kernel_size=[1, 1])
    b1 = Conv(b1, n_filters)
    b1 = Conv(b1, n_filters)
    b2 = Conv(b2, n_filters)
    fusion = tf.concat([b1, b2, b3], axis=(- 1))
    net = Conv(fusion, n_filters, kernel_size=[1, 1])
    net = relu(add(act, net))
    return net

def Upsample_bi(inputs, rate=2):
    with tf.device('/cpu:0'):
        return tf.image.resize_bicubic(inputs, ((tf.shape(inputs)[1] * rate), (tf.shape(inputs)[2] * rate)))

def ResBlockX(inputs, n_filters, b=3):
    for i in range(b):
        inputs = resBlock(inputs, n_filters, i, b)
    return inputs

def UpsampleNear(inputs, rate=2):
    return tf.image.resize_nearest_neighbor(inputs, ((tf.shape(inputs)[1] * rate), (tf.shape(inputs)[2] * rate)))

def UpsampleNear_conv(inputs, n_filters, rate=2):
    up = UpsampleNear(inputs, rate=rate)
    conv = convBlock(up, n_filters)
    return conv

def convBlock(inputs, n_filters, kernel_size=[3, 3], stride=[1, 1], reluAct=True):
    net = conv2d(inputs, n_filters, kernel_size=kernel_size, stride=stride, activation_fn=None)
    if reluAct:
        net = relu(net)
    else:
        net = elu(net)
    return net

def FPABlock(inputs, n_filters=1024, pooling_type='MAX', rate=16):
    net_1 = conv2d(inputs, n_filters, kernel_size=[1, 1], activation_fn=None)
    net_7 = slim.pool(inputs, [2, 2], stride=[2, 2], pooling_type=pooling_type)
    net_7 = conv2d(net_7, n_filters, kernel_size=[7, 7])
    net_5 = slim.pool(net_7, [2, 2], stride=[2, 2], pooling_type=pooling_type)
    net_5 = conv2d(net_5, n_filters, kernel_size=[5, 5])
    net_3 = slim.pool(net_5, [2, 2], stride=[2, 2], pooling_type=pooling_type)
    net_3 = conv2d(net_3, n_filters, kernel_size=[3, 3])
    net_3 = conv2d(net_3, n_filters, kernel_size=[3, 3])
    net_3 = relu(net_3)
    net_3 = Upsample(net_3)
    net_5 = conv2d(net_5, n_filters, kernel_size=[5, 5])
    net_5 = relu(net_5)
    net_5 = add(net_3, net_5)
    net_5 = UpsampleNear(net_5)
    net_7 = conv2d(net_7, n_filters, kernel_size=[5, 5])
    net_7 = relu(net_7)
    net_7 += net_5
    net_7 = UpsampleNear(net_7)
    net = tf.multiply(net_7, net_1)
    GAP = slim.pool(inputs, [rate, rate], stride=[rate, rate], padding='SAME', pooling_type='AVG')
    net_GAP = conv2d(GAP, n_filters, kernel_size=[1, 1])
    net_GAP = UpsampleNear(net_GAP, rate=rate)
    net = add(net, net_GAP)
    return net
