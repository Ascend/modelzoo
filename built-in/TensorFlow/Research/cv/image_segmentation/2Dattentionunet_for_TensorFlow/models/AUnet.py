import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
from tensorflow.contrib.slim import conv2d
from tensorflow.contrib.resampler import resampler

def add(tensor_a, tensor_b):
    return tf.add(tensor_a, tensor_b)

def concat(X, Y, axis=-1):
    return tf.concat([X, Y], axis=axis)

def ConvRelu(inputs, n_filters, kernel_size=[3, 3]):
    net = conv2d(inputs, n_filters, kernel_size=kernel_size)
    net = tf.nn.relu(net)

    return net

def resBlock(inputs, n_filters):
    net = ConvRelu(inputs, n_filters)
    net = ConvRelu(net, n_filters)
    net = ConvRelu(net, n_filters)
    net = add(inputs, net)
    return net

def ConvBlock(inputs, n_filters, third=True):
    net = ConvRelu(inputs, n_filters, kernel_size=[1, 1])
    net = resBlock(net, n_filters)
    net = ConvRelu(net, n_filters)
    #
    return net

def UpBlock(inputs, n_filters):
    net = Upsample(inputs)
    net = ConvBlock(net, n_filters)
    return net

def attention(tensor, att_tensor, n_filters=512, kernel_size=[1, 1]):
    g1 = conv2d(tensor, n_filters, kernel_size=kernel_size)
    x1 = conv2d(att_tensor, n_filters, kernel_size=kernel_size)
    net = add(g1, x1)
    net = tf.nn.relu(net)
    net = conv2d(net, 1, kernel_size=kernel_size)
    net = tf.nn.sigmoid(net)
    #net = tf.concat([att_tensor, net], axis=-1)
    net = net * att_tensor
    return net

def Upsample(tensor, rate=2):
    return tf.image.resize_bilinear(tensor, (tf.shape(tensor)[1] * rate, tf.shape(tensor)[2] * rate))

def build_AUnet(inputs, n_classes):
    n_filters = 64
    net_1 = ConvBlock(inputs, n_filters)
    net = slim.pool(net_1, [2, 2], stride=[2, 2], pooling_type='MAX')

    net_2 = ConvBlock(net, n_filters * 2)
    net = slim.pool(net_2, [2, 2], stride=[2, 2], pooling_type='MAX')

    net_3 = ConvBlock(net, n_filters * 4)
    net = slim.pool(net_3, [2, 2], stride=[2, 2], pooling_type='MAX')

    net_4 = ConvBlock(net, n_filters * 8, third=False)
    net = slim.pool(net_4, [2, 2], stride=[2, 2], pooling_type='MAX')

    net_5 = ConvBlock(net, n_filters * 16)
    #net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    #pooled =

    # Attention Mechanism
    net_5 = UpBlock(net, 8 * n_filters)
    net = attention(net_4, net_5, 8 * n_filters)
    net = concat(net_4, net_5)
    net = ConvBlock(net, 8 * n_filters)

    up_3 = UpBlock(net, 4 * n_filters)
    net = attention(net_3, up_3, 4 * n_filters)
    net = concat(up_3, net)
    net = ConvBlock(net, 4 * n_filters)


    up_2 = UpBlock(net, 2 * n_filters)
    net = attention(net_2, up_2, 2 * n_filters)
    net = concat(up_2, net)
    net = ConvBlock(net, 2 * n_filters)

    up_1 = UpBlock(net, n_filters)
    net = attention(net_1, up_1, n_filters)
    net = concat(up_1, net)
    net = ConvBlock(net,  n_filters)

    net = conv2d(net, n_classes, kernel_size=[1, 1], activation_fn=None)

    return net
