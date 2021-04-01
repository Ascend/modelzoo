
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def Upsampling(inputs, scale=2):
    up = tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale, tf.shape(inputs)[2]*scale])
    return up

def conv_transpose_block(inputs, n_filters, kernel_size=[2, 2]):
    net = tf.nn.relu(slim.batch_norm(inputs))  
    net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[2, 2], stride=[2, 2], activation_fn=None)
    return net


def shortcut(inputs, n_filters, kernel_size=[1,1], strides=[1, 1]):
    net = slim.conv2d(inputs, n_filters, kernel_size=kernel_size, stride=strides)
    net = slim.batch_norm(net)
    return net

def blockConv(inputs, n_filters, kernel_size=[3,3], strides=[1, 1]):
    net = slim.batch_norm(inputs)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[3, 3], stride=strides,activation_fn=None)
    return net

def build_deepUnet(inputs, num_classes):
    ##ENCODING
    net = slim.conv2d(inputs, 64, kernel_size=[3, 3], stride=[1, 1], activation_fn=None)
    net = blockConv(net, 64)
    shortpath = shortcut(inputs, 64)
    net = tf.add(shortpath, net)
    skip1= net 

    net = blockConv(net, 128, strides=[2,2])
    net = blockConv(net, 128)
    shortpath = shortcut(skip1, 128, strides=[2, 2])
    net = tf.add(shortpath, net)
    skip2 = net 

    net = blockConv(net, 256, strides=[2, 2]) 
    net = blockConv(net, 256)
    shortpath = shortcut(skip2, 256, strides=[2, 2])
    net = tf.add(shortpath, net)
    skip3 = net

    ##BRIDGE

    net = blockConv(net, 512, strides=[2, 2])
    net = blockConv(net, 512)
    shortpath = shortcut(skip3, 512, strides=[2, 2])
    net = tf.add(net, shortpath)

    ##DECODING
    #net = conv_transpose_block(net, 256)
    net = Upsampling(net)
    net = tf.concat([net, skip3], axis=3)
    skip4 = shortcut(net, 256)
    net = blockConv(net, 256)
    net = blockConv(net, 256)
    net = tf.add(net, skip4)

    #net = conv_transpose_block(net, 128)
    net =  Upsampling(net)
    net = tf.concat([net, skip2], axis=3)   
    skip5 = shortcut(net, 128)          
    net = blockConv(net, 128)
    net = blockConv(net, 128)
    net = tf.add(net, skip5)

    #net = conv_transpose_block(net, 64)
    net = Upsampling(net)
    net = tf.concat([net, skip1], axis=3)
    skip6 = shortcut(net, 64)
    net = blockConv(net, 64)
    net = blockConv(net, 64)
    net = tf.add(net, skip6)

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None)
    return net
