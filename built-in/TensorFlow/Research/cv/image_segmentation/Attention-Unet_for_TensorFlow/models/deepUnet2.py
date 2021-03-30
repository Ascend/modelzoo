
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], stride=[1, 1]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
    
	net = slim.conv2d(inputs, n_filters, kernel_size=kernel_size, stride=stride, activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net


def conv_transpose_block(inputs, n_filters, kernel_size=[2, 2]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[2, 2], stride=[2, 2], activation_fn=None)
	net = tf.nn.relu(slim.batch_norm(net))
	return net

def shortcut(inputs, n_filters, kernel_size=[1,1], strides=[1, 1]):
    net = slim.conv2d(inputs, n_filters, kernel_size=kernel_size, stride=strides)
    net = slim.batch_norm(net, fused=True)
    return net

def blockConv(inputs, n_filters, kernel_size=[3,3], strides=[1, 1]):
    net = slim.batch_norm(inputs, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[3, 3], stride=strides,activation_fn=None)
    return net

def build_deepUnet(inputs, num_classes):
    ##ENCODING
    net = slim.conv2d(inputs, 64, kernel_size=[3, 3], activation_fn=None)
    net = blockConv(net, 64)
    shortpath = shortcut(net, 64)
    net = tf.add(net, shortpath)
    skip1= net 

    net = blockConv(net, 128, strides=[2,2])
    net = blockConv(net, 128)
    shortpath = shortcut(skip1, 128, strides=[2, 2])
    net = tf.add(net, shortpath)
    skip2 = net 

    net = blockConv(net, 256, strides=[2, 2]) 
    net = blockConv(net, 256)
    shortpath = shortcut(skip2, 256, strides=[2, 2])
    net = tf.add(net, shortpath)
    skip3 = net

    ##BRIDGE

    net = blockConv(net, 512, strides=[2, 2])
    net = blockConv(net, 512)
    shortpath = shortcut(skip3, 512, strides=[2, 2])
    net = tf.add(net, shortpath)

    ##DECODING
    net = conv_transpose_block(net, 256)
    net = tf.concat([net, skip3], axis=3)
    skip4 = shortcut(net, 256)
    net = blockConv(net, 256)
    net = blockConv(net, 256)
    net = tf.add(net, skip4)

    net = conv_transpose_block(net, 128)
    net = tf.concat([net, skip2], axis=3)   
    skip5 = shortcut(net, 128)          
    net = blockConv(net, 128)
    net = blockConv(net, 128)
    net = tf.add(net, skip5)

    net = conv_transpose_block(net, 64)
    net = tf.concat([net, skip1], axis=3)
    skip6 = shortcut(net, 64)
    net = blockConv(net, 64)
    net = blockConv(net, 64)
    net = tf.add(net, skip6)

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None)
    return net
