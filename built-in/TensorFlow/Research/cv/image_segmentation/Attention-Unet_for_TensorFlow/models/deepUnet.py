
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import functions as F
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], stride=[1, 1]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non

	net = slim.conv2d(inputs, n_filters, kernel_size=kernel_size, stride=stride, activation_fn=None)
	#net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net


def conv_transpose_block(inputs, n_filters, kernel_size=[2, 2]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[2, 2], stride=[2, 2], activation_fn=None)
	net = tf.nn.relu(net)#slim.batch_norm(net))
	return net

def blockConv(net, n_filters, kernel_size=[3,3], strides=[1, 1]):
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=kernel_size, stride=strides,activation_fn=None)
    return net

def build_deepUnet(inputs, num_classes):
    ##ENCODING
	net = slim.conv2d(inputs, 64, kernel_size=[3, 3], activation_fn=None)
	net = blockConv(net, 64)
	shortpath = slim.conv2d(inputs, 64, kernel_size=[1, 1], stride=[1, 1])
	net = tf.add(net, shortpath)
	skip1= net

	net = blockConv(net, 128, strides=[2,2])
	net = blockConv(net, 128)
	shortpath = slim.conv2d(skip1, 128, kernel_size=[1, 1], stride=[2, 2])
	net = tf.add(net, shortpath)
	skip2 = net

	net = blockConv(net, 256, strides=[2, 2])
	net = blockConv(net, 256)
	shortpath = slim.conv2d(skip2, 256, kernel_size=[1, 1], stride=[2, 2])
	net = tf.add(net, shortpath)
	skip3 = net

	net = blockConv(net, 512, strides=[2, 2])
	net = blockConv(net, 512)
	shortpath = slim.conv2d(skip3, 512, kernel_size=[1, 1], stride=[2, 2])
	net = tf.add(net, shortpath)
	skip4 = net

	##BRIDGE


	net = F.FPABlock(net, n_filters=512, rate=32)

	##DECODING

	net = conv_transpose_block(net, 256)
	net = tf.concat([net, skip3], axis=3)

	skip4 = slim.conv2d(net, 256, kernel_size=[1, 1])
	net = blockConv(net, 256)
	net = blockConv(net, 256)
	net = tf.add(net, skip4)

	net = conv_transpose_block(net, 128)
	net = tf.concat([net, skip2], axis=3)
	skip5 = slim.conv2d(net, 128, kernel_size=[1, 1])
	net = blockConv(net, 128)
	net = blockConv(net, 128)
	net = tf.add(net, skip5)

	net = conv_transpose_block(net, 64)
	net = tf.concat([net, skip1], axis=3)
	skip6 = slim.conv2d(net, 64, kernel_size=[1, 1])
	net = blockConv(net, 64)
	net = blockConv(net, 64)
	net = tf.add(net, skip6)

	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None)
	return net
