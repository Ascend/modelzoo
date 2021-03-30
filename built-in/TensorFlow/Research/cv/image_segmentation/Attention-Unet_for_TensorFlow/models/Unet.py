import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os


def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
	net = slim.conv2d(inputs, n_filters, kernel_size=kernel_size, activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)

	net = slim.conv2d(net, n_filters, kernel_size=kernel_size, activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)

	return net

def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the Depthwise Separable conv block for MobileNets
	Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
	"""
	# Skip pointwise by setting num_outputs=None
	net = slim.separable_convolution2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=[3, 3], activation_fn=None)

	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	net = slim.conv2d(net, n_filters, kernel_size=[1, 1], activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
	net = tf.nn.relu(slim.batch_norm(net))
	return net

def var_weight(shape):
	var = tf.truncated_normal(shape=shape, stddev=0.01)
	return tf.Variable(var)



def build_unet(inputs, num_classes, gpu):

    #####################
	# Downsampling path #
	#####################
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	net = ConvBlock(inputs, 8)
	skip_A = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	
	net = ConvBlock(net, 16)
	skip_B = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = ConvBlock(net, 32)
	skip_C = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	
	net = ConvBlock(net, 64)
	skip_1 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = ConvBlock(net, 128)
	skip_2 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = ConvBlock(net, 256)
	skip_3 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = ConvBlock(net, 512)
	skip_4 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


	# BRIDGE
	net = ConvBlock(net, 1024)

	#####################
	# Upsampling path #
	#####################
	net = conv_transpose_block(net, 512)
	net = tf.concat([skip_4, net],  axis=3, name='fusion1')
	#net = tf.add(net, skip_4)
	net = ConvBlock(net, 512)

	
	net = conv_transpose_block(net, 256)
	#net = tf.add(net, skip_3)
	net = tf.concat([skip_3, net],  axis=3, name='fusion2')
	net = ConvBlock(net, 256)


	net = conv_transpose_block(net, 128)
	#net = tf.add(net, skip_2)
	net = tf.concat([skip_2, net],  axis=3, name='fusion3')
	net = ConvBlock(net, 128)


	net = conv_transpose_block(net, 64)
	#net = tf.add(net, skip_2)
	net = tf.concat([skip_1, net],  axis=3, name='fusion4')
	net = ConvBlock(net, 64)

	net = conv_transpose_block(net, 32)
	#net = tf.add(net, skip_2)
	net = tf.concat([skip_C, net],  axis=3, name='fusion4')
	net = ConvBlock(net, 32)

	net = conv_transpose_block(net, 16)
	#net = tf.add(net, skip_2)
	net = tf.concat([skip_B, net],  axis=3, name='fusion4')
	net = ConvBlock(net, 16)

	net = conv_transpose_block(net, 8)
	#net = tf.add(net, skip_2)
	net = tf.concat([skip_A, net],  axis=3, name='fusion4')
	net = ConvBlock(net, 8)



	#net = ConvBlock(inputs, num_classes)

	#####################
	#      Softmax      #
	#####################
	net = slim.conv2d(net, num_classes, [3, 3], activation_fn=None, scope='logits')
	return net
