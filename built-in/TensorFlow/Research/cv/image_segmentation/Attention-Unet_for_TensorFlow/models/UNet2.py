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
	#net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net


def conv_transpose_block(inputs, n_filters, kernel_size=[2, 2], stride=[2, 2]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=kernel_size, stride=stride, activation_fn=None)
	net = tf.nn.relu((net))
	return net

def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def desconv(inputs, n_filters, stride=[2, 2], res=False):

	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[2, 2], stride=stride, activation_fn=None)
    #net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	if res:
		net = tf.reshape(net, (16, 14, 14, 128))
	#net = slim.batch_norm(net, fused=True)
    #net = instance_norm(net, fused=True)
	return net

def desconv_new(inputs, n_filters, stride=[2,2], shape=[-1, 14, 14, 128]):
    shapes = [3,3,128, 128]
    weight = get_weights(shapes)
    strides = [1, stride[0], stride[1], 1]
    net = tf.nn.conv2d_transpose(inputs, weight, output_shape=[16, 14, 14, 128], strides=strides, padding='VALID')
    #net = slim.batch_norm(net, fused=True)
    #net = tf.nn.relu(net)
    #net = slim.batch_norm(net, fused=True)
    #net = instance_norm(net, fused=True)
    return net


def get_pool(pool_4):
    pool_list = []
    #pool_list.append(slim.pool(pool_4, [16, 16], stride=[1, 1], pooling_type='AVG'))
    #pool_list.append(slim.pool(pool_4, [15, 15], stride=[1, 1], pooling_type='AVG'))
    pool_list.append(slim.pool(pool_4, [14, 14], stride=[1, 1], pooling_type='AVG'))
    pool_list.append(slim.pool(pool_4, [13, 13], stride=[1, 1], pooling_type='AVG'))
    pool_list.append(slim.pool(pool_4, [12, 12], stride=[1, 1], pooling_type='AVG'))
    pool_list.append(slim.pool(pool_4, [11, 11], stride=[1, 1], pooling_type='AVG'))
    pool_list.append(slim.pool(pool_4, [9, 9], stride=[1, 1], pooling_type='AVG'))
    pool_list.append(slim.pool(pool_4, [8, 8], stride=[1, 1], pooling_type='AVG'))
    pool_list.append(slim.pool(pool_4, [3, 3], stride=[1, 1], pooling_type='AVG'))
    pool_list.append(slim.pool(pool_4, [1, 1], stride=[1, 1], pooling_type='AVG'))

    return pool_list

def get_conv(pools):
    trans_list = []
    shape = [-1, 14, 14, 128]
    strides = [ 12, 6, 4, 3, 2, 2, 1,1]
    for i in range(len(pools)):
        conv = ConvBlock(pools[i], 128)
        conv = slim.batch_norm(conv)
        trans = desconv_new(conv, 128, stride=[strides[0], strides[0]])#, res=True)
        #trans = desconv_new(conv, 128, stride=[strides[i], strides[i]], shape=shape)
        trans_list.append(trans)
    return tf.concat(trans_list, axis=-1)

def upsampling(inputs, feature_map):
	return tf.image.resize_bilinear(inputs, size=feature_map)


def interblock(inputs, level, feature_map, pooling_type):
	kernel_size = [feature_map[0]//level, feature_map[1]//level]
	stride_size = kernel_size
	pool = slim.pool(inputs, kernel_size, stride=stride_size, pooling_type=pooling_type)
	conv = ConvBlock(pool, 128, kernel_size=[1, 1])
	#conv = desconv(conv, 128, stride=feature_map)
	conv = upsampling(conv, feature_map)
	conv = slim.conv2d(conv, 128, kernel_size=[3, 3])
	return conv

def PSPnet(inputs, pooling_type='MAX'):
	feature_map = [128/8, 128/8]
	net_1 = interblock(inputs, 1, feature_map, pooling_type)
	net_2 = interblock(inputs, 11, feature_map, pooling_type)
	net_3 = interblock(inputs, 3, feature_map, pooling_type)
	net_4 = interblock(inputs, 8, feature_map, pooling_type)
	net_5 = interblock(inputs, 9, feature_map, pooling_type)
	net_6 = interblock(inputs, 12, feature_map, pooling_type)
	net_7 = interblock(inputs, 13, feature_map, pooling_type)
	net_8 = interblock(inputs, 14, feature_map, pooling_type)
	net_9 = interblock(inputs, 15, feature_map, pooling_type)
	net_10 = interblock(inputs, 16, feature_map, pooling_type)
	net = tf.concat([net_1, net_2, net_3, net_4, net_5, net_6, net_7, net_8, net_9, net_10], axis=-1)
	return net


def build_unet2(inputs, num_classes):
	 #####################
	# Downsampling path #
	#####################
	strides = [1, 2, 2, 1]

	pool = [1, 2, 2, 1]
	#net = ConvBlock(inputs, 64)
	#net = slim.conv2d(inputs, 64, kernel_size=[3, 3], activation_fn=None)
	#net = tf.nn.relu(net)

	net = ConvBlock(inputs, 64)
	net = ConvBlock(net, 64)
	skip_1 = net
	net = ConvBlock(net, 64)
	net = tf.nn.max_pool(net, pool, strides=strides, padding='VALID')

	net = ConvBlock(net, 128)
	net = ConvBlock(net, 128)
	skip_2 = net
	net = ConvBlock(net, 128)
	net = tf.nn.max_pool(net, pool, strides=strides, padding='VALID')


	net = ConvBlock(net, 256)
	net = ConvBlock(net, 256)
	skip_3 = net
	net = ConvBlock(net, 256)
	net = tf.nn.max_pool(net, pool, strides=strides, padding='VALID')


	net = ConvBlock(net, 512)
	net = ConvBlock(net, 512)
	skip_4 = net
	net = ConvBlock(net, 512)
	net = tf.nn.max_pool(net, pool, strides=strides, padding='VALID')

		#####################
	# Upsampling path #
	#####################

	net = conv_transpose_block(net, 512)
	net = tf.concat([skip_4, net],  axis=3, name='fusion1')
	#net = tf.add(net, skip_4)
	net = ConvBlock(net, 512)
	net = ConvBlock(net, 512)
	#net = tf.nn.dropout(net, keep_prob)

	net = conv_transpose_block(net, 256)
	#net = tf.add(net, skip_3)
	net = tf.concat([skip_3, net],  axis=3, name='fusion2')
	net = ConvBlock(net, 256)
	net = ConvBlock(net, 256)
	#net = tf.nn.dropout(net, keep_prob)

	net = conv_transpose_block(net, 128)
	#net = tf.add(net, skip_2)
	net = tf.concat([skip_2, net],  axis=3, name='fusion3')
	net = ConvBlock(net, 128)
	net = ConvBlock(net, 128)
	#net = tf.nn.dropout(net, keep_prob)

	net = conv_transpose_block(net, 64)
	#net = tf.add(net, skip_2)
	net = tf.concat([skip_1, net],  axis=3, name='fusion3')
	net = ConvBlock(net, 64)
	net = ConvBlock(net, 64)


	#####################
	#      Softmax      #
	#####################

	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	#net = tf.nn.sigmoid(net)
	return net
