#-*- coding:utf-8 -*-
"""
Multi_Adversarial Network
Author: AJ
Date: 2019/11/23
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from .triplet_loss import batch_all_triplet_loss
from .triplet_loss import batch_hard_triplet_loss

def lrelu(x):
    return tf.maximum(x * 0.2, x)
def weigth_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)
def bias_variable(shape,name):
    """Initialization of bias term."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial,name=name)
def conv2d(x, w, name):
    """Definition of convolutional operator."""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name)
def max_pool(x,name):
    """Definition of max-pooling."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap
def fully_conneted(x, units, use_bias=False, scope='fc'):
    import tensorflow.contrib as tf_contrib
    weight_init = tf_contrib.layers.variance_scaling_initializer()
    weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)
    def flatten(x):
        return tf.layers.flatten(x)
    with tf.compat.v1.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x,
                            units=units, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x
def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])
    return deconv_output

def FeatureGenerator(image_Batch, is_training):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(image_Batch, 64, [3, 3], stride=[1, 1], scope='conv1_0', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_0')
        ### block 1 ###
        net = slim.conv2d(net, 128, [3, 3], stride=[1, 1], scope='conv1_1', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_1')
        net = slim.conv2d(net, 196, [3, 3], stride=[1, 1], scope='conv1_2', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_2')
        net = slim.conv2d(net, 128, [3, 3], stride=[1, 1], scope='conv1_3', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_3')
        pool1_1 = slim.max_pool2d(net, [2, 2], stride=[2, 2], scope='pool1_1')

        ### block 2 ###
        net = slim.conv2d(pool1_1, 128, [3, 3], stride=[1, 1], scope='conv1_4', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_4')
        net = slim.conv2d(net, 196, [3, 3], stride=[1, 1], scope='conv1_5', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_5')
        net = slim.conv2d(net, 128, [3, 3], stride=[1, 1], scope='conv1_6', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_6')
        pool1_2 = slim.max_pool2d(net, [2, 2], stride=[2, 2], scope='pool1_2')

        ### block 3 ###
        net = slim.conv2d(pool1_2, 128, [3, 3], stride=[1, 1], scope='conv1_7', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_7')
        net = slim.conv2d(net, 196, [3, 3], stride=[1, 1], scope='conv1_8', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_8')
        net = slim.conv2d(net, 128, [3, 3], stride=[1, 1], scope='conv1_9', padding='SAME')
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None, scope='bn1_9')
        pool1_3 = slim.max_pool2d(net, [2, 2], stride=[2, 2], scope='pool1_3')
        ### concat 3 block feature ###
        short_cut_1 = tf.compat.v1.image.resize_bilinear(pool1_1, size=(pool1_3.shape[2], pool1_3.shape[2]))
        short_cut_2 = tf.compat.v1.image.resize_bilinear(pool1_2, size=(pool1_3.shape[2], pool1_3.shape[2]))
        pool_concat = tf.concat([short_cut_1, short_cut_2, pool1_3], axis=-1)
    return pool1_3, pool_concat

def DepthEstimator(pool_concat, label_dim, is_training, depth_size):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        ### followed by subsequent three convolutional grops ###
        net_depth = slim.conv2d(pool_concat, 128, [3, 3], stride=[1, 1], scope='conv4_1', padding='SAME')
        net_depth = slim.batch_norm(net_depth, is_training=is_training, activation_fn=None, scope='bn4_1')
        net_depth = slim.conv2d(net_depth, 64, [3, 3], stride=[1, 1], scope='conv4_2', padding='SAME')
        net_depth = slim.batch_norm(net_depth, is_training=is_training, activation_fn=None, scope='bn4_2')
        net_depth = slim.conv2d(net_depth, 1, [3, 3], stride=[1, 1], scope='conv4_3', padding='SAME')
        depth_map = slim.batch_norm(net_depth, is_training=is_training, activation_fn=None, scope='bn4_3')
        ### DepthMap_2_embeddings_&_logits ###
        embeddings = tf.reshape(depth_map, [-1, depth_size * depth_size * 1])
        prelogits = tf.reshape(depth_map, [-1, 1, 1, depth_size * depth_size * 1])             # (?, 1, 1, 1024)
        logits = fully_conneted(prelogits, units=label_dim, scope='fc_logit')  # (?, 2)
    return depth_map, embeddings, logits

def build_Multi_Adversarial_Loss(color_batch, depth_label_batch, label_batch, domain_batch, label_dim, alpha_beta_gamma,
                       depth_size=32, triplet_strategy='batch_all', margin=0.5, isTraining=False, order=1, reuse=False):
    """Build the inference graph """
    def show_trainable_vars(k=0):
        vars = tf.trainable_variables()
        print('show: {}'.format(k))
        for var in vars:
            print(var.name)
    def get_bin_cla_loss(logits, label_batch):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        #### Norm for the prelogits
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        total_loss = cross_entropy_mean + reg_loss * 0.0
        return accuracy, total_loss
    ### Multi_Adversarial ###
    with tf.compat.v1.variable_scope('FG_color', reuse=reuse):
        _, pool_concat_color = FeatureGenerator(color_batch, isTraining)
    with tf.compat.v1.variable_scope('DE_color', reuse=reuse):
        depth_map, embeddings_color, logits_color = DepthEstimator(pool_concat_color, label_dim, isTraining, depth_size)
    # show_trainable_vars()
    ### @1: binary classification ###
    accuracy_color, bin_cla_loss_color = get_bin_cla_loss(logits_color, label_batch)
    accuracy_rp, bin_cla_loss_rp = 0, 0
    ### @2: Depth loss ###
    depth_label = tf.split(depth_label_batch,  num_or_size_splits=3, axis=-1)[0] ### use the first channel for conveniently
    if order == 1:
        depth_loss_1 = tf.reduce_mean(tf.abs(depth_map - depth_label))
    elif order == 2:
        depth_loss_1 = tf.reduce_mean(tf.pow(depth_map - depth_label, 2))
    else:print('order is error!')
    depth_loss = depth_loss_1
    ### @3: Triplet loss based on online mining ###
    if triplet_strategy == 'batch_all':
        triplet_loss, fraction, pairwise_dist = batch_all_triplet_loss(label_batch, embeddings_color, margin, squared=False)
    elif triplet_strategy == 'batch_hard':
        triplet_loss, pairwise_dist = batch_hard_triplet_loss(label_batch, embeddings_color, margin, squared=False)
        fraction = tf.constant(0, dtype=tf.double)
    else:raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))
    ### Calculate the total losses
    total_loss = alpha_beta_gamma[0] * (bin_cla_loss_color + bin_cla_loss_rp) + \
                 alpha_beta_gamma[1] * depth_loss + \
                 alpha_beta_gamma[2] * triplet_loss
    return logits_color, embeddings_color, depth_map, accuracy_color, total_loss, bin_cla_loss_color, \
           depth_loss, triplet_loss, fraction, pairwise_dist



