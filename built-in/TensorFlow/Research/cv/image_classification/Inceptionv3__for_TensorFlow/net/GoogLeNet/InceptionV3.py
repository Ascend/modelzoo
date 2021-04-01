#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by InceptionV3 on 19-4-4

import tensorflow as tf
import tensorflow.contrib.slim as slim


"""
论文中提到的第一种模块结构,stride都是1,等效于三层
但目前google源码可以看到和这里是不一样的，而是用了inceptionv1的结构
但论文的图表中则明确指出了是拆分了5x5的卷积为两个3x3，网上基本上全部抄的google源码
我这里就按照论文复现,这里体现了原则3
"""
def inception_module_v3_1(net, scope, filter_num, stride=1):
    with tf.variable_scope(scope):
        with tf.variable_scope('bh1'):
            bh1 = slim.conv2d(net, filter_num[0], [1, 1], stride=stride, scope="bh1_conv1_1x1")
        with tf.variable_scope('bh2'):
            bh2 = slim.avg_pool2d(net, [3, 3], stride=stride, scope="bh2_avg_3x3")
            bh2 = slim.conv2d(bh2, filter_num[1], [1, 1], stride=stride, scope="bh2_conv_1x1")
        with tf.variable_scope('bh3'):
            bh3 = slim.conv2d(net, filter_num[2], [1, 1], stride=stride, scope="bh3_conv1_1x1")
            bh3 = slim.conv2d(bh3, filter_num[3], [3, 3], stride=stride, scope="bh3_conv2_3x3")
        with tf.variable_scope('bh4'):
            bh4 = slim.conv2d(net, filter_num[4], [1, 1], stride=stride, scope="bh4_conv1_1x1")
            bh4 = slim.conv2d(bh4, filter_num[5], [3, 3], stride=stride, scope="bh4_conv2_3x3")
            bh4 = slim.conv2d(bh4, filter_num[6], [3, 3], stride=stride, scope="bh4_conv3_3x3")
        net = tf.concat([bh1, bh2, bh3, bh4], axis=3)
    return net


'''
论文中提到的第二种结构,使用了1xn和nx1,论文中将n=7用来处理17x17的grid，五层
这里体现了原则3
'''
def inception_moudle_v3_2(net, scope, filter_num, stride=1):
    with tf.variable_scope(scope):
        with tf.variable_scope("bh1"):
            bh1 = slim.conv2d(net, filter_num[0], [1, 1], stride=stride, scope="bh1_conv_1x1")
        with tf.variable_scope("bh2"):
            bh2 = slim.avg_pool2d(net, [3, 3], stride=stride, scope='bh2_avg_3x3')
            bh2 = slim.conv2d(bh2, filter_num[1], [1, 1], stride=stride, scope='bh2_conv_1x1')
        with tf.variable_scope("bh3"):
            bh3 = slim.conv2d(net, filter_num[2], [1, 1], stride=stride, scope='bh3_conv1_1x1')
            bh3 = slim.conv2d(bh3, filter_num[3], [1, 7], stride=stride, scope='bh3_conv2_1x7')
            bh3 = slim.conv2d(bh3, filter_num[4], [7, 1], stride=stride, scope='bh3_conv3_7x1')
        with tf.variable_scope("bh4"):
            bh4 = slim.conv2d(net, filter_num[5], [1, 1], stride=stride, scope='bh4_conv1_1x1')
            bh4 = slim.conv2d(bh4, filter_num[6], [1, 7], stride=stride, scope='bh4_conv2_1x7')
            bh4 = slim.conv2d(bh4, filter_num[7], [7, 1], stride=stride, scope='bh4_conv3_7x1')
            bh4 = slim.conv2d(bh4, filter_num[8], [1, 7], stride=stride, scope='bh4_conv4_1x7')
            bh4 = slim.conv2d(bh4, filter_num[9], [7, 1], stride=stride, scope='bh4_conv5_7x1')
        net = tf.concat([bh1, bh2, bh3, bh4], axis=3)
    return net


'''
论文提到的第三种结构，增加了宽度,三层
体现了原则2
'''
def inception_moudle_v3_3(net, scope, filter_num, stride=1):
    with tf.variable_scope(scope):
        with tf.variable_scope("bh1"):
            bh1 = slim.conv2d(net, filter_num[0], [1, 1], stride=stride, scope='bh1_conv_1x1')
        with tf.variable_scope("bh2"):
            bh2 = slim.avg_pool2d(net, [3, 3], stride=stride, scope='bh2_avg_3x3')
            bh2 = slim.conv2d(bh2, filter_num[1], [1, 1], stride=stride, scope='bh2_conv_1x1')
        with tf.variable_scope("bh3"):
            bh3 = slim.conv2d(net, filter_num[2], [1, 1], stride=stride, scope='bh3_conv1_1x1')
            bh3_1 = slim.conv2d(bh3, filter_num[3], [3, 1], stride=stride, scope='bh3_conv2_3x1')
            bh3_2 = slim.conv2d(bh3, filter_num[4], [1, 3], stride=stride, scope='bh3_conv2_1x3')
        with tf.variable_scope("bh4"):
            bh4 = slim.conv2d(net, filter_num[5], [1, 1], stride=stride, scope='bh4_conv1_1x1')
            bh4 = slim.conv2d(bh4, filter_num[6], [3, 3], stride=stride, scope='bh4_conv2_3x3')
            bh4_1 = slim.conv2d(bh4, filter_num[7], [3, 1], stride=stride, scope='bh4_conv3_3x1')
            bh4_2 = slim.conv2d(bh4, filter_num[8], [1, 3], stride=stride, scope='bh4_conv3_1x3')
        net = tf.concat([bh1, bh2, bh3_1, bh3_2, bh4_1, bh4_2], axis=3)
    return net


'''
论文中提到用来减少grid-size的inception模块
等效三层,pad为VALID
体现了原则1
'''
def inception_moudle_v3_reduce(net, scope, filter_num):
    with tf.variable_scope(scope):
        with tf.variable_scope("bh1"):
            bh1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',scope="bh1_max_3x3")
        with tf.variable_scope("bh2"):
            bh2 = slim.conv2d(net, filter_num[0], [1, 1], stride=1, scope='bh2_conv1_1x1')
            bh2 = slim.conv2d(bh2, filter_num[1], [3, 3], stride=2, padding='VALID', scope='bh2_conv2_3x3')
        with tf.variable_scope("bh3"):
            bh3 = slim.conv2d(net, filter_num[2], [1, 1], stride=1, scope='bh3_conv1_1x1')
            bh3 = slim.conv2d(bh3, filter_num[3], [3, 3], stride=1, scope='bh3_conv2_3x3')
            bh3 = slim.conv2d(bh3, filter_num[4], [3, 3], stride=2, padding='VALID', scope='bh3_conv3_3x3')
        net = tf.concat([bh1, bh2, bh3], axis=3)
    return net


def V3_slim(inputs, num_cls, keep_prob=0.8, is_training=True, spatital_squeeze=True):
    batch_norm_params = {
        'decay': 0.998,
        'epsilon': 0.001,
        'scale': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    net = inputs
    with tf.name_scope('reshape'):
        net = tf.reshape(net, [-1, 299, 299, 3])

    with tf.variable_scope('GoogLeNet_V3'):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_initializer=slim.xavier_initializer(),
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
                with slim.arg_scope(
                        [slim.batch_norm, slim.dropout], is_training=is_training):
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
                        net = slim.conv2d(net,32,[3,3], stride=2,scope="layer1")             #149x149
                        net = slim.conv2d(net,32,[3,3], scope='layer2')                      #147x147
                        net = slim.conv2d(net,64,[3,3], padding='SAME',scope='layer3')       #147x147
                        net = slim.max_pool2d(net,[3,3], stride=2,scope='layer4')            #73x73
                        net = slim.conv2d(net,80,[3,3], scope='layer5')                      #71x71
                        net = slim.conv2d(net,192,[3,3], stride=2,scope='layer6')            #35x35

                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                        net = slim.conv2d(net, 288, [3,3], scope='layer7')
                        # 3 x inception
                        net = inception_module_v3_1(net, scope='layer8',filter_num=[64,32,48,64,64,96,96])       #35x35
                        net = inception_module_v3_1(net, scope='layer11',filter_num=[64,64,48,64,64,96,96])
                        net = inception_module_v3_1(net, scope='layer14',filter_num=[64,64,48,64,64,96,96])
                        print(net)
                        # 5 x inception
                        net = inception_moudle_v3_reduce(net, scope='layer17',filter_num=[192,384,64,96,96])  #17x17
                        net = inception_moudle_v3_2(net, scope='layer20',filter_num=[192,192,128,128,192,128,128,128,128,192])
                        net = inception_moudle_v3_2(net, scope='layer25',filter_num=[192,192,160,160,192,160,160,160,160,192])
                        net = inception_moudle_v3_2(net, scope='layer30',filter_num=[192,192,160,160,192,160,160,160,160,192])
                        net = inception_moudle_v3_2(net, scope='layer35',filter_num=[192,192,160,160,192,160,160,160,160,192])
                        print(net)
                        # 3 x inception
                        net = inception_moudle_v3_reduce(net, scope='layer40',filter_num=[192,320,192,192,192])  #8x8
                        net = inception_moudle_v3_3(net,scope='layer43',filter_num=[320,192,384,384,384,448,384,384,384])
                        net = inception_moudle_v3_3(net,scope='layer46',filter_num=[320,192,384,384,384,448,384,384,384])
                        print(net)
                        net = slim.avg_pool2d(net,[8,8],padding='VALID',scope='layer49')
                        net = slim.dropout(net)
                        net = slim.conv2d(net,num_cls,[1,1],activation_fn=None,normalizer_fn=None,scope='layer50')
                        print(net)
                        if spatital_squeeze:
                            net = tf.squeeze(net,[1,2],name='squeeze')

                        net = slim.softmax(net,scope='softmax')
                    return net



class testInceptionV3(tf.test.TestCase):
    def testBuildClassifyNetwork(self):
        inputs = tf.random_uniform((5,299,299,3))
        logits = V3_slim(inputs,10)
        print(logits)

if __name__ == '__main__':
    tf.test.main()
