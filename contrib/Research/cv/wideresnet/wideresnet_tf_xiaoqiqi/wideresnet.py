#-*-coding:utf-8-*-
import os
import tensorflow as tf

def residual_block(x, in_plane,out_plane, training, stride=1):
        out =x

        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=out_plane, kernel_size=3, strides=1, padding='same', use_bias=True)


        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.3, training=training)

        x = tf.layers.conv2d(x, filters=out_plane, kernel_size=3, strides=stride, padding='same', use_bias=True)

        if stride!=1 or in_plane!=out_plane:
            out = tf.layers.conv2d(out, filters=out_plane, kernel_size=1, strides=stride, use_bias=True)
        add = tf.add(x, out)


        return add


def wide_resnet(image_batch, training, depths, ks):
    n = int((depths - 4) / 6)
    k = ks
    wide = [16, 16 * k, 32 * k, 64 * k]

    net= tf.layers.conv2d(image_batch, filters=wide[0], kernel_size=3, strides=1, padding='same')
    print(net.shape)

    net = residual_block(net, wide[0], wide[1], training, 1)
    for i in range(1, n):
        net = residual_block(net, wide[1], wide[1], training, 1)
    print(net.shape)

    net = residual_block(net, wide[1], wide[2], training,  2)
    for i in range(1, n):
       net = residual_block(net, wide[2], wide[2],  training,1)
    print(net.shape)

    net = residual_block(net,wide[2], wide[3], training, 2)
    for i in range(1, n):
        net = residual_block(net,wide[3], wide[3], training,1)
    print(net.shape)

    net = tf.layers.batch_normalization(net,momentum=0.9, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.average_pooling2d(net, pool_size=8, strides=1)
    flatten = tf.layers.flatten(net)
    output = tf.layers.dense(flatten, units=100, activation=None, name="output")
    return output

