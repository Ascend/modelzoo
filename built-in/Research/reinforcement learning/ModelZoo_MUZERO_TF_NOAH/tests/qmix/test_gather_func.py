#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
develop the pytorch.gather function with tensorflow
"""

import tensorflow as tf
import numpy as np


def gather4d_on_dim3(inputs, indices):
    """
    gather 4dim tensor into 3dim, same to the pytorch.gather + sequeeze(3) function.
    :param inputs:
    :param indices:
    :return:
    """
    len_0d, len_1d, len_2d, len_3d = inputs.get_shape()
    inputs = tf.reshape(inputs, [-1, len_3d])
    calc_0d = inputs.get_shape()[0]

    flag_0d, flag_1d, flag_2d, flag_3d = indices.get_shape()
    indices = tf.reshape(indices, [-1, flag_3d])

    idx_matrix = tf.tile(
        tf.expand_dims(tf.range(0, len_3d, dtype=indices.dtype), 0), [calc_0d, 1]
    )
    indices_t = tf.transpose(indices)
    idx_mask = tf.equal(idx_matrix, tf.transpose(indices_t))

    inputs = tf.reshape(tf.boolean_mask(inputs, idx_mask), [flag_0d, flag_1d, flag_2d])
    return inputs


with tf.Session() as sess:
    x = tf.reshape(tf.range(0, 5000000), [50, 400, 25, 10])
    y = tf.constant(
        np.array([[1], [2], [3], [4], [5]] * 100000).reshape([50, 400, 25, 1])
    )
    #     out, idx, mask, indices_t = torch_gather_2d(x, y)
    out = gather4d_on_dim3(x, y)

    rx = sess.run(x)
    ry = sess.run(y)
    rout = sess.run(out)

    print("x:\n", np.shape(rx))

    print("y:\n", np.shape(ry))

    print("out:\n", np.shape(rout))
    print(rx[-1, -1, -10:])
    print(rout[-1, -1, -10:])
