import numpy as np

import tensorflow as tf


def Huber(y_true, y_pred, delta, reduction='mean', axis=None):
    abs_error = tf.abs(y_pred - y_true)
    quadratic = tf.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    if reduction == 'mean':
        return tf.reduce_mean(losses, axis=axis)
    elif reduction == 'sum':
        return tf.reduce_sum(losses, axis=axis)
    else:
        raise NotImplementedError


def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], ds_x[4]])

    y = tf.depth_to_space(x, block_size)

    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x


def DynFilter3D(x, F, filter_size):
    """
    3D Dynamic filtering
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    """
    # make tower
    with tf.variable_scope('DynFilter3D', reuse=tf.AUTO_REUSE) as scope:
        filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)),
                                           (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
        filter_localexpand = tf.Variable(filter_localexpand_np, dtype='float32',
                                         name='filter_localexpand')  # ,trainable=False)
        # filter_localexpand = tf.get_variable('filter_localexpand', initializer=filter_localexpand_np)
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x_localexpand = tf.nn.conv2d(x, filter_localexpand, [1, 1, 1, 1], 'SAME')  # b, h, w, 1*5*5
        x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, h, w, 1, 1*5*5
        x = tf.matmul(x_localexpand, F)  # b, h, w, 1, R*R
        x = tf.squeeze(x, axis=3)  # b, h, w, R*R

    return x
