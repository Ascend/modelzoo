import math

import tensorflow as tf

from ..utils.initializer import get_initializer, calculate_fan


def Conv2D(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilations=(1, 1), use_bias=True,
           kernel_initializer=None, bias_initializer=None,
           trainable=True, name='Conv2D'):

    if kernel_initializer is None:
        kernel_initializer = get_initializer(
            dict(type='kaiming_uniform', a=math.sqrt(5)), int(x.shape[-1]), filters, kernel_size)
    if bias_initializer is None:
        fan = calculate_fan(kernel_size, int(x.shape[-1]))
        bound = 1 / math.sqrt(fan)
        bias_initializer = tf.random_uniform_initializer(-bound, bound)

    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding.lower(),
        dilation_rate=dilations,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        name=name,
    )
    return x


def Conv3D(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', dilations=(1, 1, 1), use_bias=True,
           kernel_initializer=None, bias_initializer=None,
           trainable=True, name='Conv2D'):

    if kernel_initializer is None:
        kernel_initializer = get_initializer(
            dict(type='kaiming_uniform', a=math.sqrt(5)), int(x.shape[-1]), filters, kernel_size)
    if bias_initializer is None:
        fan = calculate_fan(kernel_size, int(x.shape[-1]))
        bound = 1 / math.sqrt(fan)
        bias_initializer = tf.random_uniform_initializer(-bound, bound)

    x = tf.layers.conv3d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding.lower(),
        dilation_rate=dilations,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        name=name,
    )
    return x
