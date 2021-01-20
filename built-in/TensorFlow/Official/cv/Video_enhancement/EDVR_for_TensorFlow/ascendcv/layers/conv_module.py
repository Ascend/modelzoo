import math

import tensorflow as tf

from .conv import Conv2D
from .act import ActLayer
from ..utils.initializer import get_initializer, calculate_fan


def ConvModule(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilations=(1, 1), use_bias=True,
               kernel_initializer=None, bias_initializer=None, act_cfg=None,
               trainable=True, name='Conv2D'):

    if act_cfg is not None:
        nonlinearity = act_cfg.get('type').lower()
        if nonlinearity == 'leakyrelu':
            a = act_cfg.get('alpha', 0.01)
        else:
            nonlinearity = 'relu'
            a = 0
        if kernel_initializer is None:
            kernel_initializer = get_initializer(
                dict(type='kaiming_uniform', a=a, nonlinearity=nonlinearity), int(x.shape[-1]), filters, kernel_size)

    x = Conv2D(x, filters, kernel_size, strides, padding, dilations, use_bias,
               kernel_initializer=kernel_initializer, bias_initializer=None,
               trainable=True, name=name)

    if act_cfg is not None:
        x = ActLayer(act_cfg)(x)

    return x
