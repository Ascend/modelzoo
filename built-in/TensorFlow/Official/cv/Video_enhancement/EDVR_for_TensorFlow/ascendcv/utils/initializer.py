import math

import tensorflow as tf


def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leakyrelu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def calculate_fan(kernel_size, in_channels, out_channels=None, mode='fan_in'):
    if mode == 'fan_in':
        fan = in_channels
    elif mode == 'fan_out':
        fan = out_channels
    else:
        raise KeyError
    for k in kernel_size:
        fan *= k
    return fan


def get_initializer(init_cfg, in_channels, out_channels, kernel_size):

    type = init_cfg.pop('type')

    if type == 'kaiming_uniform':
        a = init_cfg.pop('a', 0)
        mode = init_cfg.pop('mode', 'fan_in')
        nonlinearity = init_cfg.pop('nonlinearity', 'leakyrelu')
        fan = calculate_fan(kernel_size, in_channels, out_channels, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        initializer = tf.random_uniform_initializer(-bound, bound)
    elif type == 'kaiming_normal':
        a = init_cfg.pop('a', 0)
        mode = init_cfg.pop('mode', 'fan_in')
        nonlinearity = init_cfg.pop('nonlinearity', 'leakyrelu')
        fan = calculate_fan(kernel_size, in_channels, out_channels, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        initializer = tf.random_normal_initializer(0.0, std)
    elif type == 'xavier_uniform':
        gain = init_cfg.pop('gain', 1.)
        fan_in = calculate_fan(kernel_size, in_channels, out_channels, 'fan_in')
        fan_out = calculate_fan(kernel_size, in_channels, out_channels, 'fan_out')
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        initializer = tf.random_uniform_initializer(-a, a)
    elif type == 'xavier_normal':
        gain = init_cfg.pop('gain', 1.)
        fan_in = calculate_fan(kernel_size, in_channels, out_channels, 'fan_in')
        fan_out = calculate_fan(kernel_size, in_channels, out_channels, 'fan_out')
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        initializer = tf.random_normal_initializer(0.0, std)
    else:
        raise NotImplementedError

    return initializer
