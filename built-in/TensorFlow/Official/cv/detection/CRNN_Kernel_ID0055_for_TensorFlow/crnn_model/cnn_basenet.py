#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-18 下午3:59
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : cnn_basenet.py
# @IDE: PyCharm Community Edition
"""
The base convolution neural networks mainly implement some useful cnn functions
"""
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.contrib.framework import add_model_variable
import numpy as np


class CNNBaseModel(object):
    """
    Base model for other specific cnn ctpn_models
    """

    def __init__(self):
        pass

    @staticmethod
    def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
               stride=1, w_init=None, b_init=None,
               split=1, use_bias=True, data_format='NHWC', name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param split: split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]

            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                    else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                    else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None
            
#            tf.add_to_collection("inputdata",w[0][0][0][:10])
            
            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, name=name)

        return ret

    @staticmethod
    def relu(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def sigmoid(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.sigmoid(x=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
                else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def avgpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        if stride is None:
            stride = kernel_size

        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
            else [1, 1, kernel_size, kernel_size]

        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def globalavgpooling(inputdata, data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param data_format:
        :return:
        """
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = [1, 2] if data_format == 'NHWC' else [2, 3]

        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name)

    @staticmethod
    def layernorm(inputdata, epsilon=1e-5, use_bias=True, use_scale=True,
                  data_format='NHWC', name=None):
        """
        :param name:
        :param inputdata:
        :param epsilon: epsilon to avoid divide-by-zero.
        :param use_bias: whether to use the extra affine transformation or not.
        :param use_scale: whether to use the extra affine transformation or not.
        :param data_format:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(inputdata, list(range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if ndims == 2:
            new_shape = [1, channnel]

        if use_bias:
            beta = tf.get_variable('beta', [channnel], initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [channnel], initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def instancenorm(inputdata, epsilon=1e-5, data_format='NHWC', use_affine=True, name=None):
        """

        :param name:
        :param inputdata:
        :param epsilon:
        :param data_format:
        :param use_affine:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError("Input data of instancebn layer has to be 4D tensor")

        if data_format == 'NHWC':
            axis = [1, 2]
            ch = shape[3]
            new_shape = [1, 1, 1, ch]
        else:
            axis = [2, 3]
            ch = shape[1]
            new_shape = [1, ch, 1, 1]
        if ch is None:
            raise ValueError("Input of instancebn require known channel!")

        mean, var = tf.nn.moments(inputdata, axis, keep_dims=True)

        if not use_affine:
            return tf.divide(inputdata - mean, tf.sqrt(var + epsilon), name='output')

        beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
        gamma = tf.get_variable('gamma', [ch], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def dropout(inputdata, keep_prob, is_training, name, noise_shape=None):
        """

        :param name:
        :param inputdata:
        :param keep_prob:
        :param is_training
        :param noise_shape:
        :return:
        """

        return tf.cond(
            pred=is_training,
            true_fn=lambda: tf.nn.dropout(
                inputdata, keep_prob=keep_prob, noise_shape=noise_shape
            ),
            false_fn=lambda: inputdata,
            name=name
        )

    @staticmethod
    def fullyconnect(inputdata, out_dim, w_init=None, b_init=None,
                     use_bias=True, name=None):
        """
        Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
        It is an equivalent of `tf.layers.dense` except for naming conventions.

        :param inputdata:  a tensor to be flattened except for the first dimension.
        :param out_dim: output dimension
        :param w_init: initializer for w. Defaults to `variance_scaling_initializer`.
        :param b_init: initializer for b. Defaults to zero
        :param use_bias: whether to use bias.
        :param name:
        :return: tf.Tensor: a NC tensor named ``output`` with attribute `variables`.
        """
        shape = inputdata.get_shape().as_list()[1:]
        if None not in shape:
            inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
        else:
            inputdata = tf.reshape(inputdata, tf.stack([tf.shape(inputdata)[0], -1]))

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=inputdata, activation=lambda x: tf.identity(x, name='output'),
                              use_bias=use_bias, name=name,
                              kernel_initializer=w_init,
                              bias_initializer=b_init,
                              trainable=True, units=out_dim)
        return ret

    @staticmethod
    def layerbn(inputdata, is_training, name, momentum=0.999, eps=1e-3):
        """

        :param inputdata:
        :param is_training:
        :param name:
        :param momentum:
        :param eps:
        :return:
        """

        return tf.layers.batch_normalization(
            inputs=inputdata, training=is_training, name=name, momentum=momentum, epsilon=eps)

    @staticmethod
    def layerbn_distributed(list_input, stats_mode, data_format='NHWC',
                            float_type=tf.float32, trainable=True,
                            use_gamma=True, use_beta=True, bn_epsilon=1e-5,
                            bn_ema=0.9, name='BatchNorm'):
        """
        Batch norm for distributed training process
        :param list_input:
        :param stats_mode:
        :param data_format:
        :param float_type:
        :param trainable:
        :param use_gamma:
        :param use_beta:
        :param bn_epsilon:
        :param bn_ema:
        :param name:
        :return:
        """

        def _get_bn_variables(_n_out, _use_scale, _use_bias, _trainable, _float_type):

            if _use_bias:
                _beta = tf.get_variable('beta', [_n_out],
                                        initializer=tf.constant_initializer(),
                                        trainable=_trainable,
                                        dtype=_float_type)
            else:
                _beta = tf.zeros([_n_out], name='beta')
            if _use_scale:
                _gamma = tf.get_variable('gamma', [_n_out],
                                         initializer=tf.constant_initializer(1.0),
                                         trainable=_trainable,
                                         dtype=_float_type)
            else:
                _gamma = tf.ones([_n_out], name='gamma')

            _moving_mean = tf.get_variable('moving_mean', [_n_out],
                                           initializer=tf.constant_initializer(),
                                           trainable=False,
                                           dtype=_float_type)
            _moving_var = tf.get_variable('moving_variance', [_n_out],
                                          initializer=tf.constant_initializer(1),
                                          trainable=False,
                                          dtype=_float_type)
            return _beta, _gamma, _moving_mean, _moving_var

        def _update_bn_ema(_xn, _batch_mean, _batch_var, _moving_mean, _moving_var, _decay):

            _update_op1 = moving_averages.assign_moving_average(
                _moving_mean, _batch_mean, _decay, zero_debias=False,
                name='mean_ema_op')
            _update_op2 = moving_averages.assign_moving_average(
                _moving_var, _batch_var, _decay, zero_debias=False,
                name='var_ema_op')
            add_model_variable(moving_mean)
            add_model_variable(moving_var)

            # seems faster than delayed update, but might behave otherwise in distributed settings.
            with tf.control_dependencies([_update_op1, _update_op2]):
                return tf.identity(xn, name='output')

        # ======================== Checking valid values =========================
        if data_format not in ['NHWC', 'NCHW']:
            raise TypeError(
                "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
                "%s is an unknown data format." % data_format)
        assert type(list_input) == list

        # ======================== Setting default values =========================
        shape = list_input[0].get_shape().as_list()
        assert len(shape) in [2, 4]
        n_out = shape[-1]
        if data_format == 'NCHW':
            n_out = shape[1]

        # ======================== Main operations =============================
        means = []
        square_means = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                batch_mean = tf.reduce_mean(list_input[i], [0, 1, 2])
                batch_square_mean = tf.reduce_mean(tf.square(list_input[i]), [0, 1, 2])
                means.append(batch_mean)
                square_means.append(batch_square_mean)

        # if your GPUs have NVLinks and you've install NCCL2, you can change `/cpu:0` to `/gpu:0`
        with tf.device('/cpu:0'):
            shape = tf.shape(list_input[0])
            num = shape[0] * shape[1] * shape[2] * len(list_input)
            mean = tf.reduce_mean(means, axis=0)
            var = tf.reduce_mean(square_means, axis=0) - tf.square(mean)
            var *= tf.cast(num, float_type) / tf.cast(num - 1, float_type)  # unbiased variance

        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(name, reuse=i > 0):
                    beta, gamma, moving_mean, moving_var = _get_bn_variables(
                        n_out, use_gamma, use_beta, trainable, float_type)

                    if 'train' in stats_mode:
                        xn = tf.nn.batch_normalization(
                            list_input[i], mean, var, beta, gamma, bn_epsilon)
                        if tf.get_variable_scope().reuse or 'gather' not in stats_mode:
                            list_output.append(xn)
                        else:
                            # gather stats and it is the main gpu device.
                            xn = _update_bn_ema(xn, mean, var, moving_mean, moving_var, bn_ema)
                            list_output.append(xn)
                    else:
                        xn = tf.nn.batch_normalization(
                            list_input[i], moving_mean, moving_var, beta, gamma, bn_epsilon)
                        list_output.append(xn)

        return list_output

    @staticmethod
    def layergn(inputdata, name, group_size=32, esp=1e-5):
        """

        :param inputdata:
        :param name:
        :param group_size:
        :param esp:
        :return:
        """
        with tf.variable_scope(name):
            inputdata = tf.transpose(inputdata, [0, 3, 1, 2])
            n, c, h, w = inputdata.get_shape().as_list()
            group_size = min(group_size, c)
            inputdata = tf.reshape(inputdata, [-1, group_size, c // group_size, h, w])
            mean, var = tf.nn.moments(inputdata, [2, 3, 4], keep_dims=True)
            inputdata = (inputdata - mean) / tf.sqrt(var + esp)

            # 每个通道的gamma和beta
            gamma = tf.Variable(tf.constant(1.0, shape=[c]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(0.0, shape=[c]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, c, 1, 1])
            beta = tf.reshape(beta, [1, c, 1, 1])

            # 根据论文进行转换 [n, c, h, w, c] 到 [n, h, w, c]
            output = tf.reshape(inputdata, [-1, c, h, w])
            output = output * gamma + beta
            output = tf.transpose(output, [0, 2, 3, 1])

        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        """

        :param inputdata:
        :param axis:
        :param name:
        :return:
        """
        return tf.squeeze(input=inputdata, axis=axis, name=name)

    @staticmethod
    def deconv2d(inputdata, out_channel, kernel_size, padding='SAME',
                 stride=1, w_init=None, b_init=None,
                 use_bias=True, activation=None, data_format='channels_last',
                 trainable=True, name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param activation: whether to apply a activation func to deconv result
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :param trainable:
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'channels_last' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            ret = tf.layers.conv2d_transpose(inputs=inputdata, filters=out_channel,
                                             kernel_size=kernel_size,
                                             strides=stride, padding=padding,
                                             data_format=data_format,
                                             activation=activation, use_bias=use_bias,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init, trainable=trainable,
                                             name=name)
        return ret

    @staticmethod
    def dilation_conv(input_tensor, k_size, out_dims, rate, padding='SAME',
                      w_init=None, b_init=None, use_bias=False, name=None):
        """

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param rate:
        :param padding:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]

            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if isinstance(k_size, list):
                filter_shape = [k_size[0], k_size[1]] + [in_channel, out_dims]
            else:
                filter_shape = [k_size, k_size] + [in_channel, out_dims]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_dims], initializer=b_init)

            conv = tf.nn.atrous_conv2d(value=input_tensor, filters=w, rate=rate,
                                       padding=padding, name='dilation_conv')

            if use_bias:
                ret = tf.add(conv, b)
            else:
                ret = conv

        return ret

    @staticmethod
    def spatial_dropout(input_tensor, keep_prob, is_training, name, seed=1234):
        """
        空间dropout实现
        :param input_tensor:
        :param keep_prob:
        :param is_training:
        :param name:
        :param seed:
        :return:
        """

        def f1():
            input_shape = input_tensor.get_shape().as_list()
            noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
            return tf.nn.dropout(input_tensor, keep_prob, noise_shape, seed=seed, name="spatial_dropout")

        def f2():
            return input_tensor

        with tf.variable_scope(name_or_scope=name):

            output = tf.cond(is_training, f1, f2)

            return output

    @staticmethod
    def lrelu(inputdata, name, alpha=0.2):
        """

        :param inputdata:
        :param alpha:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            return tf.nn.relu(inputdata) - alpha * tf.nn.relu(-inputdata)

    @staticmethod
    def pad(inputdata, paddings, name):
        """

        :param inputdata:
        :param paddings:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            return tf.pad(tensor=inputdata, paddings=paddings)
