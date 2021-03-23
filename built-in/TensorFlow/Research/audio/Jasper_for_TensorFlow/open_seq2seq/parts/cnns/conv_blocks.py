# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
from .tcn import tcn

layers_dict = {
    "conv1d": tf.layers.conv1d,
    "sep_conv1d": tf.layers.separable_conv1d,
    "conv2d": tf.layers.conv2d,
    "tcn": tcn,
}


def conv_actv(layer_type, name, inputs, filters, kernel_size, activation_fn,
              strides, padding, regularizer, training, data_format, dilation=1):
  """Helper function that applies convolution and activation.
    Args:
      layer_type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[layer_type]

  if layer_type == 'sep_conv1d':
    conv = layer(
        name="{}".format(name),
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
        use_bias=False,
        data_format=data_format,
    )
  else:
    conv = layer(
        name="{}".format(name),
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation,
        kernel_regularizer=regularizer,
        use_bias=False,
        data_format=data_format,
    )

  output = conv
  if activation_fn is not None:
    output = activation_fn(output)
  return output

def conv_bn_res_bn_actv(layer_type, name, inputs, res_inputs, filters,
                        kernel_size, activation_fn, strides, padding,
                        regularizer, training, data_format, bn_momentum,
                        bn_epsilon, dilation=1,
                        drop_block_prob=0.0, drop_block=False):
  layer = layers_dict[layer_type]

  if not isinstance(res_inputs, list):
    res_inputs = [res_inputs]
    # For backwards compatibiliaty with earlier models
    res_name = "{}/res"
    res_bn_name = "{}/res_bn"
  else:
    res_name = "{}/res_{}"
    res_bn_name = "{}/res_bn_{}"

  res_aggregation = 0
  for i, res in enumerate(res_inputs):
    res = layer(
        res,
        filters,
        1,
        name=res_name.format(name, i),
        use_bias=False,
    )
    squeeze = False
    if "conv1d" in layer_type:
      axis = 1 if data_format == 'channels_last' else 2
      res = tf.expand_dims(res, axis=axis)  # NWC --> NHWC
      squeeze = True
    res = tf.layers.batch_normalization(
        name=res_bn_name.format(name, i),
        inputs=res,
        gamma_regularizer=regularizer,
        training=training,
        axis=-1 if data_format == 'channels_last' else 1,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
    )
    if squeeze:
      res = tf.squeeze(res, axis=axis)

    res_aggregation += res

  if layer_type == "sep_conv1d":
    conv = layer(
        name="{}".format(name),
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
        use_bias=False,
        data_format=data_format,
    )
  else:
    conv = layer(
        name="{}".format(name),
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation,
        kernel_regularizer=regularizer,
        use_bias=False,
        data_format=data_format,
    )

  # trick to make batchnorm work for mixed precision training.
  # To-Do check if batchnorm works smoothly for >4 dimensional tensors
  squeeze = False
  if "conv1d" in layer_type:
    axis = 1 if data_format == 'channels_last' else 2
    conv = tf.expand_dims(conv, axis=axis)  # NWC --> NHWC
    squeeze = True

  bn = tf.layers.batch_normalization(
      name="{}/bn".format(name),
      inputs=conv,
      gamma_regularizer=regularizer,
      training=training,
      axis=-1 if data_format == 'channels_last' else 1,
      momentum=bn_momentum,
      epsilon=bn_epsilon,
  )

  if squeeze:
    bn = tf.squeeze(bn, axis=axis)

  output = bn + res_aggregation

  if drop_block_prob > 0:
    if training:
      output = tf.cond(
        tf.random_uniform(shape=[]) < drop_block_prob,
        lambda: res_aggregation,
        lambda: bn + res_aggregation
        )
    elif drop_block:
      output = res_aggregation

  if activation_fn is not None:
    output = activation_fn(output)
  return output

def conv_bn_actv(layer_type, name, inputs, filters, kernel_size, activation_fn,
                 strides, padding, regularizer, training, data_format,
                 bn_momentum, bn_epsilon, dilation=1):
  """Helper function that applies convolution, batch norm and activation.
    Args:
      layer_type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[layer_type]

  if layer_type == 'sep_conv1d':
    conv = layer(
        name="{}".format(name),
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
        use_bias=False,
        data_format=data_format,
    )
  else:
    conv = layer(
        name="{}".format(name),
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation,
        kernel_regularizer=regularizer,
        use_bias=False,
        data_format=data_format,
    )

  # trick to make batchnorm work for mixed precision training.
  # To-Do check if batchnorm works smoothly for >4 dimensional tensors
  squeeze = False
  if "conv1d" in layer_type:
    axis = 1 if data_format == 'channels_last' else 2
    conv = tf.expand_dims(conv, axis=axis)  # NWC --> NHWC
    squeeze = True

  bn = tf.layers.batch_normalization(
      name="{}/bn".format(name),
      inputs=conv,
      gamma_regularizer=regularizer,
      training=training,
      axis=-1 if data_format == 'channels_last' else 1,
      momentum=bn_momentum,
      epsilon=bn_epsilon,
  )

  if squeeze:
    bn = tf.squeeze(bn, axis=axis)

  output = bn
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def conv_ln_actv(layer_type, name, inputs, filters, kernel_size, activation_fn,
                 strides, padding, regularizer, training, data_format,
                 dilation=1):
  """Helper function that applies convolution, layer norm and activation.
    Args:
      layer_type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[layer_type]

  conv = layer(
      name="{}".format(name),
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      dilation_rate=dilation,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  if data_format == 'channels_first':
    if layer_type == "conv1d":
      conv = tf.transpose(conv, [0, 2, 1])
    elif layer_type == "conv2d":
      conv = tf.transpose(conv, [0, 2, 3, 1])
  ln = tf.contrib.layers.layer_norm(
      inputs=conv,
  )
  if data_format == 'channels_first':
    if layer_type == "conv1d":
      ln = tf.transpose(ln, [0, 2, 1])
    elif layer_type == "conv2d":
      ln = tf.transpose(ln, [0, 3, 1, 2])

  output = ln
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def conv_in_actv(layer_type, name, inputs, filters, kernel_size, activation_fn,
                 strides, padding, regularizer, training, data_format,
                 dilation=1):
  """Helper function that applies convolution, instance norm and activation.
    Args:
      layer_type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[layer_type]

  conv = layer(
      name="{}".format(name),
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      dilation_rate=dilation,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  sn = tf.contrib.layers.instance_norm(
      inputs=conv,
      data_format="NHWC" if data_format == 'channels_last' else "NCHW"
  )

  output = sn
  if activation_fn is not None:
    output = activation_fn(output)
  return output
