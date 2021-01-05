import tensorflow as tf


class Layers(object):
    def __init__(self):
        self.initializer_xavier = tf.initializers.glorot_normal()

    def elu(self, inputs):
        return tf.nn.elu(inputs)

    def relu(self, inputs, name=None):
        return tf.nn.relu(inputs, name=name)

    def sigmoid(self, inputs, name=None):
        return tf.nn.sigmoid(inputs, name=name)

    def lin(self, inputs):
        w = tf.Variable(tf.ones_like(inputs), trainable=True)
        b = tf.Variable(tf.zeros_like(inputs), trainable=True)
        return inputs * w + b

    def softmax(self, inputs):
        return tf.nn.softmax(inputs, axis=-1)

    def max_pool(self, inputs, pool_size=2, stride_size=2):
        return tf.layers.max_pooling2d(inputs, pool_size=pool_size, strides=stride_size)

    def avg_pool(self, inputs, pool_size=2, stride_size=2):
        return tf.layers.average_pooling2d(inputs, pool_size=pool_size, strides=stride_size)

    def global_avgpool(self, inputs):
        return tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    def upsample(self, x, ratio=(2, 2), name=None):
        return tf.image.resize(x, (x.shape[1] * ratio[0], x.shape[2] * ratio[1]), name=name)

    def bmin(self, inputs, num_units, axis=None):
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:
            axis = -1
        num_channels = shape[axis]
        shape[axis] = num_units
        shape += [num_channels // num_units]
        outputs = tf.reduce_min(tf.reshape(inputs, shape), -2, keep_dims=False)
        return outputs

    def conv2d(self, inputs, ch_out, kernel_size=3, stride=1, padding='same', rate=1, use_bias=True):
        reg = tf.contrib.layers.l2_regularizer(1e-4)
        return tf.layers.conv2d(inputs, filters=ch_out, kernel_size=(kernel_size, kernel_size),
                                strides=(stride, stride), padding=padding, dilation_rate=(rate, rate),
                                use_bias=use_bias, kernel_regularizer=reg)

    def transpose_conv2d(self, inputs, ch_out, kernel_size=3, stride=1, padding='valid', rate=1, use_bias=True,
                         name=None):
        reg = tf.contrib.layers.l2_regularizer(1e-4)
        return tf.layers.conv2d_transpose(inputs, filters=ch_out, kernel_size=(kernel_size, kernel_size),
                                          strides=(stride, stride), padding=padding, dilation_rate=(rate, rate),
                                          use_bias=use_bias, kernel_regularizer=reg, name=name)

    def dense(self, inputs, ch_out, use_bias=True):
        return tf.layers.conv2d(inputs, filters=ch_out, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                use_bias=use_bias, kernel_regularizer=None)

    def batch_norm(self, inputs, is_training=True):
        return tf.layers.batch_normalization(inputs, training=is_training)

    def filter_response_norm(self, x, eps=1e-6):
        gamma = tf.Variable(tf.ones([1, 1, 1, x.shape[3]]), trainable=True, dtype=tf.float32)
        beta = tf.Variable(tf.zeros([1, 1, 1, x.shape[3]]), trainable=True, dtype=tf.float32)
        nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keep_dims=True)
        x = x * tf.rsqrt(nu2 + tf.abs(eps))
        return gamma * x + beta
