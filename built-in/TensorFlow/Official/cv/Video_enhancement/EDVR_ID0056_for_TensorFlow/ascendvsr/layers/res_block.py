import math
import tensorflow as tf
from ascendcv.layers import Conv2D, ActLayer


class ResBlockNoBN(object):

    def __init__(self, num_blocks, mid_channels, res_scale=1.0, act_cfg=dict(type='ReLU'), trainable=True, name='ResBlock'):
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels
        self.res_scale = res_scale
        self.name = name
        self.trainable = trainable
        self.act_cfg = act_cfg

    def build_block(self, x, idx):
        fan_in = int(x.shape[-1])
        out = Conv2D(x, self.mid_channels,
                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1/(100*fan_in))),
                     trainable=self.trainable, name='conv{}a'.format(idx))
        out = ActLayer(self.act_cfg)(out)
        out = Conv2D(out, self.mid_channels,
                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1/(100*fan_in))),
                     trainable=self.trainable, name='conv{}b'.format(idx))
        return x + out * self.res_scale

    def __call__(self, x):

        with tf.variable_scope(self.name) as scope:
            for i in range(self.num_blocks):
                x = self.build_block(x, i + 1)
            return x


class ResBlockNoBNwCA(ResBlockNoBN):

    def build_block(self, x, idx):
        fan_in = int(x.shape[-1])
        out1 = Conv2D(x, self.mid_channels,
                      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1/(100*fan_in))),
                      trainable=self.trainable, name='conv{}a'.format(idx))
        out1 = ActLayer(self.act_cfg)(out1)
        out1 = Conv2D(out1, self.mid_channels,
                      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1/(100*fan_in))),
                      trainable=self.trainable, name='conv{}b'.format(idx))

        out2 = tf.reduce_mean(out1, axis=[1, 2], keep_dims=True)
        out2 = Conv2D(out2, self.mid_channels, kernel_size=(1, 1), trainable=self.trainable, name='conv{}c'.format(idx))
        out2 = ActLayer(self.act_cfg)(out2)
        out2 = Conv2D(out2, self.mid_channels, kernel_size=(1, 1), trainable=self.trainable, name='conv{}d'.format(idx))
        out2 = out1 * out2

        out = tf.concat([out1, out2], axis=-1)
        out = Conv2D(out, self.mid_channels, kernel_size=(1, 1), trainable=self.trainable, name='conv{}e'.format(idx))

        return x + out * self.res_scale
