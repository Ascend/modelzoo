# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import tensorflow as tf
import numpy as np

from .conv import Conv2D
from ..utils.misc import pair

try:
    from npu_bridge.tbe.npu_cube_ops import deformable_conv2d
except Exception:
    print('Failed to import NPU deformable_conv2d. Please use the composed tf operator instead.')

class DeformableConvLayer(object):
    """
    Fast version
    only support kernel_size=3*3, stride=1, padding=1, num_groups=1, num_deformable_groups=1
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilations=1,
                 use_bias=True,
                 num_groups=1,
                 num_deform_groups=1,
                 trainable=True,
                 impl='tf'):
        super(DeformableConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.strides = pair(strides)
        self.padding = padding.lower()
        self.dilations = pair(dilations)
        self.use_bias = use_bias
        self.num_groups = num_groups
        self.num_deform_groups = num_deform_groups
        self.trainable = trainable
        self.kernel_intermediate_shape = []
        self.build()
        self.debug = False
        self.use_zero = False
        self.impl = impl
        
    def build(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        initializer = tf.random_uniform_initializer(-stdv, stdv)

        self.kernel_intermediate_shape = [*self.kernel_size, self.in_channels//self.num_groups, self.out_channels//self.num_groups, self.num_groups]

        self.kernel = tf.get_variable(
            "W",
            [*self.kernel_size, self.in_channels//self.num_groups, self.out_channels],
            initializer=initializer,
            trainable=self.trainable)
        if self.use_bias:
            self.bias = tf.get_variable(
                "b",
                (self.out_channels,),
                initializer=tf.constant_initializer(value=0.0),
                trainable=self.trainable)
    
    def _cal_pads(self, ih, iw):
        if self.padding == 'same':
            strh, strw = self.strides
            kh, kw = self.kernel_size
            dilh, dilw = self.dilations
            tails_h = ih % strh
            tails_w = iw % strw
            dkh = dilh * (kh - 1) + 1
            dkw = dilw * (kw - 1) + 1
            pad_h = dkh - tails_h if tails_h > 0 else dkh - strh
            pad_w = dkw - tails_w if tails_w > 0 else dkw - strw
            pads = [pad_h // 2, pad_h // 2 + pad_h % 2, pad_w // 2, pad_w // 2 + pad_w % 2]
        else:
            pads = [0, 0, 0, 0]
        return pads   
 
    def __call__(self, inputs, offset):
        if self.impl == 'tf':
            return self._call_tf(inputs, offset)
        elif self.impl == 'npu':
            return self._call_npu(inputs, offset)

    def _call_npu(self, inputs, offset):
        _, ih, iw, _ = inputs.get_shape().as_list()
        c = offset.get_shape().as_list()[3]
        assert c == self.num_deform_groups*self.kernel_size[0]*self.kernel_size[1]*3
        offset_all = offset

        pads = self._cal_pads(ih, iw)
        out = deformable_conv2d(
                inputs,
                self.kernel,
                offset_all,
                strides=[1] + list(self.strides) + [1],
                pads=pads,
                data_format='NHWC',
                dilations=[1] + list(self.dilations) + [1],
                groups=self.num_groups,
                deformable_groups=self.num_deform_groups)

        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)
        return out

    def _call_tf(self, inputs, offset):
        def _get_in_bound_mask(x_, y_):
            out_of_bound_x = tf.logical_or(tf.greater(x_, in_w-1), tf.less(x_, 0))
            out_of_bound_y = tf.logical_or(tf.greater(y_, in_h-1), tf.less(y_, 0))
            out_of_bound_mask = tf.logical_or(out_of_bound_x, out_of_bound_y)
            return 1. - tf.to_float(out_of_bound_mask)

        inputs = self._pad_input(inputs)
        # bs, in_h, in_w, _ = list(map(int, inputs.shape))
        bs, in_h, in_w, _ = inputs.get_shape().as_list()
        # bs, out_h, out_w, _ = list(map(int, offset.shape))
        bs, out_h, out_w, c = offset.get_shape().as_list()

        assert c == self.num_deform_groups*self.kernel_size[0]*self.kernel_size[1]*3
        c3 = c // 3

        # get x, y axis offset. Swap the order to 'x,y' instead of 'y,x', align with npu dcn op
        x_off = offset[:, :, :, :c3]
        y_off = offset[:, :, :, c3:c3*2]
        mask = offset[:, :, :, c3*2:]

        # for dense_image_warp, should be in 'y,x' order, as is the common sense by researchers
        # y_off = offset[:, :, :, :offset.shape[-1] // 2]
        # x_off = offset[:, :, :, offset.shape[-1] // 2:]
        
        # input feature map gird coordinates
        y, x = self._get_conv_indices(in_h, in_w)
        # y, x = tf.cast(y, tf.float32), tf.cast(x, tf.float32)
        y, x = [tf.to_float(i) for i in [y, x]]
        y, x = [tf.tile(i, [1, 1, 1, self.num_deform_groups]) for i in [y, x]]
        
        # current deformable offsets
        y, x = y + y_off, x + x_off
        # dense_image_warp op
        # y, x = y - y_off, x - x_off

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.to_int32(tf.floor(i)) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        
        # according to the strategy, prepare in_bound mask if use zero.
        # In fact, gathernd on GPU and NPU will take 0 if the index is out-of-bound,
        # while CPU will throw an error. Therefore, do an explicit masking
        if self.use_zero:
            m0 = _get_in_bound_mask(x0, y0)
            m1 = _get_in_bound_mask(x1, y0)
            m2 = _get_in_bound_mask(x0, y1)
            m3 = _get_in_bound_mask(x1, y1)
        
        # clip the indices
        y0, y, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y, y1]]
        x0, x, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x, x1]]
        
        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [self._get_pixel_values_at_point(inputs, i) for i in indices]
        
        # cast to float
        x0, x, x1, y0, y, y1 = [tf.to_float(i) for i in [x0, x, x1, y0, y, y1]]
        
        # weights
        # Re-formulate the weights calculation, ensuring w0+w1+w2+w3=1.
        y_res = y - y0
        x_res = x - x0
        w0 = (1. - y_res) * (1. - x_res)
        w1 = (1. - y_res) * x_res
        w2 = y_res * (1. - x_res)
        w3 = y_res * x_res
        
        if self.use_zero:
            w0 = m0 * w0
            w1 = m1 * w1
            w2 = m2 * w2
            w3 = m3 * w3

        w0, w1, w2, w3 = [tf.reshape(i, [*i.get_shape()[:3], self.num_deform_groups, *self.kernel_size, 1])
                          for i in [w0, w1, w2, w3]]
        # reshape of px is done in gather process. The next two lines will be removed in the next comment
        # p0, p1, p2, p3 = [tf.reshape(i, [*i.get_shape()[:3], self.num_deform_groups, *self.kernel_size,-1])
        #                  for i in [p0, p1, p2, p3]]

        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        if mask is not None:
            pixels = tf.reshape(mask, [*mask.get_shape()[:3], self.num_deform_groups, *self.kernel_size, 1]) * pixels

        # reshape the "big" feature map
        # pixels = tf.reshape(pixels, [bs, out_h, out_w, *self.kernel_size, -1])
        pixels = tf.transpose(pixels, [0,1,4,2,5,3,6])
        pixels = tf.reshape(pixels, [bs, out_h*self.kernel_size[0], out_w*self.kernel_size[1], -1])

        # conv
        # TODO abstract a group_conv class?
        kernel_reshaped = tf.reshape(self.kernel, self.kernel_intermediate_shape)
        ich = pixels.shape[-1] // self.num_groups
        out = tf.concat([tf.nn.conv2d(
                pixels[:, :, :, i*ich:(i+1)*ich], 
                kernel_reshaped[:, :, :, :, i], 
                strides=self.kernel_size, 
                padding='VALID',
                )
                for i in range(self.num_groups)], axis=-1)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)

        if self.debug:
            return out, w0, w1, w2, w3, p0, p1, p2, p3, pixels, offset, x, y, x0, x1, y0, y1
        else:
            return out

    def _pad_input(self, x):

        if self.padding == 'same':
            size = x.get_shape().as_list()[1:3]
            # size = list(map(int, x.shape))[1:3]
            pad = []
            for i in range(2):
                dilated_filter_size = 1 + self.dilations[i] * (self.kernel_size[i] - 1)
                same_output = (size[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (size[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output > valid_output:
                    p0 = (dilated_filter_size - 1) // 2
                    pad.append([p0, dilated_filter_size - 1 - p0])
                else:
                    pad.append([0, 0])

            if sum([sum(p) for p in pad]) != 0:
                x = tf.pad(x, [[0, 0]] + pad + [[0, 0]])

        return x

    def _get_conv_indices(self, feat_h, feat_w):
        """the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_image_patches(i,
                                               [1, *self.kernel_size, 1],
                                               [1, *self.strides, 1],
                                               [1, *self.dilations, 1],
                                               'VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    def _get_pixel_values_at_point(self, inputs, indices):
        """get pixel values
        :param inputs: [batch_size, Hin, Win, Cin]
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        bs, h, w, c = y.get_shape().as_list()[0: 4]

        if c % self.num_deform_groups != 0 or inputs.shape[-1] % self.num_deform_groups != 0:
            raise ValueError

        per_group_offset_ch = c // self.num_deform_groups  # kh*kw
        per_group_input_ch = inputs.shape[-1] // self.num_deform_groups
        batch_idx = tf.reshape(tf.range(0, bs), (bs, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, per_group_offset_ch))
        
        outs = []
        for j in range(self.num_deform_groups):
            pixel_idx = tf.stack([b, y[:, :, :, j*per_group_offset_ch:(j+1)*per_group_offset_ch],
                                  x[:, :, :, j*per_group_offset_ch:(j+1)*per_group_offset_ch]], axis=-1)  # [bs, h, w, per_group_offset_ch, 3]
            outs.append(tf.gather_nd(inputs[:, :, :, j*per_group_input_ch:(j+1)*per_group_input_ch], pixel_idx))  
        outs = tf.concat(outs, axis=-1)  # [bs, h, w, per_group_offset_ch, cin]

        # reshape and transpose the outputs in order to align with the outer axis order
        outs = tf.reshape(outs, [*outs.shape[:3], *self.kernel_size, self.num_deform_groups, -1])
        return tf.transpose(outs, [0,1,2,5,3,4,6])

        # stacking all the indices into one and gathering once for all will double the step time on GPU V100
        # inputs_reshaped = tf.reshape(inputs, [*inputs.shape[:3], self.num_deform_groups, per_group_input_ch])

        # ind = []
        # for j in range(self.num_deform_groups):
        #     g = tf.ones((bs, h, w, per_group_offset_ch), dtype=tf.int32) * j
        #     pixel_idx = tf.stack([b, y[:, :, :, j*per_group_offset_ch:(j+1)*per_group_offset_ch],
        #                           x[:, :, :, j*per_group_offset_ch:(j+1)*per_group_offset_ch], g], axis=-1)  # [bs, h, w, per_group_offset_ch, 4]
        #     ind.append(pixel_idx)
        # gather_ind = tf.stack(ind, axis=4)  # [bs, h, w, per_group_offset_ch, num_deform_groups, 4]
        # out = tf.gather_nd(inputs_reshaped, gather_ind)
        # return tf.reshape(out, (bs, h, w, per_group_offset_ch, inputs.shape[-1]))  # [bs, h, w, per_group_offset_ch, cin]


def DCNPack(x, extra_feat, out_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', dilations=(1, 1),
            use_bias=True, num_groups=1, num_deform_groups=1, trainable=True, dcn_version='v2', name='DCN', impl='npu'):

    with tf.variable_scope(name):
        x = tf.cast(x, tf.float32)
        if dcn_version == 'v1':
            offset = Conv2D(extra_feat, num_deform_groups * 2 * kernel_size[0] * kernel_size[1],
                            kernel_size=kernel_size, strides=strides, padding=padding, dilations=dilations,
                            use_bias=use_bias, trainable=trainable, name='conv_offset')
            offset = tf.cast(offset, tf.float32)
            mask = None
            raise Not
        elif dcn_version == 'v2':
            conv_offset = Conv2D(extra_feat, num_deform_groups * 3 * kernel_size[0] * kernel_size[1],
                                 kernel_size=kernel_size, strides=strides, padding=padding, dilations=dilations,
                                 use_bias=use_bias, trainable=trainable, name='conv_offset')
            conv_offset = tf.cast(conv_offset, tf.float32)
            # offset = conv_offset[:, :, :, :num_deform_groups * 2 * kernel_size[0] * kernel_size[1]]
            # mask = conv_offset[:, :, :, num_deform_groups * 2 * kernel_size[0] * kernel_size[1]:]
            # mask = tf.nn.sigmoid(mask)

            sigmoid_offset = tf.nn.sigmoid(conv_offset)
            weight = np.ones((1, 1, 1, num_deform_groups*kernel_size[0]*kernel_size[1]*3)).astype(np.float32)
            weight[..., num_deform_groups*kernel_size[0]*kernel_size[1]*2:] = 0.
            weight = tf.convert_to_tensor(weight)

            input_offset_mask = weight * conv_offset + (1. - weight) * sigmoid_offset
        else:
            raise NotImplementedError

        # out = DeformableConvLayer(
        #     in_channels=int(x.shape[-1]), out_channels=out_channels,
        #     kernel_size=kernel_size, strides=strides, padding=padding, dilations=dilations,
        #     use_bias=use_bias, num_groups=num_groups, num_deform_groups=num_deform_groups,
        #     trainable=trainable, impl=impl)(x, offset, mask)

        out = DeformableConvLayer(
            in_channels=int(x.shape[-1]), out_channels=out_channels,
            kernel_size=kernel_size, strides=strides, padding=padding, dilations=dilations,
            use_bias=use_bias, num_groups=num_groups, num_deform_groups=num_deform_groups,
            trainable=trainable, impl=impl)(x, input_offset_mask)

        return out
