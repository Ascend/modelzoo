# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import math

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore import Parameter


class DepthWiseConv(nn.Cell):
    def __init__(self, in_planes, kernel_size, stride, pad_mode, pad, channel_multiplier=1, has_bias=False):
        super(DepthWiseConv, self).__init__()
        self.has_bias = has_bias
        self.depthwise_conv = P.DepthwiseConv2dNative(channel_multiplier=channel_multiplier, kernel_size=kernel_size,
                                                      stride=stride, pad_mode=pad_mode, pad=pad)
        self.bias_add = P.BiasAdd()

        weight_shape = [channel_multiplier, in_planes, kernel_size[0], kernel_size[1]]
        self.weight = Parameter(initializer('ones', weight_shape), name='weight')

        if has_bias:
            bias_shape = [channel_multiplier * in_planes]
            self.bias = Parameter(initializer('zeros', bias_shape), name='bias')
        else:
            self.bias = None

    def construct(self, x):
        output = self.depthwise_conv(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


class ds_cnn(nn.Cell):
    def __init__(self, model_settings, model_size_info):
        super(ds_cnn, self).__init__()
        # N C H W
        label_count = model_settings['label_count']
        input_frequency_size = model_settings['dct_coefficient_count']
        input_time_size = model_settings['spectrogram_length']
        t_dim = input_time_size
        f_dim = input_frequency_size
        num_layers = model_size_info[0]
        conv_feat = [None] * num_layers
        conv_kt = [None] * num_layers
        conv_kf = [None] * num_layers
        conv_st = [None] * num_layers
        conv_sf = [None] * num_layers
        i = 1
        for layer_no in range(0, num_layers):
            conv_feat[layer_no] = model_size_info[i]
            i += 1
            conv_kt[layer_no] = model_size_info[i]
            i += 1
            conv_kf[layer_no] = model_size_info[i]
            i += 1
            conv_st[layer_no] = model_size_info[i]
            i += 1
            conv_sf[layer_no] = model_size_info[i]
            i += 1
        seq_cell = []
        in_channel = 1
        for layer_no in range(0, num_layers):
            if layer_no == 0:
                seq_cell.append(nn.Conv2d(in_channels=in_channel, out_channels=conv_feat[layer_no],
                                         kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                         stride=(conv_st[layer_no], conv_sf[layer_no]),
                                         pad_mode="same", padding=0, has_bias=False))
                seq_cell.append(nn.BatchNorm2d(num_features=conv_feat[layer_no], momentum=0.98))
                in_channel = conv_feat[layer_no]
            else:
                seq_cell.append(DepthWiseConv(in_planes=in_channel, kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                             stride=(conv_st[layer_no], conv_sf[layer_no]), pad_mode='same', pad=0))
                seq_cell.append(nn.BatchNorm2d(num_features=in_channel, momentum=0.98))
                seq_cell.append(nn.ReLU())
                seq_cell.append(nn.Conv2d(in_channels=in_channel, out_channels=conv_feat[layer_no], kernel_size=(1, 1),
                                         pad_mode="same"))
                seq_cell.append(nn.BatchNorm2d(num_features=conv_feat[layer_no], momentum=0.98))
                seq_cell.append(nn.ReLU())
                in_channel = conv_feat[layer_no]
            t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
            f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))
        seq_cell.append(nn.AvgPool2d(kernel_size=(t_dim, f_dim)))  # to fix ?
        seq_cell.append(nn.Flatten())
        seq_cell.append(nn.Dropout(model_settings['dropout1']))
        seq_cell.append(nn.Dense(in_channel, label_count))
        self.model = nn.SequentialCell(seq_cell)

    def construct(self, x):
        x = self.model(x)
        return x
