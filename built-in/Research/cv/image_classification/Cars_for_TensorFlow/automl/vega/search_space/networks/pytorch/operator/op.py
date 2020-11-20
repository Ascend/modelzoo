# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vega.search_space.networks.net_utils import NetTypes
from torch.autograd import Function
from vega.search_space.networks.network_factory import NetworkFactory


def import_all_operators():
    """Import all torch operators."""
    for _name in dir(nn):
        if not _name.startswith("_"):
            _cls = getattr(nn, _name)
            NetworkFactory.register_custom_cls(NetTypes.Operator, _cls)


import_all_operators()


@NetworkFactory.register(NetTypes.Operator)
class Input(nn.Module):
    """Create Input for forward x."""

    def __init__(self, key=None, data=None, cuda=False):
        self.key = key
        self.cuda = cuda
        self.data = data
        super(Input, self).__init__()

    def forward(self, x):
        """Forward x."""
        if self.data:
            return self.data
        if self.key is not None:
            x = x[self.key]
        if self.cuda:
            if torch.is_tensor(x):
                return x.cuda()
            else:
                if isinstance(x, dict):
                    x = {key: value.cuda() if torch.is_tensor(x) else value for key, value in x.items()}
                else:
                    x = [item.cuda() if torch.is_tensor(x) else item for item in x]
        return x


@NetworkFactory.register(NetTypes.Operator)
class Lambda(nn.Module):
    """Create Lambda for forward x."""

    def __init__(self, func, data=None):
        self.func = func
        self.data = data
        super(Lambda, self).__init__()

    def forward(self, x):
        """Forward x."""
        if self.data:
            return self.func(self.data)
        out = self.func(x)
        return out


@NetworkFactory.register(NetTypes.Operator)
class Reshape(nn.Module):
    """Create Lambda for forward x."""

    def __init__(self, *args):
        self.args = args
        super(Reshape, self).__init__()

    def forward(self, x):
        """Forward x."""
        return x.reshape(*self.args)


@NetworkFactory.register(NetTypes.Operator)
class Rermute(nn.Module):
    """Create Lambda for forward x."""

    def __init__(self, *args):
        self.args = args
        super(Rermute, self).__init__()

    def forward(self, x):
        """Forward x."""
        return x.rermute(*self.args)


@NetworkFactory.register(NetTypes.Operator)
def conv3x3(inchannel, outchannel, groups=1, stride=1):
    """Create conv3x3 layer.

    :param inchannel: input channel.
    :type inchannel: int
    :param outchannel: output channel.
    :type outchannel: int
    :param stride: the number to jump, default 1
    :type stride: int
    """
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


@NetworkFactory.register(NetTypes.Operator)
def conv1X1(inchannel, outchannel, stride=1):
    """Create conv1X1 layer.

    :param inchannel: input channel.
    :type inchannel: int
    :param outchannel: output channel.
    :type outchannel: int
    :param stride: the number to jump, default 1
    :type stride: int
    """
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False)


@NetworkFactory.register(NetTypes.Operator)
def conv5x5(inchannel, outchannel, stride=1, bias=False, dilation=1):
    """Create Convolution 5x5.

    :param in_planes: input channels
    :param out_planes: output channels
    :param stride: stride of the convolution
    :param bias: whether bias is contained
    :param dilation: dilation of the convolution
    :return: a convolution module
    """
    return nn.Conv2d(inchannel, outchannel, kernel_size=5, stride=stride,
                     padding=2, dilation=dilation, bias=bias)


@NetworkFactory.register(NetTypes.Operator)
def conv7x7(inchannel, outchannel, stride=1, bias=False, dilation=1):
    """Create Convolution 7x7.

    :param in_planes: input channels
    :param out_planes: output channels
    :param stride: stride of the convolution
    :param bias: whether bias is contained
    :param dilation: dilation of the convolution
    :return: a convolution module
    """
    return nn.Conv2d(inchannel, outchannel, kernel_size=7, stride=stride,
                     padding=3, dilation=dilation, bias=bias)


@NetworkFactory.register(NetTypes.Operator)
def conv_bn_relu6(inchannel, outchannel, kernel=3, stride=1):
    """Create conv1X1 layer.

    :param inchannel: input channel.
    :type inchannel: int
    :param outchannel: output channel.
    :type outchannel: int
    :param stride: the number to jump, default 1
    :type stride: int
    """
    return nn.Sequential(
        nn.Conv2d(inchannel, outchannel, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU6(inplace=True)
    )


@NetworkFactory.register(NetTypes.Operator)
def conv_bn_relu(inchannel, outchannel, kernel_size, stride, padding, affine=True):
    """Create group of Convolution + BN + Relu.

    :param C_in: input channel
    :param C_out: output channel
    :param kernel_size: kernel size of convolution layer
    :param stride: stride of convolution layer
    :param padding: padding of convolution layer
    :param affine: whether use affine in batchnorm
    :return: group of Convolution + BN + Relu
    """
    return nn.Sequential(
        nn.Conv2d(inchannel, outchannel, kernel_size, stride=stride, padding=padding,
                  bias=False),
        nn.BatchNorm2d(outchannel, affine=affine),
        nn.ReLU(inplace=False),
    )


@NetworkFactory.register(NetTypes.Operator)
class View(nn.Module):
    """Call torch.view."""

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        """Forward x."""
        return x.view(x.size(0), -1)


@NetworkFactory.register(NetTypes.Operator)
class ConvWS2d(nn.Conv2d):
    """Conv2d with weight standarlization.

    :param in_channels: input channels
    :param out_channels: output channels
    :param kernel_size: kernel size
    :param stride: stride
    :param padding: num of padding
    :param dilation: num of dilation
    :param groups: num of groups
    :param bias: bias
    :param eps: eps
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, eps=1e-5):
        """Init conv2d with weight standarlization."""
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def conv_ws_2d(self, input, weight, bias=None, stride=1, padding=0,
                   dilation=1, groups=1, eps=1e-5):
        """Conv2d with weight standarlization.

        :param input: input feature map
        :type input: torch.Tensor
        :param weight: weight of conv layer
        :type weight: torch.Tensor
        :param bias: bias
        :type bias: torch.Tensor
        :param stride: conv stride
        :type stride: int
        :param padding: num of padding
        :type padding: int
        :param dilation: num of dilation
        :type dilation: int
        :param groups: num of group
        :type groups: int
        :param eps: weight eps
        :type eps: float
        """
        c_in = weight.size(0)
        weight_flat = weight.view(c_in, -1)
        mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
        std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
        weight = (weight - mean) / (std + eps)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    def forward(self, x):
        """Forward function of conv2d with weight standarlization."""
        return self.conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                               self.dilation, self.groups, self.eps)


@NetworkFactory.register(NetTypes.Operator)
class ChannelShuffle(nn.Module):
    """Shuffle the channel of features.

    :param groups: group number of channels
    :type groups: int
    """

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Forward x."""
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


@NetworkFactory.register(NetTypes.Operator)
class Shrink_Conv(nn.Module):
    """Call torch.cat.

    :param InChannel: channel number of input
    :type InChannel: int
    :param OutChannel: channel number of output
    :type OutChannel: int
    :param growRate: growth rate of block
    :type growRate: int
    :param nConvLayers: the number of convlution layer
    :type nConvLayers: int
    :param kSize: kernel size of convolution operation
    :type kSize: int
    """

    def __init__(self, InChannel, OutChannel, growRate, nConvLayers, kSize):
        super(Shrink_Conv, self).__init__()
        self.InChan = InChannel
        self.OutChan = OutChannel
        self.G = growRate
        self.C = nConvLayers
        if self.InChan != self.G:
            self.InConv = nn.Conv2d(self.InChan, self.G, 1, padding=0, stride=1)
        if self.OutChan != self.G and self.OutChan != self.InChan:
            self.OutConv = nn.Conv2d(self.InChan, self.OutChan, 1, padding=0,
                                     stride=1)
        self.Convs = nn.ModuleList()
        self.ShrinkConv = nn.ModuleList()
        for i in range(self.C):
            self.Convs.append(nn.Sequential(*[
                nn.Conv2d(self.G, self.G, kSize, padding=(kSize - 1) // 2,
                          stride=1), nn.ReLU()]))
            if i == (self.C - 1):
                self.ShrinkConv.append(
                    nn.Conv2d((2 + i) * self.G, self.OutChan, 1, padding=0,
                              stride=1))
            else:
                self.ShrinkConv.append(
                    nn.Conv2d((2 + i) * self.G, self.G, 1, padding=0, stride=1))

    def forward(self, x):
        """Forward x."""
        if self.InChan != self.G:
            x_InC = self.InConv(x)
            x_inter = self.Convs[0](x_InC)
            x_conc = torch.cat((x_InC, x_inter), 1)
            x_in = self.ShrinkConv[0](x_conc)
        else:
            x_inter = self.Convs[0](x)
            x_conc = torch.cat((x, x_inter), 1)
            x_in = self.ShrinkConv[0](x_conc)
        for i in range(1, self.C):
            x_inter = self.Convs[i](x_in)
            x_conc = torch.cat((x_conc, x_inter), 1)
            x_in = self.ShrinkConv[i](x_conc)
        return x_in


@NetworkFactory.register(NetTypes.Operator)
class Cont_Conv(nn.Module):
    """Call torch.cat.

    :param InChannel: channel number of input
    :type InChannel: int
    :param OutChannel: channel number of output
    :type OutChannel: int
    :param growRate: growth rate of block
    :type growRate: int
    :param nConvLayers: the number of convlution layer
    :type nConvLayers: int
    :param kSize: kernel size of convolution operation
    :type kSize: int
    """

    def __init__(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        super(Cont_Conv, self).__init__()
        self.InChan = InChannel
        self.OutChan = OutChannel
        self.G = growRate
        self.C = nConvLayers
        self.shup = nn.PixelShuffle(2)
        self.Convs = nn.ModuleList()
        self.ShrinkConv = nn.ModuleList()
        for i in range(self.C):
            self.Convs.append(nn.Sequential(*[
                nn.Conv2d(self.G, self.G, kSize, padding=(kSize - 1) // 2,
                          stride=1), nn.ReLU()]))
            if i < (self.C - 1):
                self.ShrinkConv.append(
                    nn.Conv2d((2 + i) * self.G, self.G, 1, padding=0, stride=1))
            else:
                self.ShrinkConv.append(
                    nn.Conv2d(int((2 + i) * self.G / 4), self.OutChan, 1,
                              padding=0, stride=1))

    def forward(self, x):
        """Forward x."""
        x_conc = x
        for i in range(0, self.C):
            x_inter = self.Convs[i](x)
            x_inter = self.Convs[i](x_inter)
            x_inter = self.Convs[i](x_inter)
            x_conc = torch.cat((x_conc, x_inter), 1)
            if i == (self.C - 1):
                x_conc = self.shup(x_conc)
                x_in = self.ShrinkConv[i](x_conc)
            else:
                x_in = self.ShrinkConv[i](x_conc)
        return x_in


@NetworkFactory.register(NetTypes.Operator)
class Esrn_Cat(nn.Module):
    """Call torch.cat."""

    def __init__(self):
        super(Esrn_Cat, self).__init__()

    def forward(self, x):
        """Forward x."""
        return torch.cat(list(x), 1)


@NetworkFactory.register(NetTypes.Operator)
class MicroDecoder_Upsample(nn.Module):
    """Call torch.Upsample."""

    def __init__(self, collect_inds, agg_concat):
        self.collect_inds = collect_inds
        self.agg_concat = agg_concat
        super(MicroDecoder_Upsample, self).__init__()

    def forward(self, x):
        """Forward x."""
        out = x[self.collect_inds[0]]
        for i in range(1, len(self.collect_inds)):
            collect = x[self.collect_inds[i]]
            if out.size()[2] > collect.size()[2]:
                # upsample collect
                collect = nn.Upsample(size=out.size()[2:], mode='bilinear', align_corners=True)(collect)
            elif collect.size()[2] > out.size()[2]:
                out = nn.Upsample(size=collect.size()[2:], mode='bilinear', align_corners=True)(out)
            if self.agg_concat:
                out = torch.cat([out, collect], 1)
            else:
                out += collect
        out = F.relu(out)
        return out


@NetworkFactory.register(NetTypes.Operator)
class ContextualCell_v1(nn.Module):
    """New contextual cell design."""

    def __init__(self, op_names, config, inp, repeats=1, concat=False):
        """Construct ContextualCell_v1 class.

        :param op_names: list of operation indices
        :param config: list of config numbers
        :param inp: input channel
        :param repeats: number of repeated times
        :param concat: concat the result if set to True, otherwise add the result
        """
        super(ContextualCell_v1, self).__init__()
        self._ops = nn.ModuleList()
        self._pos = []
        self._collect_inds = [0]
        self._pools = ['x']
        for ind, op in enumerate(config):
            # first op is always applied on x
            if ind == 0:
                pos = 0
                op_id = op
                self._collect_inds.remove(pos)
                op_name = op_names[op_id]
                self._ops.append(OPS[op_name](inp, 1, True, repeats))  # turn-off scaling in batch norm
                self._pos.append(pos)
                self._collect_inds.append(ind + 1)
                self._pools.append('{}({})'.format(op_name, self._pools[pos]))
            else:
                pos1, pos2, op_id1, op_id2 = op
                # drop op_id from loose ends
                for ind2, (pos, op_id) in enumerate(zip([pos1, pos2], [op_id1, op_id2])):
                    if pos in self._collect_inds:
                        self._collect_inds.remove(pos)
                    op_name = op_names[op_id]
                    self._ops.append(OPS[op_name](inp, 1, True, repeats))  # turn-off scaling in batch norm
                    self._pos.append(pos)
                    # self._collect_inds.append(ind * 3 + ind2 - 1) # Do not collect intermediate
                    self._pools.append('{}({})'.format(op_name, self._pools[pos]))
                # summation
                op_name = 'sum'
                self._ops.append(AggregateCell(size_1=None, size_2=None, agg_size=inp, pre_transform=False,
                                               concat=concat))  # turn-off convbnrelu
                self._pos.append([ind * 3 - 1, ind * 3])
                self._collect_inds.append(ind * 3 + 1)
                self._pools.append('{}({},{})'.format(op_name, self._pools[ind * 3 - 1], self._pools[ind * 3]))

    def forward(self, x):
        """Do an inference on ContextualCell_v1.

        :param x: input tensor
        :return: output tensor
        """
        feats = [x]
        for pos, op in zip(self._pos, self._ops):
            if isinstance(pos, list):
                assert len(pos) == 2, "Two ops must be provided"
                feats.append(op(feats[pos[0]], feats[pos[1]]))
            else:
                feats.append(op(feats[pos]))
        out = 0
        for i in self._collect_inds:
            out += feats[i]
        return out


@NetworkFactory.register(NetTypes.Operator)
class AggregateCell(nn.Module):
    """Aggregate two cells and sum or concat them up."""

    def __init__(self, size_1, size_2, agg_size, pre_transform=True, concat=False):
        """Construct AggregateCell.

        :param size_1: channel of first input
        :param size_2: channel of second input
        :param agg_size: channel of aggregated tensor
        :param pre_transform: whether to do a transform on two inputs
        :param concat: concat the result if set to True, otherwise add the result
        """
        super(AggregateCell, self).__init__()
        self.pre_transform = pre_transform
        self.concat = concat
        if self.pre_transform:
            self.branch_1 = conv_bn_relu(size_1, agg_size, 1, 1, 0)
            self.branch_2 = conv_bn_relu(size_2, agg_size, 1, 1, 0)
        if self.concat:
            self.conv1x1 = conv_bn_relu(agg_size * 2, agg_size, 1, 1, 0)

    def forward(self, x1, x2):
        """Do an inference on AggregateCell.

        :param x1: first input
        :param x2: second input
        :return: output
        """
        if self.pre_transform:
            x1 = self.branch_1(x1)
            x2 = self.branch_2(x2)
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear', align_corners=True)(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear', align_corners=True)(x1)
        if self.concat:
            return self.conv1x1(torch.cat([x1, x2], 1))
        else:
            return x1 + x2


@NetworkFactory.register(NetTypes.Operator)
class GAPConv1x1(nn.Module):
    """Global Average Pooling + conv1x1."""

    def __init__(self, C_in, C_out):
        """Construct GAPConv1x1 class.

        :param C_in: input channel
        :param C_out: output channel
        """
        super(GAPConv1x1, self).__init__()
        if '0.2' in torch.__version__:
            # !!!!!!!!!!used for input size 448 with overall stride 32!!!!!!!!!!
            self.globalpool = nn.AvgPool2d(14)
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def forward(self, x):
        """Do an inference on GAPConv1x1.

        :param x: input tensor
        :return: output tensor
        """
        size = x.size()[2:]
        if '0.2' in torch.__version__:
            out = self.globalpool(x)
        else:
            out = x.mean(2, keepdim=True).mean(3, keepdim=True)
        out = self.conv1x1(out)
        if '0.2' in torch.__version__:
            out = nn.Upsample(size=size, mode='bilinear')(out)
        else:
            out = nn.functional.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out


@NetworkFactory.register(NetTypes.Operator)
class DilConv(nn.Module):
    """Separable dilated convolution block."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation, affine=True):
        """Construct DilConv class.

        :param C_in: input channel
        :param C_out: output channel
        :param kernel_size: kernel size of the first convolution layer
        :param stride: stride of the first convolution layer
        :param padding: padding of the first convolution layer
        :param dilation: dilation of the first convolution layer
        :param affine: whether use affine in BN
        """
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        """Do an inference on DilConv.

        :param x: input tensor
        :return: output tensor
        """
        return self.op(x)


@NetworkFactory.register(NetTypes.Operator)
class SepConv(nn.Module):
    """Separable convolution block with repeats."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, repeats=1):
        """Construct SepConv class.

        :param C_in: number of input channel
        :param C_out: number of output channel
        :param kernel_size: kernel size of the first conv
        :param stride: stride of the first conv
        :param padding: padding of the first conv
        :param dilation: dilation of the first conv
        :param affine: whether to use affine in BN
        :param repeats: number of repeat times
        """
        super(SepConv, self).__init__()

        def basic_op():
            return nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=False))

        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        """Do an inference on SepConv.

        :param x: input tensor
        :return: output tensor
        """
        return self.op(x)


@NetworkFactory.register(NetTypes.Operator)
class Zero(nn.Module):
    """Zero block."""

    def __init__(self, stride):
        """Construct Zero class.

        :param stride: stride of the output
        """
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        """Do an inference on Zero.

        :param x: input tensor
        :return: output tensor
        """
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


@NetworkFactory.register(NetTypes.Operator)
class FactorizedReduce(nn.Module):
    """Factorized reduce block."""

    def __init__(self, C_in, C_out, affine=True):
        """Construct FactorizedReduce class.

        :param C_in: input channel
        :param C_out: output channel
        :param affine: whether to use affine in BN
        """
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        """Do an inference on FactorizedReduce.

        :param x: input tensor
        :return: output tensor
        """
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


@NetworkFactory.register(NetTypes.Operator)
class Pad(nn.Module):
    """Create Input for forward x."""

    def __init__(self, planes):
        super(Pad, self).__init__()
        self.planes = planes

    def forward(self, x):
        """Forward x."""
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)


OPS = {
    'none': lambda C, stride, affine, repeats=1: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, repeats=1: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, repeats=1: nn.MaxPool2d(
        3, stride=stride, padding=1),
    'global_average_pool': lambda C, stride, affine, repeats=1: GAPConv1x1(C, C),
    'skip_connect': lambda C, stride, affine, repeats=1: Input() if stride == 1 else FactorizedReduce(
        C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, repeats=1: SepConv(C, C, 3, stride, 1, affine=affine, repeats=repeats),
    'sep_conv_5x5': lambda C, stride, affine, repeats=1: SepConv(C, C, 5, stride, 2, affine=affine, repeats=repeats),
    'sep_conv_7x7': lambda C, stride, affine, repeats=1: SepConv(C, C, 7, stride, 3, affine=affine, repeats=repeats),
    'dil_conv_3x3': lambda C, stride, affine, repeats=1: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, repeats=1: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine, repeats=1: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
    'conv1x1': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv1X1(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv5x5': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv5x5(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv7x7': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv7x7(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil2': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=2),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil3': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=3),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil12': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=12),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'sep_conv_3x3_dil3': lambda C, stride, affine, repeats=1: SepConv(
        C, C, 3, stride, 3, affine=affine, dilation=3, repeats=repeats),
    'sep_conv_5x5_dil6': lambda C, stride, affine, repeats=1: SepConv(
        C, C, 5, stride, 12, affine=affine, dilation=6, repeats=repeats)
}
