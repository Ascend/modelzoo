# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is weight init module."""
import torch.nn as nn


def constant_init(module, val, bias=0):
    """Init weight by Constant.

    :param module: target module
    :type: nn.module
    :param val: weight value
    :type: float
    :param bias: bias of method
    :type:float
    """
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Init weight by Xavier method.

    :param module: target module
    :type: nn.module
    :param gain: gain
    :type: float
    :param bias: bias of method
    :type:float
    :distribute: weight distribute
    :type: str
    """
    # assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """Init weight by Normal method.

    :param module: target module
    :type: nn.module
    :param mean: mean
    :type: float
    :param std: std
    :type: float
    :param bias: bias
    :type: float
    """
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    """Init weight by Uniform method.

    :param module: target module
    :type: nn.module
    :param a: a
    :type: float
    :param b: b
    :type: float
    :param bias: bias
    :type: float
    """
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    """Init weight by Kaiming method.

    :param module: target module
    :type: nn.module
    :param a: a
    :type: float
    :param mode: mode
    :type: float
    :param nonlinearity: nonlinearity
    :type: str
    :param bias: bias
    :distribute: weight distribute
    :type: str
    """
    # assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    """`XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch."""
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        distribution='uniform')
