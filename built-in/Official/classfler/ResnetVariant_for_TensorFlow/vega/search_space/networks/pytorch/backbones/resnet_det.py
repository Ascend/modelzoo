# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet models for detection."""
import torch.nn as nn
import torch.nn.init as init
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from ..blocks.resnet_block_det import BasicBlock, Bottleneck, make_res_layer, conv_cfg_dict, norm_cfg_dict
from torch.nn.modules.batchnorm import _BatchNorm


@NetworkFactory.register(NetTypes.BACKBONE)
class ResNet_Det(Network):
    """ResNet for detection."""

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, desc):
        """Init ResNet."""
        super(ResNet_Det, self).__init__()
        self.net_desc = desc
        self.depth = desc["depth"]
        self.num_stages = desc["num_stages"] if "num_stages" in desc else 4
        self.strides = desc["strides"] if "strides" in desc else (1, 2, 2, 2)
        self.dilations = desc["dilations"] if "dilations" in desc else (1, 1, 1, 1)
        self.out_indices = desc["out_indices"] if "out_indices" in desc else (0, 1, 2, 3)
        self.style = desc["style"] if "style" in desc else "pytorch"
        self.frozen_stages = desc["frozen_stages"] if "frozen_stages" in desc else -1
        self.conv_cfg = desc["conv_cfg"] if "conv_cfg" in desc else {"type": "Conv"}
        self.norm_cfg = desc["norm_cfg"] if "norm_cfg" in desc else {"type": "BN", "requires_grad": True}
        self.norm_eval = desc["norm_eval"] if "norm_eval" in desc else True
        self.zero_init_residual = desc["zero_init_residual"] if "zero_init_residual" in desc else False
        self.block, stage_blocks = self.arch_settings[self.depth]
        self.stage_blocks = stage_blocks[:self.num_stages]
        self.inplanes = 64
        self._make_stem_layer()
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg, )
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

    def _make_stem_layer(self):
        """Make stem layer."""
        self.conv1 = conv_cfg_dict[self.conv_cfg['type']](3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                                          bias=False)
        self.norm1_name = norm_cfg_dict[self.norm_cfg['type']][0] + '_1'
        self.norm1 = norm_cfg_dict[self.norm_cfg['type']][1](64)
        requires_grad = self.norm_cfg['requires_grad'] if 'requires_grad' in self.norm_cfg else False
        if requires_grad:
            for param in self.norm1.parameters():
                param.requires_grad = requires_grad
        self.add_module(self.norm1_name, self.norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Init weight.

        :param pretrained: pretrain model path
        :type pretrained: str
        """
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    init.constant_(m.weight, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        init.constant_(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        init.constant_(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, **kwargs):
        """Forward compute of resnet for detection.

        :param x: input feature map
        :type x: torch.Tensor
        :return: out feature map
        :rtype: torch.Tensor
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Train setting.

        :param mode: if train
        :type mode: bool
        """
        super(ResNet_Det, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
