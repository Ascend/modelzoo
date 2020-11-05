"""Modules for ResNeXt_Variant."""

import math
import torch.nn as nn
from mmdet.models.backbones.resnet import Bottleneck as _Bottleneck
from mmdet.models.backbones.resnet import BasicBlock as _BasicBlock
from mmdet.models.registry import BACKBONES
from mmdet.models.utils import build_conv_layer, build_norm_layer
from mmdet.ops import DeformConv, ModulatedDeformConv
from .resnet_variant import ResNet_Variant


class BasicBlock(_BasicBlock):
    """Class of BasicBlock for ResNeXt."""

    def __init__(self, inplanes, planes, groups=1,
                 base_width=4, base_channel=64, **kwargs):
        super(BasicBlock, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(
                self.planes * (base_width / base_channel)) * groups
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=2)
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            3,
            stride=self.stride,
            padding=self.dilation,
            dilation=self.dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(self.conv_cfg, width, self.planes * self.expansion,
                                      3, padding=self.dilation, dilation=self.dilation, groups=groups, bias=False)
        self.add_module(self.norm2_name, norm2)


class Bottleneck(_Bottleneck):
    """Class of Bottleneck for ResNeXt."""

    def __init__(self, inplanes, planes, groups=2,
                 base_width=32, base_channel=64, **kwargs):
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)
        if groups == 1:
            width = self.planes
        else:
            width = math.floor(
                self.planes * (base_width / base_channel)) * groups
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = self.dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                self.conv_cfg,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            groups = self.dcn.get('groups', 1)
            deformable_groups = self.dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                width,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation)
            self.conv2 = conv_op(
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                deformable_groups=deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)


def make_res_layer(block,
                   inplanes,
                   planes,
                   arch,
                   groups=1,
                   base_width=4,
                   base_channel=64,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):
    """Make res layer."""
    layers = []
    for i, layer_type in enumerate(arch):
        downsample = None
        stride = stride if i == 0 else 1
        if layer_type == 2:
            planes *= 2
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1])
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                groups=groups,
                base_width=base_width,
                base_channel=base_channel,
                downsample=downsample,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNeXt_Variant(ResNet_Variant):
    """Class of ResNet_Variant backbone."""

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.base_width = base_width
        self.res_layers = []
        total_expand = 0
        inplanes = planes = self.base_channel
        for i, arch in enumerate(self.arch):
            num_expand = arch.count(2)
            total_expand += num_expand
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            res_layer = make_res_layer(
                self.block,
                inplanes,
                planes,
                arch,
                groups=self.groups,
                base_width=self.base_width,
                base_channel=self.base_channel,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=self.gen_attention,
                gen_attention_blocks=self.stage_with_gen_attention[i])
            planes = self.base_channel * 2 ** total_expand
            inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = self.block.expansion * self.base_channel * 2 ** total_expand
