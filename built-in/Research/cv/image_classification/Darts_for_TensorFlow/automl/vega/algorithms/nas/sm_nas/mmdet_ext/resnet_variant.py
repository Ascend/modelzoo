"""Modules for ResNet_Variant."""

import torch.nn as nn
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.registry import BACKBONES
from mmdet.models.utils import build_conv_layer, build_norm_layer
from torch.nn.modules.batchnorm import _BatchNorm


def make_res_layer(block,
                   inplanes,
                   planes,
                   arch,
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
    """Make resnet layer."""
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
class ResNet_Variant(ResNet):
    """Class of ResNet_Variant backbone."""

    def __init__(self,
                 arch,
                 base_depth,
                 base_channel,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        self.arch = [[int(i) for i in stage] for stage in arch.split('-')]
        self.base_depth = base_depth
        self.base_channel = base_channel
        self.stage_with_gen_attention = stage_with_gen_attention
        assert len(strides) == len(dilations) == num_stages == len(self.arch)

        super(ResNet_Variant, self).__init__(
            base_depth,
            in_channels=3,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=stage_with_dcn,
            gcb=gcb,
            stage_with_gcb=stage_with_gcb,
            gen_attention=gen_attention,
            stage_with_gen_attention=stage_with_gen_attention,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual)
        # self._make_stem_layer(3)
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
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            planes = self.base_channel * 2 ** total_expand
            inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = self.block.expansion * self.base_channel * 2 ** total_expand

    @property
    def norm2(self):
        """Get norm for layer 2."""
        return getattr(self, self.norm2_name)

    def _make_stem_layer(self, input_channels):
        """Make stem layer."""
        stem_channel = self.base_channel // 2
        norm_cfg = self.norm_cfg.copy()
        if self.norm_cfg.get('type') == 'GN':
            num_groups = norm_cfg.get('num_groups')
            norm_cfg['num_groups'] = int(num_groups / 2)
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            input_channels,
            stem_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, stem_channel, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            stem_channel,
            self.base_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, self.base_channel, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

    def _freeze_stages(self):
        """Freeze stages."""
        super(ResNet_Variant, self)._freeze_stages()

        if self.frozen_stages >= 1:
            self.norm2.eval()
            for m in [self.conv2, self.norm2]:
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :type x: torch.Tensor

        :return: out feature map
        :rtype: torch.Tensor
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Train.

        :param mode: if train
        :type mode: bool
        """
        super(ResNet_Variant, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
