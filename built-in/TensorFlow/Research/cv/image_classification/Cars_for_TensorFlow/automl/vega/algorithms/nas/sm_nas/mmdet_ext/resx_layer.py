"""ResXLayer."""

import logging

import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmdet.models.registry import SHARED_HEADS

from .resnext_variant import ResNeXt_Variant as ResNeXt
from .resnext_variant import make_res_layer


@SHARED_HEADS.register_module
class ResXLayer(nn.Module):
    """Class of ResX layer."""

    def __init__(self,
                 depth,
                 stage=3,
                 stride=2,
                 dilation=1,
                 groups=4,
                 base_width=32,
                 style='caffe',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 dcn=None):
        super(ResXLayer, self).__init__()
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stage = stage
        self.fp16_enabled = False
        block, stage_blocks = ResNeXt.arch_settings[depth]
        stage_block = stage_blocks[stage]
        self.base_channel = groups * base_width
        planes = self.base_channel * 2 ** stage
        inplanes = self.base_channel * 2 ** (stage - 1) * block.expansion
        res_layer = make_res_layer(
            block,
            inplanes,
            planes,
            stage_block,
            stride=stride,
            dilation=dilation,
            groups=groups,
            base_width=base_width,
            style=style,
            with_cp=with_cp,
            norm_cfg=self.norm_cfg,
            dcn=dcn)
        self.add_module('layer{}'.format(stage + 1), res_layer)

    def init_weights(self, pretrained=None):
        """Init weight."""
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward compute."""
        res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
        out = res_layer(x)
        return out

    def train(self, mode=True):
        """Train mode."""
        super(ResXLayer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
