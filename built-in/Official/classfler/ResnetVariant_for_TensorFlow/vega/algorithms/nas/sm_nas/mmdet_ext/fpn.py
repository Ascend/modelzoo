"""FPN_."""

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmdet.core import auto_fp16
from mmdet.models.necks import FPN
from mmdet.models.registry import NECKS


@NECKS.register_module
class FPN_(FPN):
    """Class of FPN_."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 strides=[1, 2, 2, 2]):
        super(FPN_, self).__init__(in_channels=in_channels,
                                   out_channels=out_channels,
                                   num_outs=num_outs,
                                   start_level=start_level,
                                   end_level=end_level,
                                   add_extra_convs=add_extra_convs,
                                   extra_convs_on_inputs=extra_convs_on_inputs,
                                   relu_before_extra_convs=relu_before_extra_convs,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   activation=activation)
        assert len(strides) == self.num_ins
        self.strides = strides

    @auto_fp16()
    def forward(self, inputs):
        """Forward compute."""
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            scale_factor = self.strides[i]
            if scale_factor != 1:
                laterals[i - 1] += F.interpolate(
                    laterals[i], scale_factor=scale_factor, mode='nearest')
            else:
                laterals[i - 1] += laterals[i]
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
