"""Neck Modules for FPN."""

import math
from collections import OrderedDict
from .module import Module
from .get_space import neck
from ..utils import dict2str


class Neck(Module):
    """Base class of neck."""

    type = 'Neck'
    component = 'neck'
    quest_dict = dict(train_from_scratch=('optimizer', 'train_from_scratch'),
                      in_channels=('backbone', 'out_channels'),
                      in_indices=('backbone', 'out_indices'))

    def __init__(self,
                 train_from_scratch,
                 in_channels,
                 in_indices,
                 fore_part=None, *args, **kwargs):
        super(Neck, self).__init__(fore_part=fore_part, *args, **kwargs)
        self.train_from_scratch = train_from_scratch
        self.in_channels = in_channels
        self.in_indices = in_indices

        if self.train_from_scratch:
            self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
            self.conv_cfg = dict(type='ConvWS')
        else:
            self.norm_cfg = None
            self.conv_cfg = None


@neck.register_space
class FPN(Neck):
    """Class of fpn layer."""

    type = 'FPN'
    module_space = {'neck': 'FPN'}
    attr_space = {'out_channels': [128, 256, 384]}
    id_attrs = ['start_level',
                'out_channels',
                'add_extra_convs',
                'num_outs',
                'extra_convs_on_inputs']
    min_fea_size = (3, 3)

    def __init__(self,
                 start_level=0,
                 out_channels=256,
                 add_extra_convs=False,
                 num_outs=5,
                 extra_convs_on_inputs=True,
                 **kwargs):
        super(FPN, self).__init__(**kwargs)
        self.start_level = start_level
        self.out_channels = out_channels
        self.add_extra_convs = add_extra_convs
        self.num_outs = num_outs

        self.num_ins = len(self.in_channels)
        self.extra_convs_on_inputs = extra_convs_on_inputs
        # limit the stage numbers to guarantee all feature maps' scale bigger
        # than the 'min_fea_size'
        img_scales = self.quest_from(
            self.model['dataset'], 'img_scale')['train']
        if isinstance(img_scales[0], (list, tuple)):
            min_img_scale = min([max(s) for s in img_scales]), min(
                [min(s) for s in img_scales])
        else:
            min_img_scale = img_scales
        max_downsample = min(list(map(lambda x: int(
            math.log2(x[0] / x[1])), zip(min_img_scale, self.min_fea_size))))
        self.num_outs = min(
            self.num_outs,
            max_downsample - self.start_level - 1)
        # add extra levels only when the last stage output of backbone is in
        # neck's 'out_stages'
        if self.num_outs + self.start_level > self.num_ins:
            self.end_level = -1
            self.op = 'c' if self.add_extra_convs else 'p'
        else:
            self.end_level = self.num_outs + self.start_level
            self.add_extra_convs = False
            self.op = 'n'
        self.out_stages = list(
            range(
                self.start_level,
                self.start_level + self.num_outs))
        self.backbone_strides = self.quest_from(
            self.model['backbone'], 'strides')
        self.fea_strides = []
        self.out_strides = []
        for i, stage in enumerate(self.out_stages):
            last_stride = self.out_strides[-1] if i != 0 else 4 * 2 ** stage
            if stage < len(self.backbone_strides):
                stride = self.backbone_strides[stage]
                self.fea_strides.append(last_stride * stride)
            else:
                stride = 2
            self.out_strides.append(last_stride * stride)

    def __str__(self):
        """Get arch code."""
        return '{}({}-{},{},{})'.format(self.type, self.start_level,
                                        self.start_level + self.num_outs - 1,
                                        self.out_channels, self.op)

    @property
    def config(self):
        """Return config of fpn."""
        config = OrderedDict(
            type='FPN_',
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_outs=self.num_outs,
            start_level=self.start_level,
            end_level=self.end_level,
            add_extra_convs=self.add_extra_convs,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            strides=self.backbone_strides
        )
        return dict2str(config, tab=2)
