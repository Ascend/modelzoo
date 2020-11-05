"""Backbone Modules for ResNet."""

import random
from collections import OrderedDict
from copy import copy

from ...utils import dict2str
from ..get_space import backbone
from .backbone import Backbone


@backbone.register_space
class ResNet(Backbone):
    """Class of ResNet set."""

    type = 'ResNet'
    module_space = {'backbone': 'ResNet',
                    'shared_head': 'ResLayer'}
    attr_space = {'depth': [18, 34, 50, 101]}
    id_attrs = ['depth', 'num_stages']
    _base_expansions = {18: 1, 34: 1, 50: 4, 101: 4}
    _base_strides = (1, 2, 2, 2)
    _base_dilations = (1, 1, 1, 1)
    _base_out_indices = (0, 1, 2, 3)

    def __init__(self,
                 depth=50,
                 channel_scale=1,
                 **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.depth = depth
        self.channel_scale = channel_scale
        self.base_channel = 64

        self.frozen_stages = 1 if not self.train_from_scratch else -1
        self.zero_init_residual = not self.train_from_scratch
        self.num_stages = 4 if self.with_neck else 3
        self.strides = self._base_strides[:self.num_stages]
        self.dilations = self._base_dilations[:self.num_stages]
        self.out_indices = self._base_out_indices[:] if self.with_neck else (2,)
        self.out_strides = [
            2 ** (i + 2) for i in range(self.num_stages)] if self.with_neck else [16]
        expansion = self._base_expansions[self.depth]
        first_layer_channel = self.base_channel * expansion
        self.out_channels = [first_layer_channel * self.channel_scale * (2 ** i) for i in range(4)] \
            if self.with_neck else [first_layer_channel * self.channel_scale * 4]

    def __str__(self):
        """Str."""
        return self.type + str(self.depth)

    @classmethod
    def arch_decoder(cls, arch: str):
        """Decoce arch code."""
        if arch.startswith('ResNet') is False:
            raise Exception('arch must start with ResNet')
        depth = int(arch.strip('ResNet'))
        return dict(depth=depth)

    @property
    def config(self):
        """Return config."""
        cfg = OrderedDict(
            type='ResNet',
            depth=self.depth,
            num_stages=self.num_stages,
            out_indices=self.out_indices,
            strides=self.strides,
            dilations=self.dilations,
            frozen_stages=self.frozen_stages,
            zero_init_residual=self.zero_init_residual,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            style='pytorch')
        return dict2str(cfg, tab=2)
