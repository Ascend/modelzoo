"""Backbone Module for ResNeXt."""

import re
from collections import OrderedDict

from ...utils import dict2str
from ..get_space import backbone
from .backbone import Backbone


@backbone.register_space
class ResNeXt(Backbone):
    """Class of resnet."""

    type = 'ResNeXt'
    module_space = {'backbone': 'ResNeXt',
                    'shared_head': 'ResXLayer'}
    attr_space = {'depth': [50, 101]}
    id_attrs = ['depth', 'groups', 'base_width', 'num_stages']
    _base_expansions = {50: 4, 101: 4}
    _base_strides = (1, 2, 2, 2)
    _base_dilations = (1, 1, 1, 1)
    _base_out_indices = (0, 1, 2, 3)

    def __init__(self, depth=50, groups=32, base_width=4, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.base_channel = 64
        self.zero_init_residual = not self.train_from_scratch
        self.frozen_stages = 1 if not self.train_from_scratch else -1

        self.channel_scale = 1
        self.groups = groups
        self.base_width = base_width
        self.num_stages = 4 if self.with_neck else 3
        self.strides = self._base_strides[:self.num_stages]
        self.dilations = self._base_dilations[:self.num_stages]
        self.out_indices = self._base_out_indices[:] if self.with_neck else (2,)
        self.out_strides = [
            2 ** (i + 2) for i in range(self.num_stages)] if self.with_neck else [16]
        self.base_channel = 64 * self.channel_scale
        expansion = self._base_expansions[self.depth]
        first_layer_channel = self.base_channel * expansion
        self.out_channels = [first_layer_channel * self.channel_scale * (2 ** i) for i in range(4)] \
            if self.with_neck else [first_layer_channel * self.channel_scale * 4]

    def __str__(self):
        """Str."""
        return self.type + str(self.depth) + \
            '({}x{}d)'.format(self.groups, self.base_width)

    @property
    def checkpoint_name(self):
        """Get checkpoint name."""
        name = self.type + str(self.depth) + \
            '_{}x{}d'.format(self.groups, self.base_width)
        return name.lower()

    @classmethod
    def arch_decoder(cls, arch: str):
        """Return the params of model."""
        if arch.startswith('ResNeXt') is False:
            raise Exception('arch must start with ResNeXt')
        params = dict()
        if 'x' not in arch:
            depth = int(arch.strip('ResNeXt'))
            params.update(depth=depth)
        else:
            depth, groups, base_width = map(int, re.findall(r'(\d+)', arch))
            params.update(depth=depth, groups=groups, base_width=base_width)
        return params

    @property
    def config(self):
        """Return model config."""
        config = OrderedDict(
            type='ResNeXt',
            depth=self.depth,
            groups=self.groups,
            base_width=self.base_width,
            num_stages=self.num_stages,
            strides=self.strides,
            dilations=self.dilations,
            out_indices=self.out_indices,
            frozen_stages=self.frozen_stages,
            zero_init_residual=self.zero_init_residual,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            style='caffe')
        return dict2str(config, tab=2)
