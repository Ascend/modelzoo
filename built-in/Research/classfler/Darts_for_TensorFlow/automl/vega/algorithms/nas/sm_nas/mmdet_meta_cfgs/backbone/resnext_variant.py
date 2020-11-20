"""Backbone Modules for ResNeXt_Variant."""

import re
from collections import OrderedDict

from ...utils import dict2str, str2dict
from ..get_space import backbone
from . import ResNet_Variant


@backbone.register_space
class ResNeXt_Variant(ResNet_Variant):
    """Class of ResNeXt Variant."""

    type = 'ResNet_Variant'
    module_space = {'backbone': 'ResNeXt_Variant'}
    id_attrs = ['base_channel', 'arch', 'base_depth', 'groups', 'base_width']

    _block_setting = {
        50: ('Bottleneck', 16),
        101: ('Bottleneck', 33)}

    _base_strides = (1, 2, 2, 2)
    _base_dilations = (1, 1, 1, 1)
    _base_out_indices = (0, 1, 2, 3)

    def __init__(self,
                 base_depth,
                 base_channel,
                 arch,
                 base_width=4,
                 *args,
                 **kwargs):
        super().__init__(base_depth=base_depth,
                         base_channel=base_channel,
                         arch=arch,
                         *args, **kwargs)

        self.base_width = base_width
        self.groups = base_channel // 2

    def __str__(self):
        """Get arch code."""
        return self.arch_code

    @property
    def arch_code(self):
        """Return arch code."""
        return 'x{}({}x{}d)_{}_{}'.format(self.base_depth, self.groups, self.base_width,
                                          self.base_channel, self.arch)

    @property
    def base_flops(self):
        """Return flops of the base model."""
        from ...utils import base_flops, counter
        input_size = '{}x{}'.format(*self.input_size)
        flops = base_flops.ResNeXt_32x4d[self.base_depth].get(input_size, None)
        if flops is None:
            from ...utils import profile
            from mmdet.models import ResNeXt
            base_model = ResNeXt(depth=self.base_depth, groups=32, base_width=4)
            flops = profile(
                base_model,
                self.input_size,
                style='normal',
                show_result=False)['FLOPs']
        else:
            flops = counter(flops)
        return flops

    @staticmethod
    def arch_decoder(arch_code: str):
        """Return params of the model."""
        base_arch_code = {50: 'x50(32x4d)_64_111-2111-211111-211',
                          101: 'x101(32x4d)_64_111-2111-21111111111111111111111-211'}
        if arch_code.startswith('ResNeXt'):
            m = re.match(
                r'ResNeXt(?P<base_depth>.*)\((?P<groups>.*)x(?P<base_width>.*)d\)',
                arch_code)
            base_depth = int(m.group('base_depth'))
            arch_code = base_arch_code[base_depth]
        try:
            m = re.match(r'x(?P<base_depth>.*)\((?P<groups>.*)x(?P<base_width>.*)d\)_(?P<base_channel>.*)_(?P<arch>.*)',
                         arch_code)
            base_depth, groups, base_width, base_channel = map(
                int, m.groups()[:-1])
            arch = m.group('arch')
            return dict(base_depth=base_depth, groups=groups, base_width=base_width,
                        base_channel=base_channel, arch=arch)
        except BaseException:
            raise ValueError('Cannot parse arch code {}'.format(arch_code))

    @property
    def config(self):
        """Return config of the model."""
        config = OrderedDict(
            type='ResNeXt_Variant',
            arch=self.arch,
            base_depth=self.base_depth,
            base_channel=self.base_channel,
            groups=self.groups,
            base_width=self.base_width,
            num_stages=self.num_stages,
            strides=self.strides,
            dilations=self.dilations,
            out_indices=self.out_indices,
            frozen_stages=self.frozen_stages,
            zero_init_residual=False,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            style='caffe')
        return dict2str(config, tab=2)
