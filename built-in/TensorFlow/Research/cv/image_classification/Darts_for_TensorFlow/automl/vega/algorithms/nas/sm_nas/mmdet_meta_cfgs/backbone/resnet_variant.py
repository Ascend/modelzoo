"""Backbone Modules for ResNet Variant."""

import random
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from ...utils import ListDict, dict2str
from ..get_space import backbone
from .backbone import Backbone


@backbone.register_space
class ResNet_Variant(Backbone):
    """Class of ResNet Variant."""

    type = 'ResNet_Variant'
    module_space = {'backbone': 'ResNet_Variant'}
    attr_space = dict(arch=dict(num_reduction=3,
                                num_stage=4,
                                num_block=[5, 15]),
                      base_channel=[48, 56, 64, 72])
    id_attrs = ['base_channel', 'arch', 'base_depth']
    _block_setting = {18: ('BasicBlock', 8),
                      34: ('BasicBlock', 16),
                      50: ('Bottleneck', 16),
                      101: ('Bottleneck', 33)}
    _base_strides = (1, 2, 2, 2)
    _base_dilations = (1, 1, 1, 1)
    _base_out_indices = (0, 1, 2, 3)

    def __init__(self,
                 arch,
                 base_depth,
                 base_channel,
                 *args,
                 **kwargs):
        super(ResNet_Variant, self).__init__(*args, **kwargs)
        self.arch = arch
        self.base_channel = int(base_channel)
        self.depth = self.base_depth = base_depth
        block = self._block_setting[base_depth][0]
        if block not in ['BasicBlock', 'Bottleneck']:
            raise Exception(
                'Invalid block name. (should be BasicBlock or Bottleneck)')
        expansion = 1 if block == 'BasicBlock' else 4

        if self.train_from_scratch:
            self.zero_init_residual = False
            self.frozen_stages = -1
            self.norm_cfg = dict(
                type='GN',
                num_groups=int(
                    self.base_channel / 2),
                requires_grad=True)
        else:
            self.zero_init_residual = True
            self.frozen_stages = 1
        self.num_stages = len(arch.split('-'))
        self.strides = self._base_strides[:self.num_stages]
        self.dilations = self._base_dilations[:self.num_stages]
        self.out_indices = self._base_out_indices[:self.num_stages] if self.with_neck else (
            self.num_stages - 1,)
        self.out_strides = [
            2 ** (i + 2) for i in range(self.num_stages)] if self.with_neck else [16]

        num_scale = 0
        self.out_channels = []
        for stage in range(self.num_stages):
            n = self.arch.split('-')[stage].count('2')
            num_scale += n
            self.out_channels.append(
                self.base_channel * expansion * (2 ** num_scale))

    def __str__(self):
        """Get arch code."""
        return self.arch_code

    @property
    def arch_code(self):
        """Return arch code."""
        return 'r{}_{}_{}'.format(self.base_depth, self.base_channel, self.arch)

    @property
    def base_flops(self):
        """Get base flops."""
        from ...utils import base_flops, counter
        input_size = '{}x{}'.format(*self.input_size)
        base_flops = base_flops.ResNet if self.with_neck else base_flops.ResNet_C4
        flops = base_flops[self.base_depth].get(input_size, None)
        if flops is None:
            from ...utils import profile
            from mmdet.models import ResNet
            base_model = ResNet(self.base_depth,
                                num_stages=self.num_stages,
                                strides=self.strides,
                                dilations=self.dilations,
                                out_indices=self.out_indices)
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
        """Decode arch code."""
        base_arch_code = {18: 'r18_64_11-21-21-21',
                          34: 'r34_64_111-2111-211111-211',
                          50: 'r50_64_111-2111-211111-211',
                          101: 'r101_64_111-2111-21111111111111111111111-211'}
        if arch_code.startswith('ResNet'):
            base_depth = int(arch_code.split('ResNet')[-1])
            arch_code = base_arch_code[base_depth]
        try:
            m = re.match(
                r'r(?P<base_depth>.*)_(?P<base_channel>.*)_(?P<arch>.*)',
                arch_code)
            base_depth, base_channel = map(int, m.groups()[:-1])
            arch = m.group('arch')
            return dict(base_depth=base_depth,
                        base_channel=base_channel, arch=arch)
        except BaseException:
            raise ValueError('Cannot parse arch code {}'.format(arch_code))

    @property
    def flops_ratio(self):
        """Get flops ratio."""
        return round(self.size_info['FLOPs'] / self.base_flops, 3)

    @classmethod
    def sample(cls,
               method='random',
               base_depth=50,
               base_arch=None,
               sampled_archs=[],
               flops_constraint=None,
               EA_setting=dict(num_mutate=3),
               fore_part=None,
               max_sample_num=100000,
               **kwargs
               ):
        """Sample a model."""
        if flops_constraint is None:
            low_flops, high_flops = 0, float('inf')
        else:
            low_flops, high_flops = flops_constraint
        sample_num = 0
        discard = ListDict()
        params = cls.quest_param(fore_part=fore_part, **kwargs)
        while sample_num < max_sample_num:
            sample_num += 1
            if method == 'random':
                # with_neck = params.get('with_neck')
                params.update(cls.random_sample())
            elif method == 'EA':
                params.update(cls.EA_sample(base_arch=base_arch, **EA_setting))
            elif method == 'proposal':
                params.update(cls.arch_decoder(arch_code=base_arch))
            else:
                raise ValueError('Unrecognized sample method {}.')
            # construct config
            net = cls(**params, base_depth=base_depth, fore_part=fore_part)
            exist = net.name in sampled_archs + discard['arch']
            success = low_flops <= net.flops_ratio <= high_flops
            state = 'Exist' if exist else 'Success' * \
                success + 'Discard' * (not success)
            flops_info = '{}({})'.format(net.flops, net.flops_ratio)
            print(
                'Sample {}{}: {}; FLOPs={}; {}.'.format(
                    method,
                    sample_num,
                    net.name,
                    flops_info,
                    state))
            if exist:
                continue
            if success:
                return dict(arch=net, discard=discard)
            else:
                discard.append(dict(arch=net.name, flops=flops_info))

        print('Unable to get structure that satisfies the FLOPs constraint.')
        return None

    @classmethod
    def random_sample(cls):
        """Random sample a model arch."""
        arch_space = cls.attr_space['arch']
        num_reduction, num_stage = arch_space.get(
            'num_reduction'), arch_space.get('num_stage')
        base_channel = random.choice(cls.attr_space['base_channel'])
        length = random.randint(*arch_space['num_block'])
        arch = ['1'] * length
        position = np.random.choice(length, size=num_reduction, replace=False)
        for p in position:
            arch[p] = '2'
        insert = np.random.choice(length - 1, size=num_stage - 1, replace=False)
        insert = [i + 1 for i in insert]
        insert = reversed(sorted(insert))
        for i in insert:
            arch.insert(i, '-')
        return dict(base_channel=base_channel, arch=''.join(arch))

    @classmethod
    def is_valid(cls, arch):
        """Return if the arch in search space."""
        arch_space = cls.attr_space['arch']
        min_block, max_block = arch_space['num_block']
        stages = arch.split('-')
        length = 0
        for stage in stages:
            if len(stage) == 0:
                return False
            length += len(stage)
        return min_block <= length <= max_block

    @classmethod
    def _chwidth(cls, cur_channel):
        """Return new channel number."""
        base_channel = cls.attr_space['base_channel'].copy()
        try:
            base_channel.sort()
            index = base_channel.index(cur_channel)
            candidate = [
                i for i in range(index - 1, index + 2)
                if 0 <= i < len(base_channel) and i != index]
            channel = cls.base_channels[random.choice(candidate)]
        except BaseException:
            channel = random.choice(base_channel)
        return channel

    @classmethod
    def _insert(cls, arch):
        """Return new arch code."""
        idx = np.random.randint(low=0, high=len(arch))
        arch.insert(idx, '1')
        return arch, idx

    @classmethod
    def _remove(cls, arch):
        """Return new arch code."""
        ones_index = [i for i, char in enumerate(arch) if char == '1']
        idx = random.choice(ones_index)
        arch.pop(idx)
        return arch, idx

    @classmethod
    def _swap(cls, arch, R):
        """Return new arch code."""
        while True:
            not_ones_index = [
                i for i, char in enumerate(arch) if char != '1']
            idx = random.choice(not_ones_index)
            r = random.randint(1, R)
            direction = -r if random.random() > 0.5 else r
            try:
                arch[idx], arch[idx + direction] = \
                    arch[idx + direction], arch[idx]
                break
            except BaseException:
                continue
        return arch, idx, direction

    @classmethod
    def EA_sample(cls, base_arch, num_mutate=3, **kwargs):
        """Use ea to sample a model."""
        params = cls.arch_decoder(base_arch)
        base_channel, base_arch = params.get('base_channel'), params.get('arch')
        # whether to mutate base_channel
        while True:
            ops = []
            new_arch = list(base_arch)
            new_channel = base_channel
            try:
                if random.random() > 0.5:
                    new_channel = cls._chwidth(base_channel)
                    ops.append('chwidth:{}->{}'.format(base_channel, new_channel))
                for i in range(num_mutate):
                    op_idx = np.random.randint(low=0, high=3)
                    if op_idx == 0:
                        new_arch, idx = cls._insert(new_arch)
                        ops.append('insert:{}'.format(idx))
                    elif op_idx == 1:
                        new_arch, idx = cls._remove(new_arch)
                        ops.append('remove:{}'.format(idx))
                    elif op_idx == 2:
                        R = num_mutate // 2
                        new_arch, idx, direction = cls._swap(new_arch, R)
                        ops.append('swap:{}&{}'.format(idx, idx + direction))
                    else:
                        raise ('operation index out of range')
            except BaseException:
                continue
            new_arch = ''.join(new_arch)
            if cls.is_valid(new_arch) and (
                    new_arch != base_arch or new_channel != base_channel):
                break
        print(' --> '.join(ops))
        return dict(base_channel=new_channel, arch=new_arch)

    @property
    def config(self):
        """Return config dict."""
        config = OrderedDict(
            type='ResNet_Variant',
            arch=self.arch,
            base_depth=self.base_depth,
            base_channel=self.base_channel,
            num_stages=self.num_stages,
            strides=self.strides,
            dilations=self.dilations,
            out_indices=self.out_indices,
            frozen_stages=self.frozen_stages,
            zero_init_residual=self.zero_init_residual,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            style='pytorch')
        return dict2str(config, tab=2)
