"""Extend of shared head with res & resxt layer."""

from collections import OrderedDict

from .module import Module
from .get_space import shared_head
from ..utils import dict2str


class SharedHead(Module):
    """Base class of share head."""

    type = 'SharedHead'
    component = 'shared_head'
    quest_dict = dict(train_from_scratch=('optimizer', 'train_from_scratch'),
                      depth=('backbone', 'depth'),
                      stage=('backbone', 'num_stages'))

    def __init__(self, train_from_scratch, depth, stage,
                 fore_part=None, *args, **kwargs):
        super(SharedHead, self).__init__(fore_part=fore_part, *args, **kwargs)
        self.train_from_scratch = train_from_scratch
        self.depth = depth
        self.stage = stage
        if self.train_from_scratch:
            self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
            self.conv_cfg = dict(type='ConvWS')
        else:
            self.norm_cfg = dict(type='BN', requires_grad=False)
            self.conv_cfg = None


@shared_head.register_space
class ResLayer(SharedHead):
    """Class of ResNet layer."""

    type = 'ResLayer'
    module_space = {}
    id_attrs = ['depth', 'stage']

    def __init__(self,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 norm_cfg=OrderedDict(type='BN', requires_grad=False),
                 norm_eval=True,
                 with_cp=False,
                 dcn=None,
                 *args,
                 **kwargs):
        super(ResLayer, self).__init__(*args, **kwargs)

        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.dcn = dcn

    @property
    def config(self):
        """Return config."""
        cfg = OrderedDict(
            type='ResLayer',
            depth=self.depth,
            stage=self.stage,
            stride=self.stride,
            dilation=self.dilation,
            style='pytorch',
            norm_cfg=self.norm_cfg,
            norm_eval=True)
        return dict2str(cfg, tab=2)


@shared_head.register_space
class ResXLayer(ResLayer):
    """Class of ResXt layer."""

    quest_dict = dict(train_from_scratch=('optimizer', 'train_from_scratch'),
                      depth=('backbone', 'depth'),
                      stage=('backbone', 'num_stages'),
                      groups=('backbone', 'groups'),
                      base_width=('backbone', 'base_width'))

    def __init__(self, groups, base_width, *args, **kwargs):
        super(ResXLayer, self).__init__(*args, **kwargs)
        self.groups = groups
        self.base_width = base_width

    @property
    def config(self):
        """Return config."""
        cfg = OrderedDict(
            type='ResXLayer',
            depth=self.depth,
            stage=self.stage,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            base_width=self.base_width,
            style='caffe',
            norm_cfg=self.norm_cfg,
            norm_eval=True)
        return dict2str(cfg, tab=2)
