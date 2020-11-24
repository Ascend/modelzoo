"""AnchorHead Modules for RPNHead."""

from collections import OrderedDict

from .module import Module
from .get_space import rpn_head
from ..utils import dict2str


class AnchorHead(Module):
    """Base class of anchor head."""

    type = 'AnchorHead'
    component = 'rpn'
    quest_dict = dict(with_neck=('detector', 'with_neck'))

    def __init__(self,
                 with_neck,
                 in_channels,
                 anchor_strides,
                 num_classes=2,
                 anchor_base_sizes=None,
                 fore_part=None,
                 loss_cls=None,
                 loss_bbox=None,
                 *args,
                 **kwargs):
        super(AnchorHead, self).__init__(fore_part=fore_part)
        # set quested params
        self.with_neck = with_neck
        self.in_channels = in_channels
        self.anchor_strides = anchor_strides
        # set other params
        self.num_classes = num_classes
        self.anchor_base_sizes = anchor_base_sizes
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox

    @classmethod
    def quest_param(cls, fore_part=None, **kwargs):
        """Return quest param."""
        params = super().quest_param(fore_part=fore_part, **kwargs)
        with_neck = params.get('with_neck')
        if with_neck:
            in_channels = cls.quest_from(fore_part['neck'], 'out_channels')
            anchor_strides = cls.quest_from(fore_part['neck'], 'out_strides')
        else:
            in_channels = cls.quest_from(
                fore_part['backbone'], 'out_channels')[-1]
            anchor_strides = cls.quest_from(
                fore_part['backbone'], 'out_strides')
        params.update(in_channels=in_channels, anchor_strides=anchor_strides)
        return params


@rpn_head.register_space
class RPNHead(AnchorHead):
    """Class of rpn head."""

    type = 'RPNHead'
    module_space = {'rpn_head': 'RPNHead'}
    id_attrs = ['anchor_scales',
                'anchor_ratios',
                'anchor_strides',
                'feat_channels']
    target_means = [.0, .0, .0, .0]
    target_stds = [1.0, 1.0, 1.0, 1.0]

    def __init__(self,
                 loss_cls=OrderedDict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_bbox=OrderedDict(
                     type='SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=1.0),
                 anchor_ratios=None,
                 anchor_scales=None,
                 *args,
                 **kwargs):
        super(RPNHead, self).__init__(*args, **kwargs)
        # channel setting
        self.feat_channels = self.in_channels
        # anchor strides, scales and ratios
        if anchor_ratios is not None:
            self.anchor_ratios = anchor_ratios
        else:
            self.anchor_ratios = [0.5, 1.0, 2.0]
        if anchor_scales is not None:
            self.anchor_scales = anchor_scales
        else:
            if self.with_neck:
                self.anchor_scales = [8]
            else:
                self.anchor_scales = [2, 4, 8, 16, 32]

        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox

    def __str__(self):
        """Get arch code."""
        return 'RPN'

    @property
    def config(self):
        """Return config."""
        config = OrderedDict(
            type='RPNHead',
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            anchor_scales=self.anchor_scales,
            anchor_ratios=self.anchor_ratios,
            anchor_strides=self.anchor_strides,
            target_means=self.target_means,
            target_stds=self.target_stds,
            loss_cls=self.loss_cls,
            loss_bbox=self.loss_bbox
        )
        return dict2str(config, tab=2)


@rpn_head.register_space
class GARPNHead(AnchorHead):
    """Class of guide anchor rpn head."""

    type = 'GARPNHead'
    module_space = {'rpn_head': 'GARPNHead'}
    id_attrs = ['loc_filter_thr',
                'octave_base_scale',
                'scales_per_octave',
                'octave_ratios']
    target_means = [.0, .0, .0, .0]
    target_stds = [0.07, 0.07, 0.14, 0.14]
    anchoring_means = [.0, .0, .0, .0]
    anchoring_stds = [0.07, 0.07, 0.14, 0.14]

    def __init__(self,
                 octave_base_scale=8,
                 scales_per_octave=3,
                 loc_filter_thr=0.01,
                 octave_ratios=[0.5, 1.0, 2.0],
                 loss_loc=OrderedDict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_shape=OrderedDict(
                     type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
                 loss_cls=OrderedDict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_bbox=OrderedDict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 **kwargs):
        super(GARPNHead, self).__init__(**kwargs)
        self.loc_filter_thr = loc_filter_thr
        # channel setting
        self.feat_channels = self.in_channels
        # anchor strides scales and ratios
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.octave_ratios = octave_ratios
        # self.octave_ratios = [0.5, 1.0, 2.0]
        # losses
        self.loss_loc = loss_loc
        self.loss_shape = loss_shape
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox

    def __str__(self):
        """Get arch code."""
        return 'GARPN'

    @property
    def config(self):
        """Return config str."""
        config = dict(
            type='GARPNHead',
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            octave_base_scale=self.octave_base_scale,
            scales_per_octave=self.scales_per_octave,
            octave_ratios=self.octave_ratios,
            anchor_strides=self.anchor_strides,
            anchor_base_sizes=self.anchor_base_sizes,
            anchoring_means=self.anchoring_means,
            anchoring_stds=self.anchoring_stds,
            target_means=self.target_means,
            target_stds=self.target_stds,
            loc_filter_thr=self.loc_filter_thr,
            loss_loc=self.loss_loc,
            loss_shape=self.loss_shape,
            loss_cls=self.loss_cls,
            loss_bbox=self.loss_bbox)
        return dict2str(config, tab=2)
