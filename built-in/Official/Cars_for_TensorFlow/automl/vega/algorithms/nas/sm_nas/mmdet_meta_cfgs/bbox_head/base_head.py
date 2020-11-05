"""Modules for BaseHead."""

from collections import OrderedDict

from ...utils import dict2str
from ..get_space import bbox_head
from ..module import Module


class BaseHead(Module):
    """Class of box head."""

    type = 'BaseHead'
    component = 'bbox_head'
    quest_dict = dict(train_from_scratch=('optimizer', 'train_from_scratch'),
                      num_classes=('dataset', 'num_classes'),
                      with_neck=('detector', 'with_neck'))

    def __init__(self,
                 train_from_scratch,
                 num_classes,
                 with_neck,
                 in_channels,
                 roi_feat_size,
                 reg_class_agnostic=False,
                 with_avg_pool=True,
                 loss_cls=OrderedDict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=OrderedDict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 *args,
                 **kwargs
                 ):
        super(BaseHead, self).__init__(*args, **kwargs)
        self.train_from_scratch = train_from_scratch
        self.num_classes = num_classes
        self.with_neck = with_neck
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size

        self.with_avg_pool = with_avg_pool
        self.reg_class_agnostic = reg_class_agnostic
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        if self.train_from_scratch:
            self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
            self.conv_cfg = dict(type='ConvWS')
        else:
            self.norm_cfg = None
            self.conv_cfg = None

    @classmethod
    def quest_param(cls, fore_part=None):
        """Return params of the head."""
        params = super().quest_param(fore_part=fore_part)
        with_neck = params.get('with_neck')
        in_channels = cls.quest_from(fore_part['neck'], 'out_channels') if with_neck else \
            cls.quest_from(fore_part['backbone'], 'out_channels')[-1] * 2
        roi_feat_size = cls.quest_from(fore_part['roi_extractor'], 'out_size')
        if not with_neck:
            roi_feat_size //= 2
        params.update(in_channels=in_channels, roi_feat_size=roi_feat_size)
        return params
