"""BBoxHead Modules for ConvFCBBoxHead."""

from collections import OrderedDict

from ...utils import dict2str
from ..get_space import bbox_head
from .base_head import BaseHead


@bbox_head.register_space
class ConvFCBBoxHead(BaseHead):
    """Class conv fc bounding box head."""

    type = 'ConvFCBBoxHead'
    module_space = {'bbox_head': 'ConvFCBBoxHead'}

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 normalize=None,
                 **kwargs
                 ):
        super(ConvFCBBoxHead, self).__init__(**kwargs)
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.normalize = normalize

    @property
    def config(self):
        """Return config."""
        config = OrderedDict(
            type='ConvFCBBoxHead',
            in_channels=self.in_channels,
            num_shared_convs=self.num_shared_convs,
            num_shared_fcs=self.num_shared_fcs,
            num_cls_convs=self.num_cls_convs,
            num_cls_fcs=self.num_cls_fcs,
            num_reg_convs=self.num_reg_convs,
            num_reg_fcs=self.num_reg_fcs,
            conv_out_channels=self.conv_out_channels,
            fc_out_channels=self.fc_out_channels,
            roi_feat_size=self.roi_feat_size,
            num_classes=self.num_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            reg_class_agnostic=self.reg_class_agnostic,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
        )
        return dict2str(config, tab=2)


@bbox_head.register_space
class SharedFCBBoxHead(BaseHead):
    """Class of share fc layers bounding box head."""

    type = 'SharedFCBBoxHead'
    module_space = {'bbox_head': 'SharedFCBBoxHead'}
    id_attrs = ['num_fcs', 'fc_out_channels', 'target_means', 'target_stds']
    base_target_means = [0., 0., 0., 0.]
    base_target_stds = [0.1, 0.1, 0.2, 0.2]

    def __init__(self,
                 num_fcs=2,
                 fc_out_channels=1024,
                 reg_class_agnostic=False,
                 loss_cls=OrderedDict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=OrderedDict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 **kwargs):
        super(SharedFCBBoxHead, self).__init__(reg_class_agnostic=reg_class_agnostic,
                                               loss_cls=loss_cls,
                                               loss_bbox=loss_bbox,
                                               **kwargs)
        self.num_fcs = num_fcs
        self.fc_out_channels = fc_out_channels

        self.has_ga_rpn = 'GA' in self.quest_from(
            self.model['rpn_head'], 'name')
        # change target_std according to whether use ga rpn
        self.target_means = self.base_target_means[:]
        self.target_stds = self.base_target_stds[:] if not self.has_ga_rpn else \
            list(map(lambda x: x / 2, self.base_target_stds[:]))

    @property
    def config(self):
        """Return string of config."""
        config = OrderedDict(
            type='SharedFCBBoxHead',
            num_fcs=self.num_fcs,
            in_channels=self.in_channels,
            fc_out_channels=self.fc_out_channels,
            roi_feat_size=self.roi_feat_size,
            num_classes=self.num_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            reg_class_agnostic=self.reg_class_agnostic,
            loss_cls=self.loss_cls,
            loss_bbox=self.loss_bbox
        )
        return dict2str(config, tab=2)
