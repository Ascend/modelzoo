"""Base Modules for BaseHead."""

from collections import OrderedDict

from ...utils import dict2str
from ..get_space import bbox_head
from ..module import Module
from .base_head import BaseHead


@bbox_head.register_space
class BBoxHead(BaseHead):
    """Class of bounding box head."""

    type = 'BBoxHead'
    module_space = {'bbox_head': 'BBoxHead'}
    target_means = [0., 0., 0., 0.]
    target_stds = [0.1, 0.1, 0.2, 0.2]

    def __init__(self,
                 reg_class_agnostic=False,
                 with_avg_pool=True,
                 loss_cls=OrderedDict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=OrderedDict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 **kwargs
                 ):
        super(BBoxHead, self).__init__(reg_class_agnostic=reg_class_agnostic,
                                       with_avg_pool=with_avg_pool,
                                       loss_cls=loss_cls,
                                       loss_bbox=loss_bbox,
                                       **kwargs)

    @property
    def config(self):
        """Return config of the box head."""
        config = OrderedDict(
            type='BBoxHead',
            with_avg_pool=self.with_avg_pool,
            roi_feat_size=self.roi_feat_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            reg_class_agnostic=self.reg_class_agnostic,
            loss_cls=self.loss_cls,
            loss_bbox=self.loss_bbox
        )
        return dict2str(config, tab=2)
