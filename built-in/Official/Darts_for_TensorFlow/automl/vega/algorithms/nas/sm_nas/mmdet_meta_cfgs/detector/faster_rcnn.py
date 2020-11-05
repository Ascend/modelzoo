"""Detector Modules for faster r-cnn."""

from collections import OrderedDict

from ...utils import dict2str, str_warp
from ..get_space import detector
from .detector import Detector


@detector.register_space
class FasterRCNN(Detector):
    """Class of faster rcnn."""

    type = 'FasterRCNN'
    module_space = {'detector': 'FasterRCNN',
                    'neck': 'FPN',
                    'rpn_head': 'RPNHead',
                    'shared_head': None,
                    'bbox_head': 'SharedFCBBoxHead',
                    }
    id_attrs = ['rpn_samples', 'rcnn_samples']
    with_neck = True

    def __init__(self, rpn_samples=256, rcnn_samples=384, **kwargs):
        super(FasterRCNN, self).__init__(**kwargs)
        self.rpn_samples = rpn_samples
        self.rcnn_samples = rcnn_samples

    @property
    def config(self):
        """Return config."""
        return str_warp(self.type)

    @property
    def train_cfg(self):
        """Return train config."""
        cfg = OrderedDict(
            rpn=OrderedDict(
                assigner=OrderedDict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=OrderedDict(
                    type='RandomSampler',
                    num=self.rpn_samples,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=OrderedDict(
                nms_across_levels=False,
                nms_pre=2000,
                nms_post=2000,
                max_num=2000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=OrderedDict(
                assigner=OrderedDict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=OrderedDict(
                    type='RandomSampler',
                    num=self.rcnn_samples,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False))
        return dict2str(cfg, tab=1)

    @property
    def test_cfg(self):
        """Return test config."""
        cfg = OrderedDict(
            rpn=OrderedDict(
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=OrderedDict(
                score_thr=0.05, nms=OrderedDict(type='nms', iou_thr=0.5), max_per_img=100))
        return dict2str(cfg, tab=1)
