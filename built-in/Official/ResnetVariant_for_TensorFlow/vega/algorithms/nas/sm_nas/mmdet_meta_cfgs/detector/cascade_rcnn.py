"""Detector Modules for cascade rcnn."""

from collections import OrderedDict

from ...utils import dict2str, str_warp
from ..get_space import detector
from .detector import Detector


@detector.register_space
class CascadeRCNN(Detector):
    """Class of cascade rcnn."""

    type = 'CascadeRCNN'
    module_space = {'detector': 'CascadeRCNN',
                    'bbox_head': 'CascadeFCBBoxHead',
                    'shared_head': None,
                    'rpn_head': 'RPNHead',
                    }
    id_attrs = ['num_stages', 'rpn_samples', 'rcnn_samples']
    attr_space = {'num_stages': [2, 3]}
    with_neck = True
    pos_iou_thr = [0.5, 0.6, 0.7]
    neg_iou_thr = [0.5, 0.6, 0.7]
    min_pos_iou = [0.5, 0.6, 0.7]
    stage_loss_weights = [1, 0.5, 0.25]

    def __init__(self, num_stages=3, rpn_samples=256,
                 rcnn_samples=384, **kwargs):
        super().__init__(**kwargs)
        self.num_stages = num_stages
        self.rpn_samples = rpn_samples
        self.rcnn_samples = rcnn_samples

    @property
    def config(self):
        """Return config."""
        return str_warp(self.type)

    @property
    def train_cfg(self):
        """Return train config."""
        rpn = dict2str(OrderedDict(
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
            debug=False), tab=2)
        rpn_proposal = dict2str(OrderedDict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0), tab=2)
        rcnn = ''
        for i in range(self.num_stages):
            rcnn_ = OrderedDict(
                assigner=OrderedDict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=self.pos_iou_thr[i],
                    neg_iou_thr=self.neg_iou_thr[i],
                    min_pos_iou=self.min_pos_iou[i],
                    ignore_iof_thr=-1),
                sampler=OrderedDict(
                    type='RandomSampler',
                    num=self.rcnn_samples,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
            rcnn += dict2str(rcnn_, tab=2, format_first_line=True) + ',\n'
        rcnn = '[\n{}]'.format(rcnn[:-len(',\n')])
        loss_weights = self.stage_loss_weights[:self.num_stages]
        cfg = (
            "dict(\n"
            f"    rpn={rpn},\n"
            f"    rpn_proposal={rpn_proposal},\n"
            f"    rcnn={rcnn},\n"
            f"    stage_loss_weights={loss_weights})")
        return cfg

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
                score_thr=0.05, nms=OrderedDict(type='nms', iou_thr=0.5), max_per_img=100),
            keep_all_stages=False)
        return dict2str(cfg, tab=1)


@detector.register_space
class GACascadeRCNN(CascadeRCNN):
    """Class of guide anchor cascade rcnn."""

    type = 'CascadeRCNN'
    module_space = {'detector': 'GACascadeRCNN',
                    'rpn_head': 'GARPNHead',
                    'bbox_head': 'CascadeFCBBoxHead',
                    'shared_head': None}
    id_attrs = ['num_stages', 'ga_samples', 'rpn_samples', 'rcnn_samples']

    attr_space = {'num_stages': [2, 3]}
    pos_iou_thr = [0.6, 0.7, 0.75]
    neg_iou_thr = [0.6, 0.7, 0.75]
    min_pos_iou = [0.6, 0.7, 0.75]

    def __init__(self, num_stages=3, ga_samples=256,
                 rpn_samples=256, rcnn_samples=256, **kwargs):
        super(GACascadeRCNN, self).__init__(num_stages=num_stages,
                                            rpn_samples=rpn_samples,
                                            rcnn_samples=rcnn_samples,
                                            **kwargs)
        self.ga_samples = ga_samples

    @property
    def train_cfg(self):
        """Return train config."""
        rpn = dict2str(OrderedDict(
            ga_assigner=OrderedDict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            ga_sampler=OrderedDict(
                type='RandomSampler',
                num=self.ga_samples,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
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
            allowed_border=-1,
            pos_weight=-1,
            center_ratio=0.2,
            ignore_ratio=0.5,
            debug=False), tab=2)
        rpn_proposal = dict2str(OrderedDict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=300,
            nms_thr=0.7,
            min_bbox_size=0), tab=2)
        rcnn = ''
        for i in range(self.num_stages):
            rcnn_ = OrderedDict(
                assigner=OrderedDict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=self.pos_iou_thr[i],
                    neg_iou_thr=self.neg_iou_thr[i],
                    min_pos_iou=self.min_pos_iou[i],
                    ignore_iof_thr=-1),
                sampler=OrderedDict(
                    type='RandomSampler',
                    num=self.rcnn_samples,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
            rcnn += dict2str(rcnn_, tab=2, format_first_line=True) + ',\n'
        rcnn = '[\n{}]'.format(rcnn[:-len(',\n')])
        loss_weights = self.stage_loss_weights[:self.num_stages]
        cfg = (
            "dict(\n"
            f"    rpn={rpn},\n"
            f"    rpn_proposal={rpn_proposal},\n"
            f"    rcnn={rcnn},\n"
            f"    stage_loss_weights={loss_weights})")
        return cfg

    @property
    def test_cfg(self):
        """Return test config."""
        cfg = dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_num=300,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                score_thr=1e-3, nms=dict(type='nms', iou_thr=0.5), max_per_img=100),
            keep_all_stages=False)
        return dict2str(cfg, tab=1)
