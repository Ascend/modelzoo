"""BBoxHead Modules for CascadeFCBBoxHead."""

from collections import OrderedDict

from ...utils import dict2str
from ..get_space import bbox_head
from .bbox_head import BBoxHead


@bbox_head.register_space
class CascadeFCBBoxHead(BBoxHead):
    """Class of Cascade FC Bounding Box Head."""

    type = 'CascadeFCBBoxHead'
    module_space = {'bbox_head': 'CascadeFCBBoxHead'}
    id_attrs = ['num_fcs', 'fc_out_channels', 'target_means', 'target_stds']
    base_target_means = [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]
    base_target_stds = [[0.1, 0.1, 0.2, 0.2], [
        0.05, 0.05, 0.1, 0.1], [0.033, 0.033, 0.067, 0.067]]

    def __init__(self,
                 num_fcs=2,
                 fc_out_channels=1024,
                 reg_class_agnostic=True,
                 target_means=None,
                 target_stds=None,
                 loss_cls=OrderedDict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=OrderedDict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 **kwargs):
        super(CascadeFCBBoxHead, self).__init__(reg_class_agnostic=reg_class_agnostic,
                                                loss_cls=loss_cls,
                                                loss_bbox=loss_bbox,
                                                **kwargs)
        self.num_fcs = num_fcs
        self.fc_out_channels = fc_out_channels

        self.has_ga_rpn = 'GA' in self.quest_from(
            self.model['rpn_head'], 'name')
        self.num_stages = self.quest_from(self.model['detector'], 'num_stages')
        # change target_std and target_means according to whether use ga rpn
        if target_means is None:
            self.target_means = self.base_target_means[:self.num_stages]
        else:
            self.target_means = target_means

        if target_stds is None:
            self.target_stds = [list(map(lambda x: round(x * 3 / 4, 3), stds)) for stds in
                                self.base_target_stds[:self.num_stages]] \
                if self.has_ga_rpn else self.base_target_stds[:self.num_stages]
        else:
            self.target_stds = target_stds

    def __str__(self):
        """Get arch code."""
        return '{}(n={})'.format(self.type, self.num_stages)

    @property
    def config(self):
        """Return config of cascade head."""
        config = ''
        for i in range(self.num_stages):
            if self.with_neck:
                config += dict2str(OrderedDict(
                    type='SharedFCBBoxHead',
                    num_fcs=self.num_fcs,
                    in_channels=self.in_channels,
                    fc_out_channels=self.fc_out_channels,
                    roi_feat_size=self.roi_feat_size,
                    num_classes=self.num_classes,
                    target_means=self.target_means[i],
                    target_stds=self.target_stds[i],
                    reg_class_agnostic=self.reg_class_agnostic,
                    loss_cls=self.loss_cls,
                    loss_bbox=self.loss_bbox,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                ), tab=2, format_first_line=True) + ',\n'
            else:
                config += dict2str(OrderedDict(
                    type='BBoxHead',
                    with_avg_pool=self.with_avg_pool,
                    roi_feat_size=self.roi_feat_size,
                    in_channels=self.in_channels,
                    num_classes=self.num_classes,
                    target_means=self.target_means[i],
                    target_stds=self.target_stds[i],
                    reg_class_agnostic=self.reg_class_agnostic,
                    loss_cls=self.loss_cls,
                    loss_bbox=self.loss_bbox
                ), tab=2, format_first_line=True) + ',\n'
        config = config[:-len(',\n')]
        return '[\n{}]'.format(config)

    @classmethod
    def set_from_config(cls, config, fore_part=None):
        """Set from config."""
        if not isinstance(config, (list)):
            raise TypeError(
                "{}: 'config' must be a list, but get a {}.".format(
                    cls.__name__, type(config)))
        config_info = super().set_from_config(config=config, fore_part=fore_part)
        target_means = []
        target_stds = []
        for stage in range(len(config)):
            target_means.append(config[stage].get('target_means'))
            target_stds.append(config[stage].get('target_stds'))
        config_info.update(target_means=target_means, target_stds=target_stds)
        return cls(**config_info)
