"""ROIExtractor Module for RoIAlign."""

from collections import OrderedDict

from .module import Module
from .get_space import roi_extractor
from ..utils import dict2str


class ROIExtractor(Module):
    """Base class of roi extractor."""

    type = 'ROIExtractor'
    component = 'roi_extractor'
    quest_dict = dict(with_neck=('detector', 'with_neck'),
                      out_channels=('rpn_head', 'in_channels'))

    def __init__(self, with_neck, out_channels, featmap_strides,
                 fore_part=None, *args, **kwargs):
        super(ROIExtractor, self).__init__(fore_part=fore_part, *args, **kwargs)
        self.with_neck = with_neck
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

    @classmethod
    def quest_param(cls, fore_part=None, **kwargs):
        """Return quest param."""
        params = super().quest_param(fore_part=fore_part, **kwargs)
        with_neck = params.get('with_neck')
        if with_neck:
            featmap_strides = cls.quest_from(fore_part['neck'], 'fea_strides')
        else:
            featmap_strides = cls.quest_from(
                fore_part['backbone'], 'out_strides')
        params.update(featmap_strides=featmap_strides)
        return params


@roi_extractor.register_space
class RoIAlign(ROIExtractor):
    """Class of roi align."""

    type = 'RoIAlign'
    module_space = {'roi_extractor': 'RoIAlign'}
    id_attrs = ['out_size']

    def __init__(self, **kwargs):
        super(RoIAlign, self).__init__(**kwargs)
        self.out_size = 14 if not self.with_neck else 7
        self.roi_layer = OrderedDict(
            type='RoIAlign',
            out_size=self.out_size,
            sample_num=2)

    @property
    def config(self):
        """Return config."""
        config = OrderedDict(
            type='SingleRoIExtractor',
            roi_layer=self.roi_layer,
            out_channels=self.out_channels,
            featmap_strides=self.featmap_strides)
        return dict2str(config, tab=2)
