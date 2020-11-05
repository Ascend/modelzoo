"""Base Module for backbone."""

import torch
from mmdet.models import registry

from vega.algorithms.nas.sm_nas.mmdet_ext import *

from ...utils import profile
from ..module import Module


class Backbone(Module):
    """Base class of Backbone."""

    type = 'Backbone'
    component = 'backbone'
    quest_dict = dict(
        train_from_scratch=('optimizer', 'train_from_scratch'),
        with_neck=('detector', 'with_neck'))

    def __init__(self, with_neck, train_from_scratch,
                 fore_part=None, *args, **kwargs):
        super(Backbone, self).__init__(fore_part=fore_part, *args, **kwargs)

        self.train_from_scratch = train_from_scratch
        self.with_neck = with_neck

        if self.train_from_scratch:
            self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
            self.conv_cfg = dict(type='ConvWS')
        else:
            if self.with_neck:
                self.norm_cfg = dict(type='BN', requires_grad=True)
            else:
                self.norm_cfg = dict(type='BN', requires_grad=False)
            self.conv_cfg = None

    @property
    def checkpoint_name(self):
        """Return the ckeckpoint name."""
        return self.name

    @property
    def pretrained(self):
        """Return the pretrain model."""
        if self.train_from_scratch:
            return None
        else:
            try:
                pretrained = ''
                return pretrained
            except BaseException:
                return None

    @property
    def input_size(self):
        """Return input_size."""
        return self.quest_from(self.model['dataset'], 'img_scale')['test']

    def get_model(self):
        """Return model info."""
        model = registry.BACKBONES.module_dict.get(self.__class__.__name__)
        model = model(**self.dict_config)
        return model

    @property
    def size_info(self):
        """Return model size info."""
        if hasattr(self, '_size_info'):
            return self._size_info
        else:
            model = self.get_model()
            size_info = profile(
                model,
                self.input_size,
                style='normal',
                show_result=False)
            self._size_info = size_info
            del model
            torch.cuda.empty_cache()
        return size_info

    @property
    def flops_ratio(self):
        """Return flops ratio."""
        return 1

    @property
    def flops(self):
        """Return flops."""
        return self.size_info['FLOPs']

    @property
    def mac(self):
        """Return model mac."""
        return self.size_info['MAC']

    @property
    def params(self):
        """Return params."""
        return self.size_info['params']

    @staticmethod
    def arch_decoder(arch: str):
        """Decode the arch code."""
        pass

    @classmethod
    def set_from_arch_string(cls, arch_string, fore_part=None, **kwargs):
        """Return model params."""
        params = dict(fore_part=fore_part)
        params.update(cls.arch_decoder(arch_string))
        params.update(cls.quest_param(fore_part=fore_part, **kwargs))
        params.update(kwargs)
        return cls(**params)
