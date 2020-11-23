"""Network Description for mmdet model."""

import os

import mmcv
from mmdet.models import build_detector

from vega.search_space.networks import NetworkDesc


class MMDetDesc(NetworkDesc):
    """Description for mmdet model."""

    def __init__(self, sample):
        self.cost = sample['cost']
        self.desc = sample['desc']
        assert(isinstance(self.desc, str))

    def to_model(self):
        """Build pytorch model."""
        with open('tmp.py', 'w') as f:
            f.write(self.desc)
        cfg = mmcv.Config.fromfile('tmp.py')
        os.remove('tmp.py')

        pretrain_file = cfg.model.pretrained
        if not os.path.exists(pretrain_file):
            cfg.model.pretrained = None

        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        model.desc = self.desc
        model.cost = self.cost
        return model
