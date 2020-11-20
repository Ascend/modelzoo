"""Base class of Detector."""

from abc import ABCMeta, abstractmethod

from ..module import Module


class Detector(Module):
    """Base class of detector."""

    __metaclass__ = ABCMeta
    type = 'Detector'
    component = 'detector'

    def __init__(self, fore_part=None, *args, **kwargs):
        super().__init__(fore_part=fore_part, *args, **kwargs)

    @property
    @abstractmethod
    def train_cfg(self):
        """Get train config."""
        pass

    @property
    @abstractmethod
    def test_cfg(self):
        """Get test config."""
        pass

    @classmethod
    def set_from_config(cls, config, fore_part=None):
        """Return config info."""

        def get_samples(cfg, part):
            """Get sample result."""
            part = cfg.get(part + '_samples', None)
            if isinstance(part, list):
                part = part[0]
            try:
                return part.sampler.num
            except BaseException:
                return None
        config_info = super().set_from_config(config=config, fore_part=fore_part)
        train_cfg, test_cfg, num_stages = map(lambda x: getattr(config, x, None),
                                              ['train_cfg', 'test_cfg', 'num_stages'])
        config_info.update(num_stages=num_stages)
        config_info.update(dict(ga_samples=get_samples(train_cfg, 'ga'),
                                rpn_samples=get_samples(train_cfg, 'rpn'),
                                rcnn_samples=get_samples(train_cfg, 'rcnn')))
        config_info = {
            key: value for key,
            value in config_info.items() if value is not None}
        return cls(**config_info)
