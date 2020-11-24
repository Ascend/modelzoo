"""Modules for Optimizer."""

import warnings

from ..utils import dict2str
from .module import Module


class Optimizer(Module):
    """Class of potimizer."""

    id_attrs = ['optimizer', 'train_from_scratch']

    def __init__(self,
                 optimizer=dict(type='SGD', lr=0.01),
                 lr_scheduler=dict(policy='step'),
                 train_from_scratch=False,
                 *args,
                 **kwargs):
        super(Optimizer, self).__init__(*args, **kwargs)
        # optimizer_config_default = dict(
        #     grad_clip=dict(max_norm=35, norm_type=2))

        self.optimizer = optimizer.copy()
        self.lr_scheduler = lr_scheduler.copy()
        self.train_from_scratch = train_from_scratch

        self.type = self.optimizer.pop('type')
        self.policy = self.lr_scheduler.pop('policy')
        self.lr = optimizer.get('lr')

        self.Optimizer = eval(self.type)(
            train_from_scratch=train_from_scratch, **self.optimizer)
        self.LrScheduler = eval(self.policy.title() + 'Lr')(**self.lr_scheduler)
        self.optimizer.update(getattr(self.Optimizer, 'add_attrs', dict()))
        self.lr_scheduler.update(getattr(self.LrScheduler, 'add_attrs', dict()))

    @property
    def config(self):
        """Return config."""
        return dict(optimizer=self.Optimizer.config,
                    lr_config=self.LrScheduler.config)

    @classmethod
    def set_from_config(cls, config):
        """Return optimizer config."""
        config_info = super().set_from_config(config=config)
        config_info.update(dict(
            optimizer=config.optimizer,
            lr_scheduler=config.lr_config,
        ))
        return cls(**config_info)


class SGD(Module):
    """Class of sgd optimizer."""

    def __init__(self, lr, nesterov=False, paramwise_options=None,
                 train_from_scratch=False, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)
        self.lr = lr
        self.paramwise_options = paramwise_options
        self.nesterov = nesterov
        if train_from_scratch == (paramwise_options is None):
            warnings.warn(
                'Paramwise_options should be set when train from scratch.')

    @property
    def add_attrs(self):
        """Add attrs."""
        return dict(lr=self.lr, nesterov=self.nesterov,
                    paramwise_options=self.paramwise_options)

    @property
    def config(self):
        """Return config."""
        cfg = dict(type='SGD',
                   lr=self.lr,
                   nesterov=self.nesterov,
                   momentum=0.9,
                   weight_decay=0.0001,
                   paramwise_options=self.paramwise_options)
        return dict2str(cfg, tab=1)


class StepLr(Module):
    """Class of lr step."""

    def __init__(self, step=[6], warmup='linear', warmup_iters=500, warmup_ratio=0.1,
                 *args, **kwargs):
        super(StepLr, self).__init__()
        self.step = step
        self.warmup_policy = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

    @property
    def add_attrs(self):
        """Add attrs."""
        return dict(step=self.step)

    @property
    def config(self):
        """Return config."""
        lr_config = dict(policy='step',
                         warmup=self.warmup_policy,
                         warmup_iters=self.warmup_iters,
                         warmup_ratio=self.warmup_ratio,
                         step=self.step)
        return dict2str(lr_config, tab=1)


class CosineLr(Module):
    """Class of cosine lr."""

    def __init__(self, min_lr=0, warmup='linear', warmup_iters=500, warmup_ratio=0.1,
                 by_epoch=False, *args, **kwargs):
        super(CosineLr, self).__init__()
        self.min_lr = min_lr
        self.warmup_policy = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.by_epoch = by_epoch

    @property
    def config(self):
        """Return config."""
        lr_config = dict(policy='cosine',
                         warmup=self.warmup_policy,
                         warmup_iters=self.warmup_iters,
                         warmup_ratio=self.warmup_ratio,
                         min_lr=self.min_lr,
                         by_epoch=self.by_epoch)
        return dict2str(lr_config, tab=1)
