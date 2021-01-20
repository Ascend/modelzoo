from .lr_schedule import build_schedule
from .optimizer import build_optimizer


class Solver(object):

    def __init__(self, lr_cfg, opt_cfg, device, is_distributed, mix_precision, loss_scale, checkpoint_interval):
        self.lr_schedule = build_schedule(lr_cfg)
        self.opt = build_optimizer(self.lr_schedule.lr, opt_cfg, device, is_distributed, mix_precision, loss_scale)
        self.checkpoint_interval = checkpoint_interval
        self.total_step = sum(lr_cfg.total_steps)

    def update_lr(self):
        return self.lr_schedule()


def build_solver(cfg, device, is_distributed):
    lr_cfg = cfg.lr_schedule
    opt_cfg = cfg.optimizer
    checkpoint_interval = cfg.checkpoint_interval
    mix_precision = cfg.mix_precision
    loss_scale = cfg.loss_scale

    assert device in ['npu', 'gpu', 'cpu']

    return Solver(lr_cfg, opt_cfg, device, is_distributed, mix_precision, loss_scale, checkpoint_interval)
