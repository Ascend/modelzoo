# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Cosine annealing lr scheduler."""
import tensorflow as tf
import math
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class CosineAnnealingLR(object):
    """Cosine anealing learning rate with warm up.

    :param base_lr: base learning rate
    :type base_lr: float
    :param T_max: maximum number of iterations
    :type T_max: int
    :param eta_min: minimum learning
    :type eta_min: int
    :param last_epoch: index of last epoch
    :type last_epoch: float
    :param warmup: whether to warm up
    :type warmup: bool
    :param warmup_epochs: warmup epochs
    :type warmup_epochs: int
    """

    def __init__(self, base_lr, T_max, eta_min=0, last_epoch=-1, warmup=True, warmup_epochs=5):
        """Init CosineAnnealingLR."""
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.warmup = warmup
        self.warmup_epochs = warmup_epochs

    def _calc_normal_lr(self):
        if self.last_epoch == 0:
            self.current_lr = self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            self.current_lr = self.current_lr + (self.base_lr - self.eta_min) * \
                (1 - math.cos(math.pi / self.T_max)) / 2
        else:
            self.current_lr = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / \
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * \
                (self.current_lr - self.eta_min) + self.eta_min

    def _calc_closed_form_lr(self):
        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + tf.math.cos(tf.constant(math.pi) * self.last_epoch / self.T_max)) / 2
        self.current_lr = tf.cast(self.current_lr, tf.float32)

    def step(self, epoch):
        """Obtain the learning rate on global steps."""
        epoch = tf.cast(epoch, tf.float32)
        self.last_epoch = epoch
        if hasattr(self, "_calc_closed_form_lr"):
            self._calc_closed_form_lr()
        else:
            self._calc_normal_lr()
        if self.warmup:
            warmup_ratio = epoch / self.warmup_epochs
            warmup_lr = self.base_lr * warmup_ratio
            self.current_lr = tf.cond(epoch < self.warmup_epochs,
                                      lambda: warmup_lr,
                                      lambda: self.current_lr)

    def get_lr(self):
        """Get current learning rate."""
        return [self.current_lr]
