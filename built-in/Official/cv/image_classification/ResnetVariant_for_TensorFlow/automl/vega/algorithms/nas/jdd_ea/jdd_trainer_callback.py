# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for JDD."""
import torch
import torch.nn.functional as F
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class JDDTrainerCallback(Callback):
    """Callback class for JDDTrainer."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
        loss = self.trainer.loss
        grad_loss = GradientLoss()
        if self.config.cuda:
            grad_loss = grad_loss.cuda()
        # User can replace the trainer.loss and other parts
        # in before_train(), but cannot do this when train begans
        self.trainer.loss = JDDTrainerLoss(loss, grad_loss)


class JDDTrainerLoss(torch.nn.Module):
    """Define the image gradient loss."""

    def __init__(self, loss, grad_loss):
        """Initialize the jdd trainer loss."""
        super(JDDTrainerLoss, self).__init__()
        self.loss = loss
        self.grad_loss = grad_loss

    def forward(self, output, target):
        """Calculate the gradients loss of predict and groundtruth."""
        loss = self.grad_loss(output, target) + self.loss(output, target)
        return loss


class GradientLoss(torch.nn.Module):
    """Define the image gradient loss."""

    def __init__(self):
        """Initialize the grandient loss."""
        super(GradientLoss, self).__init__()

    def forward(self, predict, groundtruth):
        """Calculate the gradients loss of predict and groundtruth.

        :param predict:  predict images of model
        :type predict: array
        :param groundtruth:  ground truth of images
        :type groundtruth: array
        :return: gradients loss of predict and groundtruth
        :rtype: float
        """
        dx_1, dy_1 = self.image_gradient(predict)
        dx_2, dy_2 = self.image_gradient(groundtruth)
        loss = 0.5 * (torch.mean(torch.abs(dx_1 - dx_2)) +
                      torch.mean(torch.abs(dy_1 - dy_2)))
        return loss

    def image_gradient(self, images):
        """Calculate the gradients of images.

        :param images: images
        :type images: array
        :return: gradients of images
        :rtype: float
        """
        images = images.permute(0, 4, 1, 2, 3)
        _l = images
        r = F.pad(images, [0, 1, 0, 0])[:, :, :, :, 1:]
        t = images
        b = F.pad(images, [0, 0, 0, 1])[:, :, :, 1:, :]
        dx, dy = torch.abs(r - _l), torch.abs(b - t)
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0
        return dx.permute(0, 2, 3, 4, 1), dy.permute(0, 2, 3, 4, 1)
