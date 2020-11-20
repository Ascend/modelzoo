# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of super solution task."""
import math
import torch
from vega.core.metrics.pytorch.metrics import MetricBase
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.METRIC)
class JDDTrainerPSNRMetric(MetricBase):
    """Calculate PSNR metric between output and target."""

    def __init__(self):
        """Init psnr metric for JDDTrainer."""
        self.sum = 0.
        self.pfm = 0.
        self.data_num = 0

    def reset(self):
        """Reset stored values after each epoch for new evaluation."""
        self.sum = 0.
        self.pfm = 0.
        self.data_num = 0

    def __call__(self, output, target, *args, **kwargs):
        """Calculate JDDTrainer PSNR metric."""
        res = 0.0
        output_frames = output.permute(0, 2, 3, 1, 4)
        for ind in range(output.shape[4]):
            output_frame = torch.squeeze(output_frames[0, :, :, :, ind])
            target_frame = torch.squeeze(target[0, :, :, :, ind])
            res += self.calc_psnr(output_frame, target_frame, psnr_channel='C')
        n = output.size(0)
        self.data_num += n
        self.sum = self.sum + res * n
        self.pfm = self.sum / self.data_num
        return res

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm

    def calc_psnr(self, predict, target, psnr_channel):
        """Calculate the psnr between predict and groundtruth.

        :param predict:  predict images of model
        :type predict: array
        :param target:  ground truth of images
        :type target: array
        :param psnr_channel： type of psnr calculation
        :type psnr_channel: str
        :return: psnr
        :rtype: float
        """
        if target.nelement() == 1:
            return 0
        shave = 1
        diff = predict.clamp(0, 1) - target.clamp(0, 1)
        if diff.size(1) > 1:
            if psnr_channel == 'Y':
                gray_coeffs = [25.064, 129.057, 65.738]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        valid = diff[..., shave:-shave, shave:-shave]
        mse = valid.pow(2).mean()
        return -10 * math.log10(mse)
