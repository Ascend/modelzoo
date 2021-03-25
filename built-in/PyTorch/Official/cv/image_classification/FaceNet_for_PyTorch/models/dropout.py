#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/19 10：20
# Author   :
# @Site    :
# @File    :dropout.py

import torch
import torch.nn as nn
import numpy as np


class DroupoutV2(nn.Module):
    def __init__(self, p=0.5, inplace=False, max_seed=2 ** 10 - 1):
        super(DroupoutV2, self).__init__()
        self.p = p
        self.seed = torch.from_numpy(np.random.uniform(1, max_seed, size=(32 * 1024 * 12,)).astype(np.float32))
        self.checked = False

    def check_self(self, x):
        """Check device equipment between tensors.
        """
        if self.seed.device == x.device:
            self.checked = True
            return

        self.seed = self.seed.to(x.device)

    def forward(self, x):
        if not self.training:
            return x

        if not self.checked:
            self.check_self(x)

        x, mask, _ = torch.npu_dropoutV2(x, self.seed, p=self.p)
        return x
