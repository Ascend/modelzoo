# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""GRDA optimizer.

"Generalized Regularized Dual Averaging" is an optimizer that can learn a small sub-network during training,
if one starts from an overparameterized dense network.
Citation: Chao, S.-K., Wang, Z., Xing, Y. and Cheng, G. (2020).
Directional pruning of deep neural networks. https://arxiv.org/abs/2006.09358.
"""

import torch
from torch.optim.optimizer import Optimizer


class gRDA(Optimizer):
    """GRDA module."""

    def __init__(self, params, lr=0.01, c=0.0, mu=0.7):
        """
        Constuct gRDA class.

        :param params:  learnable  params
        :type params: list object
        :param lr:  learning rate
        :type lr: float
        :param c:  initial sparse control constant
        :type c: float
        :param mu:  sparsity control
        :type mu: float

        :return: optimizer object
        :rtype: class
        """
        defaults = dict(lr=lr, c=c, mu=mu)
        super(gRDA, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Setstate."""
        super(gRDA, self).__setstate__(state)

    def step(self, closure=None):
        """
        Optimizer gRDA performs a single optimization step.

        :param closure:  a closure that reevaluates the model
        :type closure: callable object
        :return: loss
        :rtype: float
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                param_state = self.state[p]

                if 'iter_num' not in param_state:
                    iter_num = param_state['iter_num'] = torch.zeros(1)
                    accumulator = param_state['accumulator'] = torch.FloatTensor(p.shape).to(p.device)
                    l1_accumulation = param_state['l1_accumulation'] = torch.zeros(1)
                    accumulator.data = p.clone()

                else:
                    iter_num = param_state['iter_num']
                    accumulator = param_state['accumulator']
                    l1_accumulation = param_state['l1_accumulation']
                iter_num.add_(1)
                accumulator.data.add_(-lr, d_p)

                # l1 = c * torch.pow(torch.tensor(lr), 0.5 + mu) * torch.pow(iter_num, mu)
                l1_diff = c * torch.pow(torch.tensor(lr), mu + 0.5) * torch.pow(iter_num, mu) - c * torch.pow(
                    torch.tensor(lr), mu + 0.5) * torch.pow(iter_num - 1, mu)
                l1_accumulation += l1_diff

                new_a_l1 = torch.abs(accumulator.data) - l1_accumulation.to(p.device)
                p.data = torch.sign(accumulator.data) * new_a_l1.clamp(min=0)

        return loss
