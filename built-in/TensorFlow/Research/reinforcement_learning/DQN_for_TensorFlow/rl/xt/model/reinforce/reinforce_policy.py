"""
@Author: Huaqing Wang
@license : Copyright(C), Huawei
"""
import torch.nn as nn
import torch.nn.functional as F

from xt.framework.register import Registers


@Registers.model.register
class ReinforcePolicy(nn.Module):
    """docstring for ReinforcePolicy."""
    def __init__(self, model_info):
        super(ReinforcePolicy, self).__init__()
        num_inputs = model_info["state_dim"]
        action_space = model_info["action_dim"]
        hidden_size = model_info["hidden_size"]
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        layer = F.relu(self.linear1(inputs))
        action_scores = self.linear2(layer)
        return F.softmax(action_scores)
