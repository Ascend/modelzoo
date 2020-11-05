# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Encode and decode the model config. for JDD."""
from copy import deepcopy

import numpy as np

from vega.search_space.codec import Codec
from vega.core.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.CODEC)
class JDDCodec(Codec):
    """Codec of the JDD search space."""

    def __init__(self, search_space=None, **kwargs):
        """Construct the SRCodec class.

        :param codec_name: name of the codec
        :type codec_name: str
        :param search_space: Search space of the codec
        :type search_space: dictionary
        "S_" means that the shrink RDB blcock with 1x1 convolution .
        "G_" means that the RDB block with channel shuffle and group convolution.
        "C_" means that the contextual RDB block with recursive layer.
        first number: the number of convolutional layers in a block
        second number: the growth rate of dense connected in a block
        third number: the number of output channel in a block
        """
        super(JDDCodec, self).__init__(search_space, **kwargs)
        self.func_type, self.func_prob = self.get_choices()
        self.func_type_num = len(self.func_type)

    def get_choices(self):
        """Get search space information.

        :return: the configs of the blocks
        :rtype: lists
        """
        channel_types = ['16', '32', '48']
        channel_prob = [1, 0.5, 0.2]
        block_types = ['R']
        block_prob = [1]
        model_type = self.search_space['modules'][0]
        channel_types = self.search_space[model_type]['channel_types']
        channel_prob = self.search_space[model_type]['channel_prob']
        block_types = self.search_space[model_type]['block_types']
        block_prob = self.search_space[model_type]['block_prob']

        func_type = []
        func_prob = []
        for b_id in range(len(block_types)):
            for chin_id in range(len(channel_types)):
                for chout_id in range(len(channel_types)):
                    func_type.append(block_types[b_id] + '_' + channel_types[chin_id] + '_' + channel_types[chout_id])
                    func_prob.append(block_prob[b_id] * channel_prob[chin_id] * channel_prob[chout_id])
        func_prob = np.cumsum(np.asarray(func_prob) / sum(func_prob))
        return func_type, func_prob

    def decode(self, indiv):
        """Add the network structure to config.

        :param indiv: an individual which contains network architecture code
        :type indiv: individual class
        :return: config of model structure
        :rtype: dictionary
        """
        indiv_cfg = deepcopy(self.search_space)
        model = indiv_cfg['modules'][0]
        indiv_cfg[model]['code'] = indiv.gene.tolist()
        indiv_cfg[model]['architecture'] = indiv.active_net_list()
        return indiv_cfg
