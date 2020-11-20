# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The Autogate model."""
import logging
import torch
import copy
from .deepfm import DeepFactorizationMachineModel
from .fis.layers import NormalizedWeightedFMLayer
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


@NetworkFactory.register(NetTypes.CUSTOM)
class AutoGateModel(DeepFactorizationMachineModel):
    """Automatic Feature Interaction Selection (FIS) For DeepFM.

    :param input_dim: feature space of dataset
    :type input_dim: int
    :param input_dim4lookup: feature number in `feature_id`, usually equals to number of non-zero features
    :type input_dim4lookup: int
    :param embed_dim: length of each feature's latent vector(embedding vector)
    :type embed_dim: int
    :param hidden_dims: width of each hidden layer, from bottom to top
    :type hidden_dims: list of int
    :param dropout_prob: dropout probability of all hidden layer
    :type dropout_prob: float
    :param alpha_init_mean: mean of initialization value for `alpha`, defaults to 0.5
    :type alpha_init_mean: float, optional
    :param alpha_init_radius: radius of initialization range for `alpha`, defaults to 0.001
    :type alpha_init_radius: float, optional
    :param alpha_activation:  activation function for `alpha`, one of 'tanh' or 'identity', defaults to 'tanh'
    :type alpha_activation: str, optional
    :param batch_norm: applies batch normalization before activation, defaults to False
    :type batch_norm: bool, optional
    :param layer_norm: applies layer normalization before activation, defaults to False
    :type layer_norm: bool, optional
    :param selected_pairs: use selected feature pairs only(denoted by their index in given arrangement), defaults to []
    :type selected_pairs: list of tuple, optional
    """

    def __init__(self, net_desc):
        """
        Construct the AutoGateModel class.

        :param net_desc: config of the structure
        :type net_desc: class object
        :return: return AutoGateModel class
        :rtype: class object
        """
        super().__init__(net_desc)
        self.desc = copy.deepcopy(net_desc)

        # override fm module in parent class
        self.fm = NormalizedWeightedFMLayer(input_dim4lookup=net_desc.input_dim4lookup,
                                            alpha_init_mean=net_desc.alpha_init_mean,
                                            alpha_init_radius=1e-4,
                                            alpha_activation=net_desc.alpha_activation,
                                            selected_pairs=net_desc.selected_pairs)

        self.l1_cover_params = [self.fm._alpha]
        self.l2_cover_params = [self.fm._alpha]

        self.structure_params = set([self.fm._alpha])

    def get_feature_interaction_score(self):
        """
        Retrieve feature interaction score for each pair.

        :return: feature interaction score
        :rtype: dict
        """
        feature_interaction_score = {}
        feat_i, feat_j = self.fm.pair_indexes.tolist()
        for idx, pair in enumerate(zip(feat_i, feat_j)):
            score = self.fm._alpha[idx].item()
            logging.debug("pair {} => importance score {}".format(pair, score))
            feature_interaction_score[pair] = score
        return feature_interaction_score
