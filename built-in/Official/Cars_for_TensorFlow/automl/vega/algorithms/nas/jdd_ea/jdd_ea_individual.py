# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The Individual definition of JDD EA algorithm."""
from bisect import bisect_right
from random import random, sample
import numpy as np
from .conf import JDDSearchRangeConfig


class JDDIndividual(object):
    """Construct JDD individual class.

    :param net_info: the information list of the network
    :type net_info: list
    :param search_config: search algorithm configuration
    :type search_config: dictionary
    :param search_space: search space configuration
    :type search_space: dictionary
    """

    config = JDDSearchRangeConfig()

    def __init__(self, net_info, search_config, search_space):
        """Construct initialize method."""
        self.net_info = net_info
        # TODO search_config not used.
        self.search_config = search_config
        self.min_res_st = self.config.min_res_start
        self.min_res_end = self.config.min_res_end
        self.node_num = self.config.node_num
        self.max_resolutions = self.config.max_resolutions
        model_type = search_space.modules[0]
        self.base_channel = search_space[model_type].basic_channel

        self.gene = np.zeros((self.node_num + self.max_resolutions, 2)).astype(int)
        self.init_gene()
        self.active_num = self.active_node_num()
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()
        self.fitness = 0

    def init_gene(self):
        """Initialize the gene randomly."""
        for gene_ind in range(self.node_num):
            type_prob = self.net_info.func_prob
            self.gene[gene_ind][1] = bisect_right(type_prob, random())
            self.gene[gene_ind][0] = np.random.randint(2)

        self.gene[self.node_num:self.gene.shape[0], :] = -1
        max_res_rest = (self.active_node_num() - self.min_res_st - self.min_res_end + 1) // 2
        if max_res_rest > 0:
            reso_num = np.random.randint(min(self.max_resolutions, max_res_rest))
            locations = range(self.min_res_st, self.active_node_num() - self.min_res_end + 1)
            res_loc = np.sort(np.asarray(sample(locations, reso_num * 2)))
            self.gene[self.node_num:self.node_num + reso_num, :] = \
                np.transpose(np.reshape(res_loc, (2, reso_num)), (1, 0))

    def active_node_num(self):
        """Get the number of active nodes.

        :return: the number of active nodes
        :rtype: int
        """
        active_num = np.sum(np.asarray(self.gene[0:self.node_num, 0]))
        return active_num

    def active_net_list(self):
        """Get the list of active nodes gene.

        :return: list of active nodes gene
        :rtype: list
        """
        net_list = []
        Sp2De_loc = list(self.gene[self.node_num:self.gene.shape[0], 0])
        De2Sp_loc = list(self.gene[self.node_num:self.gene.shape[0], 1])
        active_node_counter = 0
        out_channel = self.base_channel
        depth_channel = []
        conc_before = 0
        for ind in range(self.node_num):
            using, n = self.gene[ind]
            if using:
                type_str = self.net_info.func_type[n]
                if active_node_counter in Sp2De_loc:
                    net_list.append('Space-to-Depth')
                    depth_channel.append(out_channel)
                    out_channel = out_channel * 4
                if active_node_counter in De2Sp_loc:
                    net_list.append('Depth-to-Space')
                    conc_before = depth_channel.pop()
                    out_channel = out_channel // 4
                net_list.append(type_str[0] + '_' + str(out_channel + conc_before) + type_str[1:])
                conc_before = 0
                out_channel = int(type_str.split('_')[2])
                active_node_counter += 1
        return net_list

    def network_parameter(self):
        """Get the number of parameters in network.

        :return: number of parameters in network
        :rtype: int
        """
        model_info = self.active_net_list()
        model_para = 5 * self.base_channel * 9
        channel_out = self.base_channel
        for i in range(len(model_info)):
            name = model_info[i]
            key = name.split('_')
            b_type = key[0]
            if b_type == 'R':
                b_channel_in = int(key[1])
                b_channel_inter = int(key[2])
                b_channel_out = int(key[3])
                model_para += 2 * 9 * b_channel_inter * b_channel_out
                if b_channel_in != b_channel_out:
                    model_para += 9 * b_channel_in * b_channel_out
                channel_out = b_channel_out
            elif b_type == 'Depth-to-Space':
                model_para += 9 * channel_out * channel_out
        model_para += 9 * 2 * channel_out * 2 * self.base_channel + 9 * self.base_channel * 3 / 2
        return model_para

    def network_flops(self):
        """Get the FLOPS of network.

        :return: the FLOPS of network
        :rtype: float
        """
        model_info = self.active_net_list()
        feature_size = np.asarray([1024, 1824])
        model_flops = 5 * self.base_channel * 9 * feature_size[0] * feature_size[1]
        channel_out = self.base_channel
        for i in range(len(model_info)):
            name = model_info[i]
            key = name.split('_')
            b_type = key[0]
            if b_type == 'R':
                b_channel_in = int(key[1])
                b_channel_inter = int(key[2])
                b_channel_out = int(key[3])
                model_flops += 2 * 9 * b_channel_inter * b_channel_out * feature_size[0] * feature_size[1]
                if b_channel_in != b_channel_out:
                    model_flops += 9 * b_channel_in * b_channel_out * feature_size[0] * feature_size[1]
                channel_out = b_channel_out
            elif b_type == 'Space-to-Depth':
                feature_size = feature_size / 2
            elif b_type == 'Depth-to-Space':
                model_flops += 9 * channel_out * channel_out * feature_size[0] * feature_size[1]
                feature_size = feature_size * 2
        model_flops += 9 * 2 * channel_out * 2 * self.base_channel * feature_size[0] * feature_size[1]
        feature_size = feature_size * 2
        model_flops += 9 * self.base_channel * 3 * feature_size[0] * feature_size[1] / 2
        return model_flops * 2

    def copy(self, source):
        """Copy the individual from another individual.

        :param source: the source Individual
        :type source: Individual Objects
        """
        self.net_info = source.net_info
        self.search_config = source.search_config
        self.min_res_st = source.min_res_st
        self.min_res_end = source.min_res_end
        self.node_num = source.node_num
        self.max_resolutions = source.max_resolutions
        self.base_channel = source.base_channel
        self.gene = source.gene.copy()
        self.active_num = source.active_num
        self.active_net = source.active_net
        self.parameter = source.parameter
        self.fitness = source.fitness
        self.flops = source.flops

    def update_fitness(self, performance):
        """Update fitness of one individual.

        :param performance: the score of the evalution
        :type performance: float
        """
        self.fitness = performance

    def update_gene(self, new_gene):
        """Update the gene of individual.

        :param new_gene: new gene
        :type new_gene: list
        """
        self.gene = new_gene.copy()
        self.active_num = self.active_node_num()
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()

    def mutation_resolutions(self):
        """Mutate the resolution of network."""
        self.gene[self.node_num:self.gene.shape[0], :] = -1
        max_res_rest = (self.active_node_num() - self.min_res_st - self.min_res_end + 1) // 2
        if max_res_rest > 0:
            reso_num = np.random.randint(min(self.max_resolutions, max_res_rest))
            locations = range(self.min_res_st, self.active_node_num() - self.min_res_end + 1)
            res_loc = np.sort(np.asarray(sample(locations, reso_num * 2)))
            self.gene[self.node_num:self.node_num + reso_num, :] = \
                np.transpose(np.reshape(res_loc, (2, reso_num)), (1, 0))
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()

    def mutation_using(self, mutation_rate=0.05):
        """Mutate the using gene of individual.

        :param mutation_rate: the prosibility to mutate, defaults to 0.05
        :type mutation_rate: float
        """
        for node_ind in range(self.node_num):
            if np.random.rand() < mutation_rate:
                self.gene[node_ind][0] = 1 - self.gene[node_ind][0]
        self.active_num = self.active_node_num()
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()

    def mutation_node(self, mutation_rate=0.05):
        """Mutate the active node type of individual.

        :param mutation_rate: the prosibility to mutate, defaults to 0.05
        :type mutation_rate: float
        """
        for node_ind in range(self.node_num):
            if self.gene[node_ind][0] and np.random.rand() < mutation_rate:
                type_prob = self.net_info.func_prob
                self.gene[node_ind][1] = bisect_right(type_prob, random())
        self.gene[self.node_num:self.gene.shape[0], :] = -1
        max_res_rest = (self.active_node_num() - self.min_res_st - self.min_res_end + 1) // 2
        if max_res_rest > 0:
            reso_num = np.random.randint(min(self.max_resolutions, max_res_rest))
            locations = range(self.min_res_st, self.active_node_num() - self.min_res_end + 1)
            res_loc = np.sort(np.asarray(sample(locations, reso_num * 2)))
            self.gene[self.node_num:self.node_num + reso_num, :] = \
                np.transpose(np.reshape(res_loc, (2, reso_num)), (1, 0))
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()
