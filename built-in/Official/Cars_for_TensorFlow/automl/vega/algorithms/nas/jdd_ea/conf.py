# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""


class JDDSearchPolicyConfig(object):
    """JDD Search Policy Config."""

    num_generation = 20
    num_individual = 8
    num_elitism = 4
    mutation_rate = 0.05


class JDDSearchRangeConfig(object):
    """JDD Search Policy Config."""

    node_num = 16
    min_active = 6
    min_res_start = 1
    min_res_end = 1
    min_flops = 0
    max_flops = 160000000000
    max_resolutions = 5


class JDDSearchConfig(object):
    """JDD Search Config."""

    codec = 'JDDCodec'
    policy = JDDSearchPolicyConfig
    range = JDDSearchRangeConfig
    objective_keys = 'SRMetric'
