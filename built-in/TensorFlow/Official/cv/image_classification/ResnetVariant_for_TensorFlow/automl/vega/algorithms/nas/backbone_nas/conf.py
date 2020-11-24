# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""

from vega.search_space.search_algs import ParetoFrontConfig


class BackboneNasPolicyConfig(object):
    """BackboneNas Policy Config."""

    random_ratio = 0.2
    num_mutate = 10


class BackboneNasRangeConfig(object):
    """BackboneNas Range Config."""

    max_sample = 100
    min_sample = 10


class BackboneNasConfig(object):
    """BackboneNas Config."""

    codec = 'BackboneNasCodec'
    policy = BackboneNasPolicyConfig()
    range = BackboneNasRangeConfig()
    pareto = ParetoFrontConfig()
    objective_keys = 'accuracy'
