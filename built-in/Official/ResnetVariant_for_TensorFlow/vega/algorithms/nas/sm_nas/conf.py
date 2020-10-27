# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""


class SMNasConfig(object):
    """SR Config."""

    max_sample = 2
    min_sample = 1
    pareto = {}
    train_setting = {}
    search_space = {}
    data_setting = {}

    random_ratio = 0.2
    num_mutate = 10
    sample_base = True
    sample_setting = {}
