# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""

from .base import BaseConfig


class JDDCommonConfig(BaseConfig):
    """Default Dataset config for JDDConfig."""

    root = None


class JDDCommonTrainConfig(JDDCommonConfig):
    """Default Dataset config for JDDConfig."""

    batch_size = 16


class JDDCommonValConfig(JDDCommonConfig):
    """Default Dataset config for JDDConfig."""

    pass


class JDDCommonTestConfig(JDDCommonConfig):
    """Default Dataset config for JDDConfig."""

    pass


class JDDConfig(object):
    """Default Dataset config for JDDConfig."""

    common = JDDCommonConfig
    train = JDDCommonTrainConfig
    val = JDDCommonValConfig
    test = JDDCommonTestConfig
