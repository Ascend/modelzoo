# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Manage Loss class."""
import logging
from inspect import isclass
from functools import partial
import vega
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.config import obj2config
from ...conf import LossConfig, TrainerConfig


class Loss(object):
    """Register and call loss class."""

    config = LossConfig()

    def __init__(self):
        """Initialize."""
        # register pytorch loss as default
        loss_name = self.config.type
        self._cls = ClassFactory.get_cls(ClassType.LOSS, loss_name)

    def __call__(self):
        """Call loss cls."""
        params = obj2config(self.config).get("params", {})
        logging.debug("Call Loss. name={}, params={}".format(self._cls.__name__, params))
        try:
            if params:
                cls_obj = self._cls(**params) if isclass(self._cls) else partial(self._cls, **params)
            else:
                cls_obj = self._cls() if isclass(self._cls) else partial(self._cls)
            if vega.is_torch_backend() and TrainerConfig().cuda:
                cls_obj = cls_obj.cuda()
            return cls_obj
        except Exception as ex:
            logging.error("Failed to call Loss name={}, params={}".format(self._cls.__name__, params))
            raise ex


if vega.is_torch_backend():
    import torch.nn as torch_nn
    import timm.loss as timm_loss

    ClassFactory.register_from_package(torch_nn, ClassType.LOSS)
    ClassFactory.register_from_package(timm_loss, ClassType.LOSS)
elif vega.is_tf_backend():
    import tensorflow.losses as tf_loss

    ClassFactory.register_from_package(tf_loss, ClassType.LOSS)
