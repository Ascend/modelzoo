# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Callbacks called at certain points of trainer."""

import vega
from vega.core.common.class_factory import ClassFactory, ClassType
from .callback import Callback


class CallbackList(object):
    """A container for managing registered Callback Objects."""

    def __init__(self, customs, disables):
        """Init Callback container."""
        self.trainer = None
        self.make_batch = None
        self.train_step = None
        self.valid_step = None
        self.model_fn = None
        self.train_input_fn = None
        self.valid_input_fn = None
        self.params = {}
        self.callbacks = self._get_callbacks(customs, disables)
        for callback in self.callbacks:
            # Get make_batch if callback has defined one
            if type(callback).make_batch != Callback.make_batch:
                if self.make_batch is None:
                    self.make_batch = callback.make_batch
                else:
                    raise ValueError("Multiple make_batch are defined!")
            # Get train_step if callback has defined one
            if type(callback).train_step != Callback.train_step:
                if self.train_step is None:
                    self.train_step = callback.train_step
                else:
                    raise ValueError("Multiple train_step are defined!")
            # Get valid_step if callback has defined one
            if type(callback).valid_step != Callback.valid_step:
                if self.valid_step is None:
                    self.valid_step = callback.valid_step
                else:
                    raise ValueError("Multiple valid_step are defined!")
            # Get model_fn if callback has defined one
            if type(callback).model_fn != Callback.model_fn:
                if self.model_fn is None:
                    self.model_fn = callback.model_fn
                else:
                    raise ValueError("Multiple model_fn are defined!")
            # Get train_input_fn if callback has define one
            if type(callback).train_input_fn != Callback.train_input_fn:
                if self.train_input_fn is None:
                    self.train_input_fn = callback.train_input_fn
                else:
                    raise ValueError("Multiple train_input_fn are defined!")
            # Get valid_input_fn if callback has define one
            if type(callback).valid_input_fn != Callback.valid_input_fn:
                if self.valid_input_fn is None:
                    self.valid_input_fn = callback.valid_input_fn
                else:
                    raise ValueError("Multiple valid_input_fn are defined!")

    def _get_callbacks(self, customs, disables):
        defaults = []
        if vega.is_torch_backend():
            defaults = ["ModelStatistics", "MetricsEvaluator", "ModelCheckpoint", "PerformanceSaver",
                        "LearningRateScheduler", "ProgressLogger", "ReportCallback"]
        elif vega.is_tf_backend():
            # defaults = ["ModelStatistics", "MetricsEvaluator", "PerformanceSaver",
            #             "ProgressLogger", "ReportCallback"]
            defaults = ["ModelStatistics", "MetricsEvaluator", "PerformanceSaver",
                        "ProgressLogger"]
        custom_disables = []
        disables = disables if disables else []
        customs = customs if customs else []
        if customs:
            if isinstance(customs, str):
                customs = [customs]
            for customs_name in customs:
                callback_class = ClassFactory.get_cls(ClassType.CALLBACK, customs_name)
                # Sort the callbacks
                if hasattr(callback_class, "disable_callbacks"):
                    _disables = callback_class.disable_callbacks
                    if not isinstance(_disables, list):
                        _disables = [_disables]
                    custom_disables = _disables
        callbacks = set([_cls for _cls in defaults + customs if _cls not in disables + custom_disables])
        callbacks = [ClassFactory.get_cls(ClassType.CALLBACK, _cls)() for _cls in callbacks]
        callbacks = sorted(callbacks, key=lambda callback: callback.priority)
        return callbacks

    def _set_params(self, trainer):
        params = {
            'epochs': trainer.epochs,
            'is_chief': trainer.is_chief,
            'use_cuda': trainer.use_cuda,
            'do_validation': trainer.do_validation,
            'is_detection_trainer': trainer.config.is_detection_trainer
        }
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        """Set the trainer object for callback container."""
        self.trainer = trainer
        self._set_params(trainer)
        # Replace the default make_batch of Trainer
        if self.make_batch is not None:
            self.trainer.make_batch = self.make_batch
        # Replace the default train_step of Trainer
        if self.train_step is not None:
            self.trainer.train_step = self.train_step
        # Replace the default train_step of Trainer
        if self.valid_step is not None:
            self.trainer.valid_step = self.valid_step
        # Replace the default model_fn of Trainer
        if self.model_fn is not None:
            self.trainer.model_fn = self.model_fn
            self.trainer._init_tf_estimator()
        # Replace the default train_input_fn of Trainer
        if self.train_input_fn is not None:
            self.trainer.train_input_fn = self.train_input_fn
        # Replace the default valid_input_fn of Trainer
        if self.valid_input_fn is not None:
            self.trainer.valid_input_fn = self.valid_input_fn

        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def before_train(self, logs=None):
        """Call before_train of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_train(logs)

    def before_epoch(self, epoch, logs=None):
        """Call before_epoch of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_epoch(epoch, logs)

    def before_train_step(self, batch_index, logs=None):
        """Call before_train_step of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_train_step(batch_index, logs)

    def after_train_step(self, batch_index, logs=None):
        """Call after_train_step of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_train_step(batch_index, logs)

    def after_epoch(self, epoch, logs=None):
        """Call after_epoch of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_epoch(epoch, logs)

    def after_train(self, logs=None):
        """Call after_train of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_train(logs)

    def before_valid(self, logs=None):
        """Call before_valid of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_valid(logs)

    def before_valid_step(self, batch_index, logs=None):
        """Call before_valid_step of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.before_valid_step(batch_index, logs)

    def after_valid_step(self, batch_index, logs=None):
        """Call after_valid_step of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_valid_step(batch_index, logs)

    def after_valid(self, logs=None):
        """Call after_valid of the managed callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.after_valid(logs)
