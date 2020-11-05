# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Callbacks called at certain points of trainer."""

from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class Callback(object):
    """Abstract class for buiding new callbacks."""

    priority = 100

    def __init__(self):
        """Init callback object."""
        self.trainer = None
        self.params = None

    def set_trainer(self, trainer):
        """Set trainer object for current callback."""
        self.trainer = trainer

    def set_params(self, params):
        """Set parameters for current callback."""
        self.params = params

    def before_train(self, logs=None):
        """Be called before the training process.

        Subclasses should override this for their own purposes
        """

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoch during the training process.

        Subclasses should override this for their own purposes
        """

    def before_train_step(self, batch_index, logs=None):
        """Be called before each batch training.

        Subclasses should override this for their own purposes
        """

    def make_batch(self, batch):
        """Be called on each batch training.

        Subclasses should override this for their own purposes
        This will replace the default make_batch function in the
        trainer.
        """

    def train_step(self, batch):
        """Be called on each batch training.

        Subclasses should override this for their own purposes
        This will replace the default train_step function in the
        trainer.
        """

    def valid_step(self, batch):
        """Be called on each batch validing.

        Subclasses should override this for their own purposes
        This will replace the default valid_step function in the
        valider.
        """

    def model_fn(self, features, labels, mode):
        """Be called on each epoch in tf backend.

        Subclasses should override this for their own purposes
        This will replace the default model_fn function in the
        trainer.
        """

    def train_input_fn(self):
        """Be called on each epoch in tf backend.

        Subclasses should override this for their own purposes
        This will replace the default train_input_fn function in the
        trainer.
        """

    def valid_input_fn(self):
        """Be called on each epoch in tf backend.

        Subclasses should override this for their own purposes
        This will replace the default valid_input_fn function in the
        trainer.
        """

    def after_train_step(self, batch_index, logs=None):
        """Be called after each batch training.

        Subclasses should override this for their own purposes
        """

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch during the training process.

        Subclasses should override this for their own purposes
        """

    def after_train(self, logs=None):
        """Be called after the training process.

        Subclasses should override this for their own purposes
        """

    def before_valid(self, logs=None):
        """Be called before the validation.

        Subclasses should override this for their own purposes

        Also called before a validation batch during the train function
        """

    def before_valid_step(self, batch_index, logs=None):
        """Be called before a batch evaluation or validation.

        Subclasses should override this for their own purposes

        Also called before a validation batch during the train function
        if validition is requied
        """

    def after_valid_step(self, batch_index, logs=None):
        """Be called after a batch validation.

        Subclasses should override this for their own purposes

        Also called after a validation batch during the train function,
        if validition is requied
        """

    def after_valid(self, logs=None):
        """Be called after the validation.

        Subclasses should override this for their own purposes
        """
