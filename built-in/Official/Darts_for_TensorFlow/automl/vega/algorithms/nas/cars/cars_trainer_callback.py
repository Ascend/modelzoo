# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""CARS trainer."""
import numpy as np
from collections import namedtuple
import logging
import vega
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space import SearchSpace
from vega.search_space.search_algs import SearchAlgorithm
from vega.core.trainer.callbacks import Callback
from vega.core.trainer.modules.optimizer import Optimizer
from vega.core.trainer.modules.lr_schedulers import LrScheduler
from vega.core.trainer.modules.losses import Loss


if vega.is_torch_backend():
    import torch
    import torch.nn as nn
    import torch.backends.cudnn as cudnn
elif vega.is_tf_backend():
    import tensorflow as tf

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


@ClassFactory.register(ClassType.CALLBACK)
class CARSTrainerCallback(Callback):
    """A special callback for CARSTrainer."""

    disable_callbacks = ["ModelStatistics"]

    def __init__(self):
        super(CARSTrainerCallback, self).__init__()
        self.alg_policy = None

    def before_train(self, logs=None):
        """Be called before the training process."""
        # Use zero valid_interval to supress default valid step
        self.trainer.valid_interval = 0
        self.trainer.config.report_on_epoch = True
        if vega.is_torch_backend():
            cudnn.benchmark = True
            cudnn.enabled = True
        self.search_alg = SearchAlgorithm(SearchSpace().search_space)
        self.alg_policy = self.search_alg.config.policy
        self.set_algorithm_model(self.trainer.model)
        # setup alphas
        n_individual = self.alg_policy.num_individual
        self.alphas = np.stack([self.search_alg.random_sample_path()
                                for i in range(n_individual)], axis=0)
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        self.epoch = epoch

    def train_step(self, batch):
        """Replace the default train_step function."""
        self.trainer.model.train()
        input, target = batch
        self.trainer.optimizer.zero_grad()
        alphas = torch.from_numpy(self.alphas).cuda()
        for j in range(self.alg_policy.num_individual_per_iter):
            i = np.random.randint(0, self.alg_policy.num_individual, 1)[0]
            if self.epoch < self.alg_policy.warmup:
                alpha = torch.from_numpy(self.search_alg.random_sample_path()).cuda()
                # logits = self.trainer.model.forward_random(input)
            else:
                alpha = alphas[i]
            logits = self.trainer.model(input, alpha)
            loss = self.trainer.loss(logits, target)
            loss.backward(retain_graph=True)
            if self.epoch < self.alg_policy.warmup:
                break
        nn.utils.clip_grad_norm_(
            self.trainer.model.parameters(), self.trainer.config.grad_clip)
        self.trainer.optimizer.step()
        return {'loss': loss.item(),
                'train_batch_output': logits}

    def model_fn(self, features, labels, mode):
        """Define cars model_fn used by TensorFlow Estimator."""
        logging.info('Cars model function action')
        self.trainer.loss = Loss()()

        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            epoch = tf.cast(global_step, tf.float32) / tf.cast(len(self.trainer.train_loader), tf.float32)
            self.trainer.lr_scheduler = LrScheduler()()
            self.trainer.optimizer = Optimizer()(lr_scheduler=self.trainer.lr_scheduler,
                                                 epoch=epoch,
                                                 distributed=self.trainer.distributed)
            alphas = tf.convert_to_tensor(self.alphas)
            for j in range(self.alg_policy.num_individual_per_iter):
                i = np.random.randint(0, self.alg_policy.num_individual, 1)[0]
                if self.epoch < self.alg_policy.warmup:
                    alpha = tf.convert_to_tensor(self.search_alg.random_sample_path())
                else:
                    alpha = alphas[i]
                logits = self.trainer.model(features, alpha, True)
                logits = tf.cast(logits, tf.float32)
                loss = self.trainer.loss(logits=logits, labels=labels)
                grads, vars = zip(*self.trainer.optimizer.compute_gradients(loss))
                if j == 0:
                    accum_grads = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in grads]
                accum_grads = [accum_grads[k] + grads[k] for k in range(len(grads))]
                if self.epoch < self.alg_policy.warmup:
                    break
            clipped_grads, _ = tf.clip_by_global_norm(accum_grads, self.trainer.config.grad_clip)
            minimize_op = self.trainer.optimizer.apply_gradients(list(zip(clipped_grads, vars)), global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(minimize_op, update_ops)

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            alpha = tf.convert_to_tensor(self.trainer.valid_alpha)
            logits = self.trainer.model(features, alpha, False)
            logits = tf.cast(logits, tf.float32)
            loss = self.trainer.loss(logits=logits, labels=labels)
            eval_metric_ops = self.trainer.valid_metrics(logits, labels)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.alphas = self.search_alg.search_evol_arch(epoch, self.alg_policy, self.trainer, self.alphas)

    def set_algorithm_model(self, model):
        """Set model to algorithm.

        :param model: network model
        :type model: torch.nn.Module
        """
        self.search_alg.set_model(model)
