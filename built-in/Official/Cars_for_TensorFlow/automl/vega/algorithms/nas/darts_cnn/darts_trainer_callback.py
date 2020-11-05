# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""DARTS trainer."""
import logging
import os
from copy import deepcopy
import vega
from vega.core.common import Config, FileOps
from vega.algorithms.nas.darts_cnn import DartsNetworkTemplateConfig
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.callbacks import Callback
from vega.search_space import SearchSpace
from vega.search_space.search_algs import SearchAlgorithm
from vega.core.trainer.modules.optimizer import Optimizer
from vega.core.trainer.modules.lr_schedulers import LrScheduler
from vega.core.trainer.modules.losses import Loss

if vega.is_torch_backend():
    import torch
elif vega.is_tf_backend():
    import tensorflow as tf


@ClassFactory.register(ClassType.CALLBACK)
class DartsTrainerCallback(Callback):
    """A special callback for DartsTrainer."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        self.unrolled = self.trainer.config.unrolled
        self.device = self.trainer.config.device
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self.loss = self.trainer.loss
        self.search_alg = SearchAlgorithm(SearchSpace().search_space)
        self._set_algorithm_model(self.model)
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        if vega.is_torch_backend():
            self.valid_loader_iter = iter(self.trainer.valid_loader)

    def before_train_step(self, epoch, logs=None):
        """Be called before a batch training."""
        # Get current train batch directly from logs
        train_batch = logs['train_batch']
        train_input, train_target = train_batch
        # Prepare valid batch data by using valid loader from trainer
        try:
            valid_input, valid_target = next(self.valid_loader_iter)
        except Exception:
            self.valid_loader_iter = iter(self.trainer.valid_loader)
            valid_input, valid_target = next(self.valid_loader_iter)
        valid_input, valid_target = valid_input.to(self.device), valid_target.to(self.device)
        # Call arch search step
        self._train_arch_step(train_input, train_target, valid_input, valid_target)

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        child_desc_temp = self.search_alg.codec.calc_genotype(self._get_arch_weights())
        logging.info('normal = %s', child_desc_temp[0])
        logging.info('reduce = %s', child_desc_temp[1])
        self._save_descript()

    def after_train(self, logs=None):
        """Be called after Training."""
        self.trainer._backup()

    def _train_arch_step(self, train_input, train_target, valid_input, valid_target):
        lr = self.lr_scheduler.get_lr()[0]
        self.search_alg.step(train_input, train_target, valid_input, valid_target,
                             lr, self.optimizer, self.loss, self.unrolled)

    def _set_algorithm_model(self, model):
        self.search_alg.set_model(model)

    def train_input_fn(self):
        """Input function for search."""

        def map_to_dict(td, vd):
            return {'train': td[0], 'valid': vd[0]}, {'train': td[1], 'valid': vd[1]}

        dataset = tf.data.Dataset.zip((self.trainer.train_loader.input_fn(),
                                       self.trainer.valid_loader.input_fn()))
        dataset = dataset.map(lambda td, vd: map_to_dict(td, vd))
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return dataset

    def model_fn(self, features, labels, mode):
        """Darts model_fn used by TensorFlow Estimator."""
        logging.info('Darts model function action')
        global_step = tf.train.get_global_step()
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            features, valid_features = features['train'], features['valid']
            labels, valid_labels = labels['train'], labels['valid']
            # update arch
            epoch = tf.cast(global_step, tf.float32) / tf.cast(len(self.trainer.train_loader), tf.float32)
            self.trainer.lr_scheduler = LrScheduler()()
            self.trainer.optimizer = Optimizer()(lr_scheduler=self.trainer.lr_scheduler,
                                                 epoch=epoch,
                                                 distributed=self.trainer.distributed)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            arch_minimize_op = self.search_alg.step(valid_x=valid_features,
                                                    valid_y=valid_labels,
                                                    lr=self.trainer.lr_scheduler.get_lr()[0])
            train_op = tf.group(arch_minimize_op, update_ops)

        logits = self.model(features, mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.cast(logits, tf.float32)
        self.trainer.loss = Loss()()
        loss = self.trainer.loss(logits=logits, labels=labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.control_dependencies([train_op]):
                weight_ops = self.model.get_weight_ops()
                train_op = self.trainer._init_minimize_op(loss, global_step, weight_ops)

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.trainer.valid_metrics(logits, labels)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    def _get_arch_weights(self):
        if vega.is_torch_backend():
            arch_weights = self.model.arch_weights
        elif vega.is_tf_backend():
            sess_config = self.trainer._init_session_config()
            with tf.Session(config=sess_config) as sess:
                # tf.reset_default_graph()
                checkpoint_file = tf.train.latest_checkpoint(self.trainer.get_local_worker_path())
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
                arch_weights = self.model.arch_weights
                arch_weights = [weight.eval() for weight in arch_weights]
        return arch_weights

    def _save_descript(self):
        """Save result descript."""
        template_file = self.config.darts_template_file
        genotypes = self.search_alg.codec.calc_genotype(self._get_arch_weights())
        if template_file == "{default_darts_cifar10_template}":
            template = DartsNetworkTemplateConfig.cifar10
        elif template_file == "{default_darts_imagenet_template}":
            template = DartsNetworkTemplateConfig.imagenet
        else:
            dst = FileOps.join_path(self.trainer.get_local_worker_path(), os.path.basename(template_file))
            FileOps.copy_file(template_file, dst)
            template = Config(dst)
        model_desc = self._gen_model_desc(genotypes, template)
        self.trainer.config.codec = model_desc

    def _gen_model_desc(self, genotypes, template):
        model_desc = deepcopy(template)
        model_desc.super_network.normal.genotype = genotypes[0]
        model_desc.super_network.reduce.genotype = genotypes[1]
        return model_desc
