# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""TIMM method trainer."""
import os
import copy
import importlib
import torch
from vega.core.common import Config, obj2config
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps
from vega.core.trainer.callbacks import Callback
from vega.core.common.loader import load_conf_from_desc

try:
    from timm import create_model
    from timm.optim.optim_factory import create_optimizer, add_weight_decay
    from timm.scheduler import create_scheduler
    from timm.data import Dataset, create_transform
    from timm.utils import ModelEma
    # additional dependencies
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from timm.data.loader import fast_collate, PrefetchLoader
    from timm.data.distributed_sampler import OrderedDistributedSampler
except Exception:
    # logging.warning("timm not been installed, {}".format(str(e)))
    pass
try:
    import apex
    from apex import amp
except Exception:
    # logging.warning("apex not been installed, {}".format(str(e)))
    pass
try:
    import horovod.torch as hvd
except Exception:
    # logging.warning("horovod not been installed, {}".format(str(e)))
    pass


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        rand_erase_prob=0.,
        rand_erase_mode='const',
        rand_erase_count=1,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        fp16=False,
        tf_preprocessing=False,
        world_size=None,
        rank=None
):
    """Create data loader for timm."""
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
    )

    sampler = None
    if distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset, num_replicas=world_size, rank=rank)

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=is_training,
    )
    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            re_prob=rand_erase_prob if is_training else 0.,
            re_mode=rand_erase_mode,
            re_count=rand_erase_count,
            mean=mean,
            std=std,
            fp16=fp16)

    return loader


@ClassFactory.register(ClassType.CALLBACK)
class TimmTrainerCallback(Callback):
    """A special callback for TimmTrainer."""

    disable_callbacks = ["LearningRateScheduler", "ModelStatistics"]

    def before_train(self, logs=None):
        """Be called before the training process."""
        self._init_all_settings()

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        if self.distributed:
            self.trainer.train_loader.sampler.set_epoch(epoch)
        self.num_updates = epoch * len(self.trainer.train_loader)
        self.trainer.model.train()

    def make_batch(self, batch):
        """Prepare batch data for train_step."""
        input, target = batch
        if self.config.cuda and not self.config.prefetcher:
            input, target = input.cuda(), target.cuda()
        return input, target

    def train_step(self, batch):
        """Train one step of model."""
        input, target = batch
        self.trainer.optimizer.zero_grad()
        logits = self.trainer.model(input)
        loss = self.trainer.loss(logits, target)
        if self.use_amp:
            with amp.scale_loss(loss, self.trainer.optimizer) as scaled_loss:
                scaled_loss.backward()
                self.trainer.optimizer.synchronize()
            with self.trainer.optimizer.skip_synchronize():
                self.trainer.optimizer.step()
        else:
            loss.backward()
            self.trainer.optimizer.step()
        if self.use_ema:
            self.model_ema.update(self.trainer.model)
        self.num_updates += 1
        self.trainer.lr_scheduler.step_update(num_updates=self.num_updates)
        return {'loss': loss.item(),
                'train_batch_output': logits}

    def before_valid(self, epoch, logs=None):
        """Be called before valid loop."""
        if self.use_ema:
            self.trainer.model = self.model_ema.ema
        self.trainer.model.eval()

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        if self.use_ema:
            self.trainer.model = self.model
        self.trainer.lr_scheduler.step(epoch=epoch + 1)
        if self.trainer.is_chief:
            self.trainer._backup()

    def _init_all_settings(self):  # noqa: C901
        """Init all settings from config."""
        self.config = self.trainer.config
        if self.trainer.hps and self.trainer.hps.get('trainer'):
            load_conf_from_desc(self.config, self.trainer.hps.get('trainer'))
        self.trainer._init_distributed_setting()
        if self.config.cuda:
            self.trainer._init_cuda_setting()
        self.epochs = self.trainer.epochs
        self.distributed = self.trainer.distributed
        self.trainer.model = self._init_model()
        self.model = self.trainer.model
        self.use_syncbn = self.config.syncbn
        self.trainer.use_syncbn = self.use_syncbn
        if self.use_syncbn:
            self.trainer.model = apex.parallel.convert_syncbn_model(self.trainer.model)
        self.trainer.optimizer = self._init_optimizer()
        self.use_ema = hasattr(self.config, 'model_ema')
        if self.use_ema:
            self.model_ema = self._init_model_ema()
        self.trainer.lr_scheduler = self._init_lr_scheduler()
        self.trainer.loss = self._init_loss()
        if self.distributed:
            self.trainer._init_horovod_setting()
        self.use_amp = self.config.amp
        if self.use_amp:
            self.trainer.model, self.trainer.optimizer = amp.initialize(self.trainer.model,
                                                                        self.trainer.optimizer,
                                                                        opt_level='O1')
        self._init_dataloader()
        self.trainer.valid_metrics = self.trainer._init_metrics(None)
        self.trainer._init_step_functions(self.make_batch, self.train_step, None)
        self.trainer.callbacks._set_params(self.trainer)

        self.trainer.has_built = True

    def _init_model_ema(self):
        """Init Model Ema."""
        args = self.config.model_ema
        model_ema = ModelEma(self.trainer.model,
                             decay=args.model_ema_decay,
                             device='cpu' if args.model_ema_force_cpu else '',
                             resume=None)
        return model_ema

    def _init_model(self):
        """Init network model from timm according to model type in config."""
        args = self.config.model_desc
        model = create_model(args.model_name,
                             pretrained=args.pretrained,
                             num_classes=args.num_classes,
                             drop_rate=args.drop,
                             drop_path_rate=args.drop_path,
                             global_pool=args.gp,
                             bn_tf=args.bn_tf,
                             bn_momentum=args.bn_momentum,
                             bn_eps=args.bn_eps,
                             checkpoint_path=args.initial_checkpoint)
        if self.config.cuda:
            model = model.cuda()
        return model

    def _init_optimizer(self):
        """Init optimizer from timm according to optim type in config."""
        optimizer = create_optimizer(self.config.optim, self.trainer.model)
        if self.distributed:
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=self.trainer.model.named_parameters(),
                                                 compression=hvd.Compression.none)
        return optimizer

    def _init_lr_scheduler(self):
        """Init lr scheduler from timm according to type in config."""
        args = obj2config(copy.deepcopy(self.config.lr_scheduler))
        args['epochs'] = self.config.epochs
        lr_scheduler, self.config.epochs = create_scheduler(Config(args), self.trainer.optimizer)
        start_epoch = args.get('start_epoch', 0)
        lr_scheduler.step(start_epoch)
        return lr_scheduler

    def _init_loss(self):
        """Init loss function from timm according to type in config."""
        loss_name = self.config.loss.type
        loss_config = obj2config(copy.deepcopy(self.config.loss))
        loss_class = getattr(importlib.import_module('timm.loss'), loss_name)
        loss_fn = loss_class(**loss_config)
        if self.config.cuda:
            loss_fn = loss_fn.cuda()
        return loss_fn

    def _reset_sync_opt(self):
        """Rest sysnc opt."""
        params = add_weight_decay(self.model, self.config.optim.weight_decay)
        self.optimizer.param_groups = []
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.optimizer.add_param_group(param_group)

    def _init_dataloader(self):
        """Init dataloader from timm."""
        if self.distributed and hvd.local_rank() == 0 and 'remote_data_dir' in self.config.dataset:
            FileOps.copy_folder(self.config.dataset.remote_data_dir, self.config.dataset.data_dir)
        if self.distributed:
            hvd.join()
        args = self.config.dataset
        train_dir = os.path.join(self.config.dataset.data_dir, 'train')
        dataset_train = Dataset(train_dir)
        world_size, rank = None, None
        if self.distributed:
            world_size, rank = hvd.size(), hvd.rank()
        self.trainer.train_loader = create_loader(
            dataset_train,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=self.config.prefetcher,
            rand_erase_prob=args.reprob,
            rand_erase_mode=args.remode,
            rand_erase_count=args.recount,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='random',
            mean=tuple(args.mean),
            std=tuple(args.std),
            num_workers=args.workers,
            distributed=self.distributed,
            world_size=world_size,
            rank=rank
        )
        valid_dir = os.path.join(self.config.dataset.data_dir, 'val')
        dataset_eval = Dataset(valid_dir)
        self.trainer.valid_loader = create_loader(
            dataset_eval,
            input_size=tuple(args.input_size),
            batch_size=4 * args.batch_size,
            is_training=False,
            use_prefetcher=self.config.prefetcher,
            interpolation=args.interpolation,
            mean=tuple(args.mean),
            std=tuple(args.std),
            num_workers=args.workers,
            distributed=self.distributed,
            world_size=world_size,
            rank=rank
        )
