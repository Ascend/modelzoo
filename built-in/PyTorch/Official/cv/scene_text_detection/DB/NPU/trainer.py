#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import os

import torch
from tqdm import tqdm

from experiment import Experiment
from data.data_loader import DistributedSampler
from apex import amp

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.init_device()
        if self.experiment.train.data_loader.seed is not None:
            print("fix seed, seed = ", self.experiment.train.data_loader.seed)
            seed_everything(self.experiment.train.data_loader.seed)
        self.structure = experiment.structure
        self.logger = experiment.logger
        self.model_saver = experiment.train.model_saver

        # FIXME: Hack the save model path into logger path
        self.model_saver.dir_path = self.logger.save_dir(
            self.model_saver.dir_path)
        self.current_lr = 0

        self.total = 0

    def init_device(self):
        if torch.npu.is_available():
            self.device = torch.device(f'npu:{self.experiment.device_id}')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(
            self.device, self.experiment.distributed, self.experiment.local_rank)
        return model

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.experiment.train.scheduler.learning_rate.get_learning_rate(
            epoch, step)

        for group in optimizer.param_groups:
            group['lr'] = lr
        self.current_lr = lr

    def train(self):
        self.logger.report_time('Start')
        self.logger.args(self.experiment)
        model = self.init_model()
        train_data_loader = self.experiment.train.data_loader.train_loader
        if self.experiment.validation:
            validation_loaders = self.experiment.validation.data_loaders

        self.steps = 0
        epoch = 0
        if self.experiment.train.checkpoint:
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta

        # Init start epoch and iter
        optimizer = self.experiment.train.scheduler.create_optimizer(
            model.parameters())

        amp.register_float_function(torch, "sigmoid")
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1024)
        self.logger.report_time('Init')

        model.train()
        while True:
            self.logger.info('Training epoch ' + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(train_data_loader)

            for batch in train_data_loader:
                self.update_learning_rate(optimizer, epoch, self.steps)

                self.logger.report_time("Data loading")

                if self.experiment.validation and\
                        self.steps % self.experiment.validation.interval == 0 and\
                        self.steps > self.experiment.validation.exempt:
                    self.validate(validation_loaders, model, epoch, self.steps)
                self.logger.report_time('Validating ')
                if self.logger.verbose:
                    torch.npu.synchronize()

                self.train_step(model, optimizer, batch,
                                epoch=epoch, step=self.steps)
                if self.logger.verbose:
                    torch.npu.synchronize()
                self.logger.report_time('Forwarding ')

                self.model_saver.maybe_save_model(
                    model, epoch, self.steps, self.logger)

                self.steps += 1
                self.logger.report_eta(self.steps, self.total, epoch)

            epoch += 1
            if epoch > self.experiment.train.epochs:
                if self.experiment.local_rank in [-1, 0]:
                    self.model_saver.save_checkpoint(model, 'final')
                    if self.experiment.validation:
                        self.validate(validation_loaders, model, epoch, self.steps)
                    self.logger.info('Training done')
                break
            iter_delta = 0

    def train_step(self, model, optimizer, batch, epoch, step, **kwards):
        # Temporary solution for huge memory usage by topk.
        # This should be deleted in future version.
        if step == 0 and epoch == 0:
            tmp = torch.rand(16, 16, 640, 640, dtype=torch.float32).npu()
            top, _ = torch.topk(tmp.view(-1).half(), 4000000)
        optimizer.zero_grad()
        results = model.forward(batch, training=True)
        if len(results) == 2:
            l, pred = results
            metrics = {}
        elif len(results) == 3:
            l, pred, metrics = results

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).npu()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if step % self.experiment.logger.log_interval == 0:
            if isinstance(l, dict):
                line = '\t'.join(line)
                log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(step, epoch, line, self.current_lr)
                self.logger.info(log_info)
            else:
                self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
                    step, epoch, loss.item(), self.current_lr))
            self.logger.add_scalar('loss', loss, step)
            self.logger.add_scalar('learning_rate', self.current_lr, step)
            for name, metric in metrics.items():
                self.logger.add_scalar(name, metric, step)
                self.logger.info('%s: %6f' % (name, metric))

            self.logger.report_time('Logging')

    def validate(self, validation_loaders, model, epoch, step):
        all_matircs = {}
        model.eval()
        for name, loader in validation_loaders.items():
            if self.experiment.validation.visualize:
                metrics, vis_images = self.validate_step(
                    loader, model, True)
                self.logger.images(
                    os.path.join('vis', name), vis_images, step)
            else:
                metrics, vis_images = self.validate_step(loader, model, False)
            for _key, metric in metrics.items():
                key = name + '/' + _key
                if key in all_matircs:
                    all_matircs[key].update(metric.val, metric.count)
                else:
                    all_matircs[key] = metric

        for key, metric in all_matircs.items():
            self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))
        self.logger.metrics(epoch, self.steps, all_matircs)
        model.train()
        return all_matircs

    def validate_step(self, data_loader, model, visualize=False):
        raw_metrics = []
        vis_images = dict()
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred)
            raw_metric, interested = self.structure.measurer.validate_measure(
                batch, output)
            raw_metrics.append(raw_metric)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.visualize(
                    batch, output, interested)
                vis_images.update(vis_image)
        metrics = self.structure.measurer.gather_measure(
            raw_metrics, self.logger)
        return metrics, vis_images

    def to_np(self, x):
        return x.cpu().data.numpy()
