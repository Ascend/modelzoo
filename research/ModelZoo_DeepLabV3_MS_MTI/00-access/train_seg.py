import sys
import os
import gflags
import logging
import time
import numpy as np

from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
context.set_context(device_id=int(os.getenv('DEVICE_ID')), enable_auto_mixed_precision=True)
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import TrainOneStepCell
import mindspore.nn as nn
from mindspore.ops.operations import TensorAdd
from mindspore.ops import operations as P
from mindspore.common.initializer import XavierUniform, HeUniform
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import _InternalCallbackParam, RunContext
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from data import data_generator
from loss import loss
from nets import net_factory
from utils import learning_rates

FLAGS = gflags.FLAGS

gflags.DEFINE_string('data_file', '',
                     'path and name of one mindrecord file')

gflags.DEFINE_string('train_dir', '',
                     'where training log and ckpts saved')

gflags.DEFINE_integer('train_epochs', 300, 'number of shards')

gflags.DEFINE_integer('batch_size', 32, 'batch size')

gflags.DEFINE_integer('crop_size', 513, 'crop size')

gflags.DEFINE_float('base_lr', 0.02, 'base learning rate')

gflags.DEFINE_integer('lr_decay_step', 40000, 'crop size')

gflags.DEFINE_integer('lr_decay_rate', 0.1, 'learning rate decay rate')

gflags.DEFINE_string('lr_type', 'cos', 'type of learning rate')

gflags.DEFINE_float('min_scale', 0.5, 'minimum scale of data augumentation')

gflags.DEFINE_float('max_scale', 2.0, 'maximum scale of data augumentation')

gflags.DEFINE_multi_float('image_mean', [103.53, 116.28, 123.675], 'image means')

gflags.DEFINE_multi_float('image_std', [57.375, 57.120, 58.395], 'image stds')

gflags.DEFINE_integer('ignore_label', 255, 'ignore label')

gflags.DEFINE_integer('num_classes', 21, 'number of classes')

gflags.DEFINE_string('model', 'deeplab_v3_s16', 'select model')

gflags.DEFINE_bool('freeze_bn', False, 'freeze bn')

gflags.DEFINE_float('loss_scale', 3072.0, 'loss scale')

gflags.DEFINE_string('ckpt_pre_trained', '', 'select pre-trained model')

gflags.DEFINE_integer('save_steps', 3000, 'steps interval for saving')

gflags.DEFINE_integer('keep_checkpoint_max', 10, 'max checkpoint for saving')


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


def train():
    FLAGS(sys.argv)

    # dataset
    data_set = data_generator.SegDataset(data_file=FLAGS.data_file,
                                         batch_size=FLAGS.batch_size,
                                         crop_size=FLAGS.crop_size,
                                         image_mean=FLAGS.image_mean,
                                         image_std=FLAGS.image_std,
                                         max_scale=FLAGS.max_scale,
                                         min_scale=FLAGS.min_scale,
                                         ignore_label=FLAGS.ignore_label,
                                         num_classes=FLAGS.num_classes,
                                         num_readers=2,
                                         num_parallel_calls=4)

    dataset = data_set.get_dataset(repeat=FLAGS.train_epochs)
    data_iter = dataset.create_dict_iterator()

    # nets
    if FLAGS.model == 'deeplab_v3_s16':
        network = net_factory.nets_map[FLAGS.model]('train', FLAGS.num_classes, 16, FLAGS.freeze_bn)
    elif FLAGS.model == 'deeplab_v3_s8':
        network = net_factory.nets_map[FLAGS.model]('train', FLAGS.num_classes, 8, FLAGS.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(FLAGS.model))
    print(network)
    loss_ = loss.SoftmaxCrossEntropyLoss(FLAGS.num_classes, FLAGS.ignore_label)
    loss_.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(network, loss_)
    if FLAGS.ckpt_pre_trained:
        param_dict = load_checkpoint(FLAGS.ckpt_pre_trained)
        load_param_into_net(train_net, param_dict)

    print('pretrain', param_dict.keys())
    print('network', network.parameters_dict().keys())
    iters_per_epoch = dataset.get_dataset_size()
    print('iters_per_epoch', iters_per_epoch)
    total_train_steps = iters_per_epoch * FLAGS.train_epochs
    if FLAGS.lr_type == 'cos':
        lr_iter = learning_rates.cosine_lr(FLAGS.base_lr, total_train_steps,
                                           total_train_steps)
    elif FLAGS.lr_type == 'poly':
        lr_iter = learning_rates.poly_lr(FLAGS.base_lr, total_train_steps,
                                         total_train_steps, end_lr=0.0, power=0.9)
    elif FLAGS.lr_type == 'exp':
        lr_iter = learning_rates.exponential_lr(FLAGS.base_lr,
                                                FLAGS.lr_decay_step, FLAGS.lr_decay_rate, total_train_steps,
                                                staircase=True)
    else:
        raise ValueError('unknown learning rate type')
    opt = nn.Momentum(params=train_net.trainable_params(),
                      learning_rate=lr_iter, momentum=0.9, weight_decay=0.0001,
                      loss_scale=FLAGS.loss_scale)

    train_net = TrainOneStepCell(train_net, opt, sens=FLAGS.loss_scale)
    train_net.set_train(True)

    # callback for saving ckpts
    config_ck = CheckpointConfig(save_checkpoint_steps=FLAGS.save_steps,
                                 keep_checkpoint_max=FLAGS.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=FLAGS.model,
                                 directory=FLAGS.train_dir, config=config_ck)

    cb_params = _InternalCallbackParam()
    cb_params.cur_step_num = 0
    cb_params.batch_num = iters_per_epoch
    cb_params.train_network = train_net
    cb_params.epoch_num = FLAGS.train_epochs
    cb_params.step_num = total_train_steps
    run_context = RunContext(cb_params)

    # logging
    LOG_FORMAT = "%(asctime)s %(message)s"
    DATE_FORMAT = '%Y-%m-%d  %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        filename=os.path.join(FLAGS.train_dir, 'log')
                        )

    total_fps = []
    for ep in range(FLAGS.train_epochs):

        cb_params.cur_epoch_num = ep + 1
        for it in range(iters_per_epoch):
            cb_params.cur_step_num += 1
            sample_ = data_iter.get_next()
            start_time = time.time()
            loss_val = train_net(Tensor(sample_['data'], mstype.float32),
                                 Tensor(sample_['label']))
            end_time = time.time()
            total_fps.append(FLAGS.batch_size / (end_time - start_time))
            mean_fps = np.mean(total_fps)
            print('epoch: {}   step: {}    loss: {}    mean_fps: {} imgs/sec'.format(ep + 1, it + 1,
                                                                                     loss_val, mean_fps))
            ckpoint_cb.step_end(run_context)
            logging.info('epoch: {}   step: {}    loss: {}   mean_fps: {} imgs/sec'.format(ep + 1, it + 1,
                                                                                            loss_val, mean_fps))


if __name__ == '__main__':
    train()











