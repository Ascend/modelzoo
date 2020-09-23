import argparse
import time
import numpy as np

import mindspore
from mindspore import Tensor, context
import mindspore.common.dtype as mstype

from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint, _get_merged_param_data
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.model import Model

from src.config import Config_CNNCTC
from src.callback import LossCallBack
from src.dataset import ST_MJ_Generator_batch_fixed_length
from src.CNNCTC.model import CNNCTC_Model, ctc_loss, WithLossCell, TrainOneStepCell

config = Config_CNNCTC()
CHARACTER = config.CHARACTER

NUM_CLASS = config.NUM_CLASS
HIDDEN_SIZE = config.HIDDEN_SIZE
FINAL_FEATURE_WIDTH = config.FINAL_FEATURE_WIDTH

TRAIN_BATCH_SIZE = config.TRAIN_BATCH_SIZE
TRAIN_DATASET_SIZE = config.TRAIN_DATASET_SIZE

TRAIN_EPOCHS = config.TRAIN_EPOCHS

CKPT_PATH = config.CKPT_PATH
SAVE_PATH = config.SAVE_PATH

LR = config.LR
MOMENTUM = config.MOMENTUM
LOSS_SCALE = config.LOSS_SCALE
SAVE_CKPT_PER_N_STEP = config.SAVE_CKPT_PER_N_STEP
KEEP_CKPT_MAX_NUM = config.KEEP_CKPT_MAX_NUM

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False,
                    save_graphs_path=".", enable_auto_mixed_precision=False)


def dataset_creator():
    ds = GeneratorDataset(ST_MJ_Generator_batch_fixed_length, ['img', 'label_indices', 'text', 'sequence_length'],
                          num_parallel_workers=4)
    ds.set_dataset_size(int(TRAIN_DATASET_SIZE // TRAIN_BATCH_SIZE))
    ds = ds.repeat(TRAIN_EPOCHS)
    return ds


def train():
    ds = dataset_creator()
    # network
    net = CNNCTC_Model(NUM_CLASS, HIDDEN_SIZE, FINAL_FEATURE_WIDTH)
    net.set_train(True)

    if CKPT_PATH != '':
        param_dict = load_checkpoint(CKPT_PATH)
        load_param_into_net(net, param_dict)
        print('parameters loaded!')
    else:
        print('train from scratch...')

    criterion = ctc_loss()
    opt = mindspore.nn.RMSProp(params=net.trainable_params(), centered=True, learning_rate=LR, momentum=MOMENTUM,
                               loss_scale=LOSS_SCALE)

    net = WithLossCell(net, criterion)
    net = TrainOneStepCell(net, opt)

    callback = LossCallBack()
    # set parameters of check point
    config_ck = CheckpointConfig(save_checkpoint_steps=SAVE_CKPT_PER_N_STEP, keep_checkpoint_max=KEEP_CKPT_MAX_NUM)
    # apply parameters of check point
    ckpoint_cb = ModelCheckpoint(prefix="CNNCTC", config=config_ck, directory=SAVE_PATH)

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(LOSS_SCALE, False)
    model = Model(net, loss_scale_manager=loss_scale_manager)
    model._train_network = net
    model.train(TRAIN_EPOCHS, ds, callbacks=[callback, ckpoint_cb], dataset_sink_mode=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FasterRcnn training")
    parser.add_argument("--ckpt_path", type=str, default="", help="Pretrain file path.")
    args_opt = parser.parse_args()
    if args_opt.ckpt_path != "":
        CKPT_PATH = args_opt.ckpt_path
    train()
