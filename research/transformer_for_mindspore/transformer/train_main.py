import os
import numpy as np
import pytest
import time
import math
import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
#import mindspore.data as dt
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn.optim import AdamWeightDecayDynamicLR, Adam
from mindspore.nn.optim import Lamb, Momentum
from mindspore.train.model import Model
from mindspore.nn import WithLossCell
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint
import mindspore.communication.management as D
from mindspore.train.parallel_utils import ParallelMode
import mindspore.dataset.engine as de
import mindspore._c_dataengine as deMap
import mindspore.dataset.transforms.c_transforms as deC
from mindspore import context

from transformer_model import TransformerConfig
from transformer_for_train  import TransformerTraining, TransformerTrainingLoss, \
    TransformerNetworkWithLoss, TransformerTrainOneStepCell, TransformerTrainOneStepWithLossScaleCell
from lr_schedule import dynamic_lr

device_id=os.getenv('DEVICE_ID', None)
if device_id is None:
    raise Exception("device_id is None!!!")
device_id=int(device_id)
context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=True,
        device_target="Davinci",
        reserve_class_name_in_scope=False,
        device_id=device_id)

def gen_fake_distribute_file_json(distribute_file, world_size=1, local_rank=0):
    random_method = 'RANDOM'
    shuffle = 'ON'
    seed = 0

    content = '{\n' \
              '  "deviceNum":%s,\n' \
              '  "deviceId": %s,\n' \
              '  "shardConfig":"%s",\n' \
              '  "shuffle":"%s",\n' \
              '  "seed": %s\n' \
              '}' % (world_size, local_rank, random_method, shuffle, seed)

    with open(distribute_file, 'w') as fw:
        fw.write(content)


def test_gen_fake_minddata():
    local_rank = D.get_rank()
    world_size = D.get_group_size()
    minddata_json = 'minddata_json'
    print('world_size:{}'.format(world_size))
    print('local_rank:{}'.format(local_rank))

    # first remove minddata json and data
    import shutil
    if os.path.exists(minddata_json):
        shutil.rmtree(minddata_json)

    if not os.path.exists(minddata_json):
        os.makedirs(minddata_json)

    for rank in range(world_size):
        distribute_file = os.path.join(minddata_json, 'distribution_{}.json'.format(rank))
        gen_fake_distribute_file_json(distribute_file, world_size=world_size, local_rank=rank)



def load_test_data(dataset_dir, epoch_count=1, batch_size=1, sink_mode=False, distribute_file=''):
    schema_file = dataset_dir + 'datasetSchema.json'
    data_file = []
    dirs = os.listdir(dataset_dir)
    for file in dirs:
        if 'ende.l128.tfrecord' in file:
            data_file.append(os.path.join(dataset_dir, file))

    if distribute_file != '':
        test_gen_fake_minddata()
    ds = de.StorageDataset(data_file, schema_file, distribution=distribute_file,
                        columns_list=["source_eos_ids","source_eos_mask",
                                      "target_sos_ids","target_sos_mask",
                                      "target_eos_ids","target_eos_mask"])

    #ds = de.TFRecordDataset(data_file, schema_file,
    #                    columns_list=["source_eos_ids","source_eos_mask",
    #                                  "target_sos_ids","target_sos_mask",
    #                                  "target_eos_ids","target_eos_mask"],
    #                    shuffle=False, num_shards=rank_size, shard_id=rank_id)

    ### circular sinking for TDT mode ###
    sink_step = 1
    ori_dataset_size = ds.get_dataset_size()
    if sink_mode:
        ds.set_dataset_size(sink_step * batch_size)
        repeat_count = epoch_count * ori_dataset_size // ds.get_dataset_size()

    # apply shuffle operations
    buffer_size = ori_dataset_size
    ds = ds.shuffle(buffer_size=buffer_size)

    type_cast_op = deC.TypeCast(mstype.int32)
    ds = ds.map(input_columns="source_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="source_eos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_mask", operations=type_cast_op)
    
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    if sink_mode:
        ds = ds.repeat(repeat_count)
    else:
        ds = ds.repeat(1)

    ds.channel_name = 'transformer'
    return ds


def get_config(version='large', batch_size=16):
    if version == 'large':
        return TransformerConfig(
            batch_size=batch_size,
            seq_length=128,
            vocab_size=36560,
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="relu",
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
            max_position_embeddings=128,
            initializer_range=0.02,
            label_smoothing=0.1,
            input_mask_from_dataset=True,
            dtype=mstype.float32,
            compute_type=mstype.float16)
    elif version == 'base':
        return TransformerConfig(
            batch_size=batch_size,
            seq_length=128,
            vocab_size=46528,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_act="relu",
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
            max_position_embeddings=128,
            initializer_range=0.02,
            label_smoothing=0.1,
            input_mask_from_dataset=True,
            dtype=mstype.float32,
            compute_type=mstype.float16)
    raise ValueError("unknown config version")


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))

time_stamp_init = False
time_stamp_first = 0

class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times

        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()

        cb_params = run_context.original_args()
        with open("./loss.log", "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first, cb_params.cur_epoch_num, cb_params.cur_step_num, str(cb_params.net_outputs)))
            f.write('\n')


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.
  Args:
    shape: Integer shape tuple or TF tensor shape.
  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)


def weight_variable(shape):
    #ones = np.ones(shape).astype(np.float32)
    #return Tensor(ones*0.001)
    scale_shape = shape
    fan_in, fan_out = _compute_fans(scale_shape)
    scale = 1.0 / max(1., (fan_in + fan_out) / 2.)
    limit = math.sqrt(3.0 * scale)
    values =  np.random.uniform(-limit, limit, shape).astype(np.float32)
    return Tensor(values)

def one_weight(shape):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)

def zero_weight(shape):
    #np.random.seed(1)
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)

def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units**-0.5, shape).astype(np.float32)
    return Tensor(norm)

def argparse_init():
    parser = argparse.ArgumentParser(description='transformer')
 
    parser.add_argument("--data_path", type=str, default="./dataset/")  # The location of the input tfrecord data.
    parser.add_argument("--train_epochs", type=int, default=52)  # The number of epochs used to train.
    parser.add_argument("--batch_size", type=int, default=96)  # The batch size used to train.
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/")  # The location of the checkpoint file.
    return parser


def test_transformer_train_single():
    parser = argparse_init()
    args, _ = parser.parse_known_args()
    dataset_dir = args.data_path
    epoch_count = args.train_epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path

    version = os.getenv('VERSION', 'large')
    sink_mode = False
    dataset = load_test_data(dataset_dir, epoch_count=epoch_count, batch_size=batch_size, sink_mode=sink_mode)

    config = get_config(version=version, batch_size=batch_size)
    netwithloss = TransformerNetworkWithLoss(config, True)

    params = netwithloss.trainable_params()

    for param in params:
       name = param.name
       value = param.default_input
       if isinstance(value, Tensor):
           if name.endswith(".gamma"):
               param.default_input = one_weight(value.asnumpy().shape)
           elif name.endswith(".beta") or name.endswith(".bias"):
               param.default_input = zero_weight(value.asnumpy().shape)
           elif "embedding" in name:
               param.default_input = normal_weight(value.asnumpy().shape, config.hidden_size)
           else:
               param.default_input = weight_variable(value.asnumpy().shape)

    lr = Tensor(dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay", training_steps=4498687//batch_size*epoch_count, learning_rate=1.0, warmup_steps=16000, hidden_size=config.hidden_size), mstype.float32)
    #optimizer = Adam(netwithloss.trainable_params(), lr, loss_scale=128.0)
    optimizer = Adam(netwithloss.trainable_params(), lr)

    #netwithgrads = TransformerTrainOneStepCell(netwithloss, optimizer=optimizer, sens=128.0)
    # dynamic loss scale
    scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_factor=2, scale_window=2000)
    update_cell = scale_manager.get_update_cell()
    netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)

    netwithgrads.set_train(True)
    
    if sink_mode:
        epoch_size = dataset.get_repeat_count()
    else:
        epoch_size = epoch_count

    model = Model(netwithgrads)
    callback = LossCallBack()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=2500, keep_checkpoint_max=200)
    ckpoint_cb = ModelCheckpoint(prefix='transformer', directory=checkpoint_path, config=ckpt_config)
    model.train(epoch_size, dataset, callbacks=[callback, ckpoint_cb], dataset_sink_mode=sink_mode)


def test_transformer_train_parallel():
    parser = argparse_init()
    args, _ = parser.parse_known_args()
    dataset_dir = args.data_path
    epoch_count = args.train_epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path

    rank_size=os.getenv('RANK_SIZE', None)
    rank_id=os.getenv('RANK_ID', None)
    rank_size=int(rank_size)
    rank_id=int(rank_id)
    if rank_size is None:
        raise Exception("rank_size is None!!!")
    if rank_id is None:
        raise Exception("rank_id is None!!!")
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.DATA_PARALLEL,
        device_num=rank_size,
        parameter_broadcast=True,
        mirror_mean=True)
    D.init()

    version = os.getenv('VERSION', 'large')
    sink_mode = False
    minddata_dir = 'minddata_json'
    distribute_file = os.path.join(minddata_dir, 'distribution_{}.json'.format(D.get_rank()))
    dataset = load_test_data(dataset_dir, epoch_count=epoch_count, batch_size=batch_size, sink_mode=sink_mode, distribute_file=distribute_file)

    config = get_config(version=version, batch_size=batch_size)
    netwithloss = TransformerNetworkWithLoss(config, True)

    params = netwithloss.trainable_params()

    for param in params:
       name = param.name
       value = param.default_input
       if isinstance(value, Tensor):
           if name.endswith(".gamma"):
               param.default_input = one_weight(value.asnumpy().shape)
           elif name.endswith(".beta") or name.endswith(".bias"):
               param.default_input = zero_weight(value.asnumpy().shape)
           elif "embedding" in name:
               param.default_input = normal_weight(value.asnumpy().shape, config.hidden_size)
           else:
               param.default_input = weight_variable(value.asnumpy().shape)

    lr = Tensor(dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay", training_steps=4498687//rank_size//batch_size*epoch_count, learning_rate=1.0, warmup_steps=16000, hidden_size=config.hidden_size), mstype.float32)
 
    #optimizer = Adam(netwithloss.trainable_params(), lr, loss_scale=128.0)
    optimizer = Adam(netwithloss.trainable_params(), lr)

    #netwithgrads = TransformerTrainOneStepCell(netwithloss, optimizer=optimizer, sens=128.0)
    # dynamic loss scale
    scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_factor=2, scale_window=2000)
    update_cell = scale_manager.get_update_cell()
    netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)

    netwithgrads.set_train(True)
    
    if sink_mode:
        epoch_size = dataset.get_repeat_count()
    else:
        epoch_size = epoch_count

    model = Model(netwithgrads)
    callback = LossCallBack()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=2500, keep_checkpoint_max=200)
    ckpoint_cb = ModelCheckpoint(prefix='transformer', directory=checkpoint_path, config=ckpt_config)
    model.train(epoch_size, dataset, callbacks=[callback, ckpoint_cb], dataset_sink_mode=sink_mode)

if __name__ == '__main__':
    rank_size=os.getenv('RANK_SIZE', None)
    if rank_size is None:
        test_transformer_train_single()
    else: 
        test_transformer_train_parallel()


