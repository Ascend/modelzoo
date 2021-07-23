# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train_criteo."""
import os
import sys
import argparse
import numpy as np

import moxing as mox
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.common import set_seed

base_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/..")
sys.path.append(base_path + "/../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepfm import ModelBuilder, AUCMetric, PredictWithSigmoid, DeepFMModel
from src.config import DataConfig, ModelConfig, TrainConfig
from src.dataset import create_dataset, DataType
from src.callback import EvalCallBack, LossCallBack

parser = argparse.ArgumentParser(description='CTR Prediction')
parser.add_argument(
    '--train_url',
    type=str,
    default="",
    help='the obs path model saved')
parser.add_argument('--data_url', type=str, default="",
                    help='the obs path training data saved')
parser.add_argument('--batch_size', type=int, default=16000,
                    help='number of samples for one step training')
parser.add_argument(
    '--data_field_size',
    type=int,
    default=39,
    help='field size of data')
parser.add_argument(
    '--data_vocab_size',
    type=int,
    default=184965,
    help='vocab size of data')
parser.add_argument(
    '--train_epochs',
    type=int,
    default=15,
    help='training epochs')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=5e-4,
    help='learning rate')
parser.add_argument(
    '--resume',
    type=str,
    default="",
    help='obs path of pretrained deepfm model')
parser.add_argument(
    "--file_name",
    type=str,
    default="deepfm",
    help="output file name.")
parser.add_argument(
    "--file_format",
    type=str,
    choices=[
        "AIR",
        "ONNX",
        "MINDIR"],
    default="AIR",
    help="file format")
parser.add_argument('--do_eval', type=bool, default=True,
                    help='evaluate or not betweeen training, only support "True" or "False". Default: "True"')
parser.add_argument('--device_target', type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                    help="device target, support Ascend, GPU and CPU.")
args_opt, _ = parser.parse_known_args()


def get_epoch(ckpt_name):
    """Get epoch from checkpoint name"""
    start = ckpt_name.find('-')
    start += len('-')
    end = ckpt_name.find('_', start)
    epoch = eval(ckpt_name[start:end].strip())
    return epoch


def get_ckpt_epoch(ckpt_dir):
    """Get newest checkpoint name and epoch"""
    ckpt_epoch = {}
    files = os.listdir(ckpt_dir)
    for file_name in files:
        file_path = os.path.join(ckpt_dir, file_name)
        if os.path.splitext(file_path)[1] == '.ckpt':
            epoch = get_epoch(file_name)
            ckpt_epoch[file_name] = epoch
    newest_ckpt = max(ckpt_epoch, key=ckpt_epoch.get)
    max_epoch = ckpt_epoch[newest_ckpt]
    return newest_ckpt, max_epoch


def set_context(args, rank_size):
    """set context"""
    if rank_size > 1:
        if args.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(
                mode=context.GRAPH_MODE,
                device_target=args.device_target,
                device_id=device_id)
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
            rank_id = int(os.environ.get('RANK_ID'))
        elif args.device_target == "GPU":
            init()
            context.set_context(
                mode=context.GRAPH_MODE,
                device_target=args.device_target)
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank_id = get_rank()
        else:
            print("Unsupported device_target ", args.device_target)
            exit()
    else:
        if args.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(
                mode=context.GRAPH_MODE,
                device_target=args.device_target,
                device_id=device_id)
        else:
            context.set_context(
                mode=context.GRAPH_MODE,
                device_target=args.device_target)
        rank_size = None
        rank_id = None
    return rank_size, rank_id


def copy_files(args, rank_size, eval_file_name, loss_file_name):
    """copy files to obs"""
    if not os.path.exists(os.path.dirname(ckpt_path)):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    mox.file.copy_parallel(ckpt_path, args.train_url)
    if rank_size is not None:
        mox.file.copy(eval_file_name, os.path.join(
            args.train_url, 'auc_' + str(get_rank()) + '.log'))
        mox.file.copy(loss_file_name, os.path.join(
            args.train_url, 'loss_' + str(get_rank()) + '.log'))
    else:
        mox.file.copy(
            eval_file_name, os.path.join(
                args.train_url, 'auc.log'))
        mox.file.copy(
            loss_file_name,
            os.path.join(
                args.train_url,
                'loss.log'))


def train(args, dataset_path, ckpt_path):
    """Training include resume training"""
    rank_size = int(os.environ.get("RANK_SIZE", 1))
    set_seed(1)
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()
    data_config.data_vocab_size = args.data_vocab_size
    model_config.data_vocab_size = args.data_vocab_size
    eval_file_name = "./auc.log"
    loss_file_name = "./loss.log"
    rank_size, rank_id = set_context(args=args, rank_size=rank_size)
    mox.file.shift('os', 'mox')
    mox.file.copy_parallel(args.data_url, dataset_path)
    ds_train = create_dataset(dataset_path,
                              train_mode=True,
                              epochs=1,
                              batch_size=args.batch_size,
                              data_type=DataType(data_config.data_format),
                              rank_size=rank_size,
                              rank_id=rank_id)

    steps_size = ds_train.get_dataset_size()
    if model_config.convert_dtype:
        model_config.convert_dtype = args.device_target != "CPU"
    train_config.learning_rate = args.learning_rate
    model_builder = ModelBuilder(model_config, train_config)
    train_net, eval_net = model_builder.get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(
        train_net,
        eval_network=eval_net,
        metrics={
            "auc": auc_metric})
    time_callback = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_callback = LossCallBack(loss_file_path=loss_file_name)
    callback_list = [time_callback, loss_callback]
    # resuem training
    if os.path.isfile(args.resume):
        resume_ckpt_name = args.resume.split('/')[-1]
        resume_dir = os.path.join(ckpt_path, "resume")
        if not os.path.exists(resume_dir):
            os.makedirs(resume_dir)
        resumed_ckpt = os.path.join(resume_dir, resume_ckpt_name)
        mox.file.copy(args.resume, resumed_ckpt)
        epoch = get_epoch(resume_ckpt_name)
        args.train_epochs = args.train_epochs - epoch
        param_dict = load_checkpoint(resumed_ckpt)
        load_param_into_net(train_net, param_dict)
        print('load_model {} success'.format(resumed_ckpt))
    else:
        print('{} not set/exists or not a pre-trained file'.format(args.resume))
    if train_config.save_checkpoint and rank_id in list([None, 0]):
        if rank_size:
            train_config.ckpt_file_name_prefix = train_config.ckpt_file_name_prefix + \
                str(get_rank())
            ckpt_path = os.path.join(
                ckpt_path, 'ckpt_' + str(get_rank()) + '/')
        if args.device_target != "Ascend":
            config_ck = CheckpointConfig(save_checkpoint_steps=steps_size,
                                         keep_checkpoint_max=train_config.keep_checkpoint_max)
        else:
            config_ck = CheckpointConfig(save_checkpoint_steps=train_config.save_checkpoint_steps,
                                         keep_checkpoint_max=train_config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=train_config.ckpt_file_name_prefix,
                                  directory=ckpt_path,
                                  config=config_ck)
        callback_list.append(ckpt_cb)
    if args.do_eval:
        ds_eval = create_dataset(dataset_path, train_mode=False,
                                 epochs=1,
                                 batch_size=args.batch_size,
                                 data_type=DataType(data_config.data_format))
        eval_callback = EvalCallBack(model, ds_eval, auc_metric,
                                     eval_file_path=eval_file_name)
        callback_list.append(eval_callback)
    print("start training")
    try:
        model.train(args.train_epochs, ds_train, callbacks=callback_list)
    finally:
        copy_files(args=args, rank_size=rank_size, eval_file_name=eval_file_name,
                   loss_file_name=loss_file_name)


def model_tran(args, dataset_path, ckpt_path):
    """transform ckpt file to air file"""
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.device_target,
        device_id=0)
    data_config = DataConfig()
    model_config = ModelConfig()
    data_config.batch_size = 1
    data_config.data_field_size = args.data_field_size
    model_config.batch_size = 1
    model_config.data_vocab_size = args.data_vocab_size
    deepfm_net = DeepFMModel(model_config)
    network = PredictWithSigmoid(deepfm_net)
    network.set_train(False)
    ckpt, _ = get_ckpt_epoch(ckpt_path)
    load_checkpoint(os.path.join(ckpt_path, ckpt), net=network)
    batch_ids = Tensor(np.zeros(
        [data_config.batch_size, data_config.data_field_size]).astype(np.int32))
    batch_wts = Tensor(np.zeros(
        [data_config.batch_size, data_config.data_field_size]).astype(np.float32))
    input_data = [batch_ids, batch_wts]
    export(
        network,
        *input_data,
        file_name="deepfm",
        file_format=args.file_format)
    mox.file.copy(
        "deepfm." +
        args.file_format.lower(),
        os.path.join(
            args.train_url,
            "deepfm." +
            args.file_format.lower()))
    print("export finished ")


if __name__ == '__main__':
    dataset_path = "/cache/dataset"
    ckpt_path = "/cache/output"

    train(args_opt, dataset_path, ckpt_path)

    rank_size = int(os.environ.get("RANK_SIZE", 1))
    rank_id = int(os.environ.get('RANK_ID'))

    if rank_size > 1:
        if rank_id == 0:
            ckpt_path = os.path.join(ckpt_path, "ckpt_0")
            model_tran(args_opt, dataset_path, ckpt_path)
    else:
        model_tran(args_opt, dataset_path, ckpt_path)

    print("{:*^30}".format("successful!"))
