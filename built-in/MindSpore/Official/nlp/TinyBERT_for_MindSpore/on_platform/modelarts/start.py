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
"""general distill script"""
from src.tinybert_model import BertModelCLS
from src.tinybert_for_gd_td import BertEvaluationWithLossScaleCell, BertNetworkWithLoss_td, BertEvaluationCell
from src.td_config import phase1_cfg, phase2_cfg, eval_cfg, td_teacher_net_cfg, td_student_net_cfg
from src.tinybert_for_gd_td import BertTrainWithLossScaleCell, BertNetworkWithLoss_gd, BertTrainCell
from src.gd_config import common_cfg, bert_teacher_net_cfg, bert_student_net_cfg
from src.assessment_method import Accuracy
from src.utils import LossCallBack, ModelSaveCkpt, EvalCallBack, BertLearningRate
from src.dataset import create_tinybert_dataset, DataType
from mindspore.common import set_seed
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from mindspore.train.callback import TimeMonitor
from mindspore.train.model import Model
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import numpy as np
import re
import datetime
import argparse
import moxing as mox
import os
import sys
base_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/..")
sys.path.append(base_path + "/../../")


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(
        description='tinybert task distill && general distill')
    parser.add_argument("--distill_type", type=str, default="td", choices=['gd', 'td'],
                        help='task distill or general distill')
    parser.add_argument("--device_target", type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")

    # td param
    parser.add_argument("--td_do_train", type=str, default="true", choices=["true", "false"],
                        help="Do train task, default is true.")
    parser.add_argument("--td_do_eval", type=str, default="true", choices=["true", "false"],
                        help="Do eval task, default is true.")
    parser.add_argument("--td_phase1_epoch_size", type=int, default=10,
                        help="Epoch size for td phase 1, default is 10.")
    parser.add_argument("--td_task_name", type=str, default="", choices=["SST-2", "QNLI", "MNLI"],
                        help="The name of the task to train.")
    parser.add_argument(
        "--td_phase2_epoch_size",
        type=int,
        default=3,
        help="Epoch size for td phase 2, default is 3.")
    parser.add_argument(
        "--td_load_teacher_ckpt_obs",
        type=str,
        default="",
        help="Load bert checkpoint file which fineturn in task dataset")
    parser.add_argument(
        "--td_load_gd_ckpt_obs",
        type=str,
        default="",
        help="td step:Load tinybert checkpoint file path which pretrained in gd")
    parser.add_argument(
        "--td_load_td1_ckpt_obs",
        type=str,
        default="",
        help="td step:Load tinybert checkpoint file path which finetuned in td, used to final eval step")

    # gd param
    # parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument(
        "--gd_epoch_size",
        type=int,
        default="3",
        help="Epoch size, default is 1.")
    parser.add_argument(
        "--gd_device_num",
        type=int,
        default=1,
        help="Use device nums, default is 1.")
    parser.add_argument(
        "--gd_save_ckpt_path",
        type=str,
        default="/tmp/ckpt_save/",
        help="Save checkpoint path")
    parser.add_argument(
        "--gd_resume_ckpt_obs",
        type=str,
        default="",
        help="resume Load checkpoint file path")
    parser.add_argument(
        "--gd_load_teacher_ckpt_obs",
        type=str,
        default="",
        help="Load bert checkpoint file path in gd step")

    parser.add_argument("--do_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--enable_data_sink", type=str, default="true", choices=["true", "false"],
                        help="Enable data sink, default is true.")
    parser.add_argument(
        "--save_ckpt_step",
        type=int,
        default=100,
        help="Steps to save checkpoint.")
    parser.add_argument(
        "--max_ckpt_num",
        type=int,
        default=1,
        help="Enable data sink, default is true.")
    parser.add_argument(
        "--data_sink_steps",
        type=int,
        default=1,
        help="Sink steps for each epoch, default is 1.used in gd")
    parser.add_argument(
        '--train_url',
        type=str,
        default="",
        help='the obs path model saved')
    parser.add_argument('--data_url', type=str, default="",
                        help='the obs path training data saved')
    # parser.add_argument("--gd_train_data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    # parser.add_argument("--td_train_data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    # parser.add_argument("--td_eval_data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    # parser.add_argument("--schema_dir", type=str, default="", help="Schema path, it is better to use absolute path")

    parser.add_argument("--dataset_type", type=str, default="tfrecord",
                        help="dataset type tfrecord/mindrecord, default is tfrecord")
    parser.add_argument(
        "--export_file_format",
        type=str,
        choices=[
            "AIR",
            "ONNX",
            "MINDIR"],
        default="AIR",
        help="file format")
    parser.add_argument(
        "--file_name",
        type=str,
        default="tinybert",
        help="output file name.")
    args = parser.parse_args()
    return args


args_opt = parse_args()

# create td ckpt save path
td_phase1_save_ckpt_dir = ""
td_phase2_save_ckpt_dir = ""

data_step_path = '/tmp/data/'
mox.file.copy_parallel(args_opt.data_url, data_step_path)
gd_train_data_dir = os.path.join(data_step_path, 'wiki')
td_train_data_dir = os.path.join(data_step_path, 'SST-2/train')
td_eval_data_dir = os.path.join(data_step_path, 'SST-2/eval')
DEFAULT_NUM_LABELS = 2
DEFAULT_SEQ_LENGTH = 128
task_params = {"SST-2": {"num_labels": 2, "seq_length": 128},
               "QNLI": {"num_labels": 2, "seq_length": 128},
               "MNLI": {"num_labels": 3, "seq_length": 128}}


class Task:
    """
    Encapsulation class of get the task parameter.
    """

    def __init__(self, task_name):
        self.task_name = task_name

    @property
    def num_labels(self):
        if self.task_name in task_params and "num_labels" in task_params[self.task_name]:
            return task_params[self.task_name]["num_labels"]
        return DEFAULT_NUM_LABELS

    @property
    def seq_length(self):
        if self.task_name in task_params and "seq_length" in task_params[self.task_name]:
            return task_params[self.task_name]["seq_length"]
        return DEFAULT_SEQ_LENGTH


task = Task(args_opt.td_task_name)

# --------gd code-------


def run_general_distill():
    """
    run general distill
    """

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args_opt.device_target,
        device_id=int(
            os.getenv('DEVICE_ID')))
    context.set_context(reserve_class_name_in_scope=False)
    context.set_context(variable_memory_max_size="30GB")

    save_ckpt_dir = os.path.join(args_opt.gd_save_ckpt_path,
                                 datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M'))

    rank = 0
    if args_opt.distribute == "true":
        if args_opt.device_target == 'Ascend':
            D.init()
            device_num = args_opt.gd_device_num
            rank = D.get_rank()
            save_ckpt_dir = save_ckpt_dir + '_ckpt_' + str(rank)
        else:
            D.init()
            device_num = D.get_group_size()
            rank = D.get_rank()
            save_ckpt_dir = save_ckpt_dir + '_ckpt_' + str(rank)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
    else:
        rank = 0
        device_num = 1

    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)

    enable_loss_scale = True
    if args_opt.device_target == "GPU":
        if bert_student_net_cfg.compute_type != mstype.float32:
            logger.warning(
                'Compute about the student only support float32 temporarily, run with float32.')
            bert_student_net_cfg.compute_type = mstype.float32
        # Backward of the network are calculated using fp32,
        # and the loss scale is not necessary
        enable_loss_scale = False
    # -----cp ckpt
    gd_resume_ckpt = ""
    if not args_opt.gd_resume_ckpt_obs == "":
        gd_resume_ckpt = os.path.join(
            '/tmp/', args_opt.gd_resume_ckpt_obs.split('/')[-1])
        mox.file.copy(args_opt.gd_resume_ckpt_obs, gd_resume_ckpt)
    if not args_opt.gd_load_teacher_ckpt_obs == "":
        gd_load_teacher_ckpt = os.path.join(
            '/tmp/', args_opt.gd_load_teacher_ckpt_obs.split('/')[-1])
        mox.file.copy(args_opt.gd_load_teacher_ckpt_obs, gd_load_teacher_ckpt)

    netwithloss = BertNetworkWithLoss_gd(teacher_config=bert_teacher_net_cfg,
                                         teacher_ckpt=gd_load_teacher_ckpt,
                                         student_config=bert_student_net_cfg,
                                         resume_ckpt=gd_resume_ckpt,
                                         is_training=True, use_one_hot_embeddings=False)

    if args_opt.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif args_opt.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        raise Exception("dataset format is not supported yet")
    dataset = create_tinybert_dataset('gd', common_cfg.batch_size, device_num, rank,
                                      args_opt.do_shuffle, gd_train_data_dir, None,
                                      data_type=dataset_type)
    dataset_size = dataset.get_dataset_size()
    print('dataset size: ', dataset_size)
    print("dataset repeatcount: ", dataset.get_repeat_count())
    if args_opt.enable_data_sink == "true":
        repeat_count = args_opt.gd_epoch_size * dataset_size // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.gd_epoch_size
        time_monitor_steps = dataset_size

    lr_schedule = BertLearningRate(learning_rate=common_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=common_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(
                                       dataset_size * args_opt.gd_epoch_size / 10),
                                   decay_steps=int(
                                       dataset_size * args_opt.gd_epoch_size),
                                   power=common_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(
        filter(
            common_cfg.AdamWeightDecay.decay_filter,
            params))
    other_params = list(
        filter(
            lambda x: not common_cfg.AdamWeightDecay.decay_filter(x),
            params))
    group_params = [{'params': decay_params, 'weight_decay': common_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(
        group_params,
        learning_rate=lr_schedule,
        eps=common_cfg.AdamWeightDecay.eps)

    callback = [TimeMonitor(time_monitor_steps), LossCallBack(), ModelSaveCkpt(netwithloss.bert,
                                                                               args_opt.save_ckpt_step,
                                                                               args_opt.max_ckpt_num,
                                                                               save_ckpt_dir)]
    if enable_loss_scale:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=common_cfg.loss_scale_value,
                                                 scale_factor=common_cfg.scale_factor,
                                                 scale_window=common_cfg.scale_window)
        netwithgrads = BertTrainWithLossScaleCell(
            netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    else:
        netwithgrads = BertTrainCell(netwithloss, optimizer=optimizer)
    model = Model(netwithgrads)
    model.train(repeat_count, dataset, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == "true"),
                sink_size=args_opt.data_sink_steps)

# --------td code-----------


def run_predistill():
    """
    run predistill
    """
    cfg = phase1_cfg
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args_opt.device_target,
        device_id=int(
            os.getenv('DEVICE_ID')))
    context.set_context(reserve_class_name_in_scope=False)
    # ------cp ckpt
    if not args_opt.td_load_teacher_ckpt_obs == "":
        td_load_teacher_ckpt_path = os.path.join(
            '/tmp/', args_opt.td_load_teacher_ckpt_obs.split('/')[-1])
        mox.file.copy(
            args_opt.td_load_teacher_ckpt_obs,
            td_load_teacher_ckpt_path)
    if not args_opt.td_load_gd_ckpt_obs == "":
        td_load_gd_ckpt_path = os.path.join(
            '/tmp/', args_opt.td_load_gd_ckpt_obs.split('/')[-1])
        mox.file.copy(args_opt.td_load_gd_ckpt_obs, td_load_gd_ckpt_path)

    load_teacher_checkpoint_path = td_load_teacher_ckpt_path
    load_student_checkpoint_path = td_load_gd_ckpt_path
    netwithloss = BertNetworkWithLoss_td(teacher_config=td_teacher_net_cfg, teacher_ckpt=load_teacher_checkpoint_path,
                                         student_config=td_student_net_cfg, student_ckpt=load_student_checkpoint_path,
                                         is_training=True, task_type='classification',
                                         num_labels=task.num_labels, is_predistill=True)

    rank = 0
    device_num = 1

    if args_opt.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif args_opt.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        raise Exception("dataset format is not supported yet")
    dataset = create_tinybert_dataset('td', cfg.batch_size,
                                      device_num, rank, args_opt.do_shuffle,
                                      td_train_data_dir, None,
                                      data_type=dataset_type)

    dataset_size = dataset.get_dataset_size()
    print('td1 dataset size: ', dataset_size)
    print('td1 dataset repeatcount: ', dataset.get_repeat_count())
    if args_opt.enable_data_sink == 'true':
        repeat_count = args_opt.td_phase1_epoch_size * \
            dataset_size // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.td_phase1_epoch_size
        time_monitor_steps = dataset_size

    optimizer_cfg = cfg.optimizer_cfg

    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size / 10),
                                   decay_steps=int(
                                       dataset_size * args_opt.td_phase1_epoch_size),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(
        filter(
            optimizer_cfg.AdamWeightDecay.decay_filter,
            params))
    other_params = list(
        filter(
            lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x),
            params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(
        group_params,
        learning_rate=lr_schedule,
        eps=optimizer_cfg.AdamWeightDecay.eps)
    callback = [TimeMonitor(time_monitor_steps), LossCallBack(), ModelSaveCkpt(netwithloss.bert,
                                                                               args_opt.save_ckpt_step,
                                                                               args_opt.max_ckpt_num,
                                                                               td_phase1_save_ckpt_dir)]
    if enable_loss_scale:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)
        netwithgrads = BertEvaluationWithLossScaleCell(
            netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    else:
        netwithgrads = BertEvaluationCell(netwithloss, optimizer=optimizer)

    model = Model(netwithgrads)
    model.train(repeat_count, dataset, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == 'true'),
                sink_size=args_opt.data_sink_steps)


def run_task_distill(ckpt_file):
    """
    run task distill
    """
    if ckpt_file == '':
        raise ValueError("Student ckpt file should not be None")
    cfg = phase2_cfg
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args_opt.device_target,
        device_id=int(
            os.getenv('DEVICE_ID')))

    if not args_opt.td_load_teacher_ckpt_obs == "":
        td_load_teacher_ckpt_path = os.path.join(
            '/tmp/', args_opt.td_load_teacher_ckpt_obs.split('/')[-1])
        mox.file.copy(
            args_opt.td_load_teacher_ckpt_obs,
            td_load_teacher_ckpt_path)

    load_teacher_checkpoint_path = td_load_teacher_ckpt_path
    load_student_checkpoint_path = ckpt_file
    netwithloss = BertNetworkWithLoss_td(teacher_config=td_teacher_net_cfg, teacher_ckpt=load_teacher_checkpoint_path,
                                         student_config=td_student_net_cfg, student_ckpt=load_student_checkpoint_path,
                                         is_training=True, task_type='classification',
                                         num_labels=task.num_labels, is_predistill=False)

    rank = 0
    device_num = 1
    train_dataset = create_tinybert_dataset('td', cfg.batch_size,
                                            device_num, rank, args_opt.do_shuffle,
                                            td_train_data_dir, None)

    dataset_size = train_dataset.get_dataset_size()
    print('td2 train dataset size: ', dataset_size)
    print('td2 train dataset repeatcount: ', train_dataset.get_repeat_count())
    if args_opt.enable_data_sink == 'true':
        repeat_count = args_opt.td_phase2_epoch_size * \
            train_dataset.get_dataset_size() // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.td_phase2_epoch_size
        time_monitor_steps = dataset_size

    optimizer_cfg = cfg.optimizer_cfg

    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(
                                       dataset_size * args_opt.td_phase2_epoch_size / 10),
                                   decay_steps=int(
                                       dataset_size * args_opt.td_phase2_epoch_size),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(
        filter(
            optimizer_cfg.AdamWeightDecay.decay_filter,
            params))
    other_params = list(
        filter(
            lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x),
            params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(
        group_params,
        learning_rate=lr_schedule,
        eps=optimizer_cfg.AdamWeightDecay.eps)

    eval_dataset = create_tinybert_dataset('td', eval_cfg.batch_size,
                                           device_num, rank, args_opt.do_shuffle,
                                           td_eval_data_dir, None)
    print('td2 eval dataset size: ', eval_dataset.get_dataset_size())

    if args_opt.td_do_eval.lower() == "true":
        callback = [TimeMonitor(time_monitor_steps), LossCallBack(),
                    EvalCallBack(netwithloss.bert, eval_dataset)]
    else:
        callback = [TimeMonitor(time_monitor_steps), LossCallBack(),
                    ModelSaveCkpt(netwithloss.bert,
                                  args_opt.save_ckpt_step,
                                  args_opt.max_ckpt_num,
                                  td_phase2_save_ckpt_dir)]
    if enable_loss_scale:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)

        netwithgrads = BertEvaluationWithLossScaleCell(
            netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    else:
        netwithgrads = BertEvaluationCell(netwithloss, optimizer=optimizer)
    model = Model(netwithgrads)
    model.train(repeat_count, train_dataset, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == 'true'),
                sink_size=args_opt.data_sink_steps)


def do_eval_standalone():
    """
    do eval standalone
    """
    ckpt_file = os.path.join(
        '/tmp/', args_opt.td_load_td1_ckpt_obs.split('/')[-1])
    mox.file.copy(args_opt.td_load_td1_ckpt_obs, ckpt_file)

    if ckpt_file == '':
        raise ValueError("Student ckpt file should not be None")
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args_opt.device_target,
        device_id=int(
            os.getenv('DEVICE_ID')))
    eval_model = BertModelCLS(
        td_student_net_cfg,
        False,
        task.num_labels,
        0.0,
        phase_type="student")
    param_dict = load_checkpoint(ckpt_file)
    new_param_dict = {}
    for key, value in param_dict.items():
        new_key = re.sub('tinybert_', 'bert_', key)
        new_key = re.sub('^bert.', '', new_key)
        new_param_dict[new_key] = value
    load_param_into_net(eval_model, new_param_dict)
    eval_model.set_train(False)

    eval_dataset = create_tinybert_dataset('td', batch_size=eval_cfg.batch_size,
                                           device_num=1, rank=0, do_shuffle="false",
                                           data_dir=td_eval_data_dir,
                                           schema_dir=None)
    print('eval dataset size: ', eval_dataset.get_dataset_size())
    print('eval dataset batch size: ', eval_dataset.get_batch_size())

    callback = Accuracy()
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in eval_dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits = eval_model(input_ids, token_type_id, input_mask)
        callback.update(logits[3], label_ids)
    acc = callback.acc_num / callback.total_num
    print("======================================")
    print("============== acc is {}".format(acc))
    print("======================================")

# -------export code-----------


def model_export():
    td_student_net_cfg.seq_length = task.seq_length
    td_student_net_cfg.batch_size = 1
    eval_model = BertModelCLS(
        td_student_net_cfg,
        False,
        task.num_labels,
        0.0,
        phase_type="student")
    param_dict = load_checkpoint('eval_model.ckpt')
    new_param_dict = {}
    for key, value in param_dict.items():
        new_key = re.sub('tinybert_', 'bert_', key)
        new_key = re.sub('^bert.', '', new_key)
        new_param_dict[new_key] = value

    load_param_into_net(eval_model, new_param_dict)
    eval_model.set_train(False)

    input_ids = Tensor(
        np.zeros(
            (td_student_net_cfg.batch_size,
             task.seq_length),
            np.int32))
    token_type_id = Tensor(
        np.zeros(
            (td_student_net_cfg.batch_size,
             task.seq_length),
            np.int32))
    input_mask = Tensor(
        np.zeros(
            (td_student_net_cfg.batch_size,
             task.seq_length),
            np.int32))

    input_data = [input_ids, token_type_id, input_mask]

    export(eval_model, *input_data, file_name="tinybert",
           file_format=args_opt.export_file_format)
    print("export complete!!!!")
    mox.file.copy(
        "tinybert.air",
        os.path.join(
            args_opt.train_url,
            "tinybert.air"))


if __name__ == '__main__':
    if args_opt.distill_type == 'gd':
        set_seed(0)

        run_general_distill()
        mox.file.copy_parallel(args_opt.gd_save_ckpt_path, args_opt.train_url)
    elif args_opt.distill_type == 'td':

        td_phase1_save_ckpt_dir = os.path.join(
            args_opt.gd_save_ckpt_path,
            'tinybert_td_phase1_save_ckpt')
        td_phase2_save_ckpt_dir = os.path.join(
            args_opt.gd_save_ckpt_path,
            'tinybert_td_phase2_save_ckpt')
        if not os.path.exists(td_phase1_save_ckpt_dir):
            os.makedirs(td_phase1_save_ckpt_dir)
        if not os.path.exists(td_phase2_save_ckpt_dir):
            os.makedirs(td_phase2_save_ckpt_dir)

        if args_opt.td_do_train.lower() != "true" and args_opt.td_do_eval.lower() != "true":
            raise ValueError(
                "do_train or do eval must have one be true, please confirm your config")

        enable_loss_scale = True
        if args_opt.device_target == "GPU":
            if td_student_net_cfg.compute_type != mstype.float32:
                logger.warning(
                    'Compute about the student only support float32 temporarily, run with float32.')
                td_student_net_cfg.compute_type = mstype.float32
            # Backward of the network are calculated using fp32,
            # and the loss scale is not necessary
            enable_loss_scale = False

        td_teacher_net_cfg.seq_length = task.seq_length
        td_student_net_cfg.seq_length = task.seq_length
        if args_opt.td_do_train == "true":
            # run predistill
            run_predistill()
            lists = os.listdir(td_phase1_save_ckpt_dir)
            if lists:
                lists.sort(
                    key=lambda fn: os.path.getmtime(
                        td_phase1_save_ckpt_dir + '/' + fn))
                name_ext = os.path.splitext(lists[-1])
                if name_ext[-1] != ".ckpt":
                    raise ValueError(
                        "Invalid file, checkpoint file should be .ckpt file")
                newest_ckpt_file = os.path.join(
                    td_phase1_save_ckpt_dir, lists[-1])
                # run task distill
                run_task_distill(newest_ckpt_file)
                mox.file.copy_parallel(
                    args_opt.gd_save_ckpt_path, args_opt.train_url)
                mox.file.copy(
                    "eval_model.ckpt",
                    os.path.join(
                        args_opt.train_url,
                        "eval_model.ckpt"))
                mox.file.copy(
                    "eval.log", os.path.join(
                        args_opt.train_url, "eval.log"))
                model_export()
            else:
                raise ValueError(
                    "Checkpoint file not exists, please make sure ckpt file has been saved")
        else:
            do_eval_standalone()
