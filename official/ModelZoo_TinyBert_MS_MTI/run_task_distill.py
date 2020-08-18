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

"""task distill script"""

import os
import re
import argparse
from mindspore import Tensor
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay
from src.dataset import create_tinybert_dataset
from src.utils import LossCallBack, ModelSaveCkpt, EvalCallBack, BertLearningRate
from src.assessment_method import Accuracy
from src.td_config import phase1_cfg, phase2_cfg, td_teacher_net_cfg, td_student_net_cfg
from src.tinybert_for_gd_td import BertEvaluationCell, BertNetworkWithLoss_td
from src.tinybert_model import BertModelCLS

_cur_dir = os.getcwd()
td_phase1_save_ckpt_dir = os.path.join(_cur_dir, 'tinybert_td_phase1_save_ckpt')
td_phase2_save_ckpt_dir = os.path.join(_cur_dir, 'tinybert_td_phase2_save_ckpt')
if not os.path.exists(td_phase1_save_ckpt_dir):
    os.makedirs(td_phase1_save_ckpt_dir)
if not os.path.exists(td_phase2_save_ckpt_dir):
    os.makedirs(td_phase2_save_ckpt_dir)

def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='tinybert task distill')
    parser.add_argument("--device_target", type=str, default="Ascend", help="NPU device, default is Ascend.")
    parser.add_argument("--do_train", type=str, default="true", help="Do train task, default is true.")
    parser.add_argument("--do_eval", type=str, default="true", help="Do eval task, default is true.")
    parser.add_argument("--td_phase1_epoch_size", type=int, default=10,
                        help="Epoch size for td phase 1, default is 10.")
    parser.add_argument("--td_phase2_epoch_size", type=int, default=3, help="Epoch size for td phase 2, default is 3.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--num_labels", type=int, default=2, help="Classfication task, support SST2, QNLI, MNLI.")
    parser.add_argument("--do_shuffle", type=str, default="true", help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--enable_data_sink", type=str, default="true", help="Enable data sink, default is true.")
    parser.add_argument("--save_ckpt_step", type=int, default=100, help="Enable data sink, default is true.")
    parser.add_argument("--max_ckpt_num", type=int, default=1, help="Enable data sink, default is true.")
    parser.add_argument("--data_sink_steps", type=int, default=1, help="Sink steps for each epoch, default is 1.")
    parser.add_argument("--load_teacher_ckpt_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--load_gd_ckpt_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--load_td1_ckpt_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_dir", type=str, default="", help="Schema path, it is better to use absolute path")

    args = parser.parse_args()
    return args

args_opt = parse_args()
def run_predistill():
    """
    run predistill
    """
    cfg = phase1_cfg
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    context.set_context(reserve_class_name_in_scope=False)
    load_teacher_checkpoint_path = args_opt.load_teacher_ckpt_path
    load_student_checkpoint_path = args_opt.load_gd_ckpt_path
    netwithloss = BertNetworkWithLoss_td(teacher_config=td_teacher_net_cfg, teacher_ckpt=load_teacher_checkpoint_path,
                                         student_config=td_student_net_cfg, student_ckpt=load_student_checkpoint_path,
                                         is_training=True, task_type='classification',
                                         num_labels=args_opt.num_labels, is_predistill=True)

    rank = 0
    device_num = 1
    dataset = create_tinybert_dataset('td', td_teacher_net_cfg.batch_size,
                                      device_num, rank, args_opt.do_shuffle,
                                      args_opt.train_data_dir, args_opt.schema_dir)

    dataset_size = dataset.get_dataset_size()
    if args_opt.enable_data_sink == 'true':
        repeat_count = args_opt.td_phase1_epoch_size * dataset.get_dataset_size() // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.td_phase1_epoch_size
        time_monitor_steps = dataset_size

    optimizer_cfg = cfg.optimizer_cfg

    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size / 10),
                                   decay_steps=int(dataset_size * args_opt.td_phase1_epoch_size),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: x not in decay_params, params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    callback = [TimeMonitor(time_monitor_steps), LossCallBack(), ModelSaveCkpt(netwithloss.bert,
                                                                               args_opt.save_ckpt_step,
                                                                               args_opt.max_ckpt_num,
                                                                               td_phase1_save_ckpt_dir)]
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                             scale_factor=cfg.scale_factor,
                                             scale_window=cfg.scale_window)
    netwithgrads = BertEvaluationCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
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
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    load_teacher_checkpoint_path = args_opt.load_teacher_ckpt_path
    load_student_checkpoint_path = ckpt_file
    netwithloss = BertNetworkWithLoss_td(teacher_config=td_teacher_net_cfg, teacher_ckpt=load_teacher_checkpoint_path,
                                         student_config=td_student_net_cfg, student_ckpt=load_student_checkpoint_path,
                                         is_training=True, task_type='classification',
                                         num_labels=args_opt.num_labels, is_predistill=False)

    rank = 0
    device_num = 1
    train_dataset = create_tinybert_dataset('td', td_teacher_net_cfg.batch_size,
                                            device_num, rank, args_opt.do_shuffle,
                                            args_opt.train_data_dir, args_opt.schema_dir)

    dataset_size = train_dataset.get_dataset_size()
    if args_opt.enable_data_sink == 'true':
        repeat_count = args_opt.td_phase2_epoch_size * train_dataset.get_dataset_size() // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.td_phase2_epoch_size
        time_monitor_steps = dataset_size

    optimizer_cfg = cfg.optimizer_cfg

    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size * args_opt.td_phase2_epoch_size / 10),
                                   decay_steps=int(dataset_size * args_opt.td_phase2_epoch_size),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: x not in decay_params, params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)

    eval_dataset = create_tinybert_dataset('td', td_teacher_net_cfg.batch_size,
                                           device_num, rank, args_opt.do_shuffle,
                                           args_opt.eval_data_dir, args_opt.schema_dir)
    if args_opt.do_eval.lower() == "true":
        callback = [TimeMonitor(time_monitor_steps), LossCallBack(),
                    ModelSaveCkpt(netwithloss.bert,
                                  args_opt.save_ckpt_step,
                                  args_opt.max_ckpt_num,
                                  td_phase2_save_ckpt_dir),
                    EvalCallBack(netwithloss.bert, eval_dataset)]
    else:
        callback = [TimeMonitor(time_monitor_steps), LossCallBack(),
                    ModelSaveCkpt(netwithloss.bert,
                                  args_opt.save_ckpt_step,
                                  args_opt.max_ckpt_num,
                                  td_phase2_save_ckpt_dir)]
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                             scale_factor=cfg.scale_factor,
                                             scale_window=cfg.scale_window)

    netwithgrads = BertEvaluationCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    model.train(repeat_count, train_dataset, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == 'true'),
                sink_size=args_opt.data_sink_steps)

def do_eval_standalone():
    """
    do eval standalone
    """
    ckpt_file = args_opt.load_td1_ckpt_path
    if ckpt_file == '':
        raise ValueError("Student ckpt file should not be None")
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    eval_model = BertModelCLS(td_student_net_cfg, False, args_opt.num_labels, 0.0, phase_type="student")
    param_dict = load_checkpoint(ckpt_file)
    new_param_dict = {}
    for key, value in param_dict.items():
        new_key = re.sub('tinybert_', 'bert_', key)
        new_key = re.sub('^bert.', '', new_key)
        new_param_dict[new_key] = value
    load_param_into_net(eval_model, new_param_dict)
    eval_model.set_train(False)

    eval_dataset = create_tinybert_dataset('td', batch_size=1,
                                           device_num=1, rank=0, do_shuffle="false",
                                           data_dir=args_opt.eval_data_dir,
                                           schema_dir=args_opt.schema_dir)
    callback = Accuracy()
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in eval_dataset.create_dict_iterator():
        input_data = []
        for i in columns_list:
            input_data.append(Tensor(data[i]))
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits = eval_model(input_ids, token_type_id, input_mask)
        callback.update(logits[3], label_ids)
    acc = callback.acc_num / callback.total_num
    print("======================================")
    print("============== acc is {}".format(acc))
    print("======================================")

if __name__ == '__main__':
    if args_opt.do_train.lower() != "true" and args_opt.do_eval.lower() != "true":
        raise ValueError("do_train or do eval must have one be true, please confirm your config")
    if args_opt.do_train == "true":
        # run predistill
        run_predistill()
        lists = os.listdir(td_phase1_save_ckpt_dir)
        if lists:
            lists.sort(key=lambda fn: os.path.getmtime(td_phase1_save_ckpt_dir+'/'+fn))
            name_ext = os.path.splitext(lists[-1])
            if name_ext[-1] != ".ckpt":
                raise ValueError("Invalid file, checkpoint file should be .ckpt file")
            newest_ckpt_file = os.path.join(td_phase1_save_ckpt_dir, lists[-1])
            # run task distill
            run_task_distill(newest_ckpt_file)
        else:
            raise ValueError("Checkpoint file not exists, please make sure ckpt file has been saved")
    else:
        do_eval_standalone()
