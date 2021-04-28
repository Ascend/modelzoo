# coding=utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import ast
import os
import sys
from os.path import join, dirname, realpath

import moxing as mox

# 将modelarts上级目录加入PYTHONPATH
sys.path.append(join(dirname(realpath(__file__)), os.path.pardir))
from src import config as src_config


CACHE_TRAIN_DATA_URL = "/cache/train_data_url"
CACHE_TRAIN_OUT_URL = "/cache/train_out_url"


def get_special_args_for_modelarts(args):
    data_dir = CACHE_TRAIN_DATA_URL
    pre_trained = args.pre_trained
    if pre_trained and not pre_trained.startswith(data_dir):
        # 需要将预训练模型放在数据目录中
        pre_trained = os.path.join(data_dir, args.pre_trained)

    return {
        'pre_trained': pre_trained,
    }


def update_argparse_args(args, params):
    args.__dict__.update(params)


def parse_args():
    parser = argparse.ArgumentParser(description="SSD training")
    # 模型输出目录
    parser.add_argument("--train_url",
                        type=str, default='', help='the path model saved')
    # 数据集目录
    parser.add_argument("--data_url",
                        type=str, default='', help='the training data')

    # 原参数
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                        help="run platform, support Ascend, GPU and CPU.")
    parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False,
                        help="If set it true, only create Mindrecord, default is False.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate, default is 0.05.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--epoch_size", type=int, default=5, help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default=None, help="Pretrained Checkpoint file path.")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size.")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=10, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    parser.add_argument('--freeze_layer', type=str, default="none", choices=["none", "backbone"],
                        help="freeze the weights of network, support freeze the backbone's weights, "
                             "default is not freezing.")
    parser.add_argument("--run_eval", type=ast.literal_eval, default=False,
                        help="Run evaluation when training, default is False.")
    parser.add_argument("--save_best_ckpt", type=ast.literal_eval, default=True,
                        help="Save best checkpoint when run_eval is True, default is True.")
    parser.add_argument("--eval_start_epoch", type=int, default=40,
                        help="Evaluation start epoch when run_eval is True, default is 40.")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval when run_eval is True, default is 1.")

    # 适配config.py中的参数
    parser.add_argument("--using_model", type=str, default='ssd_mobilenet_v1_fpn',
                        choices=['ssd300', 'ssd_vgg16', 'ssd_mobilenet_v1_fpn',
                                 'ssd_resnet50_fpn'])
    parser.add_argument("--feature_extractor_base_param", type=str, default="")
    parser.add_argument("--coco_root", type=str, default="")
    parser.add_argument("--classes_label_path", type=str, default="labels.txt")
    parser.add_argument("--num_classes", type=int, default=81)
    parser.add_argument("--voc_root", type=str, default="")
    parser.add_argument("--voc_json", type=str, default="")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--anno_path", type=str, default="coco_labels.txt")

    args_opt = parser.parse_args()
    # 更新命令行参数
    update_argparse_args(args_opt, get_special_args_for_modelarts(args_opt))

    return args_opt


def update_config(args_opts):
    """
    补全在config中的数据集路径
    Args:
        args_opts:
        config:

    Returns:

    """
    src_config.using_model = args_opts.using_model
    config = src_config.config_map[args_opts.using_model]
    if config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * \
                   config.num_default[i]
        config.num_ssd_boxes = num

    data_dir = CACHE_TRAIN_DATA_URL

    # mindrecord格式数据集路径 更新为选择的数据集路径
    config.mindrecord_dir = data_dir

    # 补全数据集路径
    dataset = args_opts.dataset
    if dataset == 'coco':
        coco_root = args_opts.coco_root
        config.coco_root = os.path.join(data_dir, coco_root)
        print(f"update config.coco_root {coco_root} to {config.coco_root}")
    elif dataset == 'voc':
        voc_root = args_opts.voc_root
        config.voc_root = os.path.join(data_dir, voc_root)
        print(f"update config.voc_root {voc_root} to {config.voc_root}")
    else:
        image_dir = args_opts.image_dir
        anno_path = args_opts.anno_path
        config.image_dir = os.path.join(data_dir, image_dir)
        config.anno_path = os.path.join(data_dir, anno_path)
        print(f"update config.image_dir {image_dir} to {config.image_dir}")
        print(f"update config.anno_path {anno_path} to {config.anno_path}")

    with open(os.path.join(data_dir, args_opts.classes_label_path), 'r') as f:
        config.classes = [line.strip() for line in f.readlines()]
    config.num_classes = args_opts.num_classes

    # 补全预训练模型路径
    feature_extractor_base_param = args_opts.feature_extractor_base_param
    if args_opts.pre_trained:
        # 迁移学习不需要该参数
        config.feature_extractor_base_param = ""
        print('update config.feature_extractor_base_param to "" on pretrain.')
    elif feature_extractor_base_param:
        # 需要将预训练模型放在数据目录中
        config.feature_extractor_base_param = os.path.join(
            data_dir, feature_extractor_base_param)
        print(f"update config.feature_extractor_base_param "
              f"{feature_extractor_base_param} to "
              f"{config.feature_extractor_base_param}")
    src_config.config.clear()
    src_config.config.update(config)
    print(f"config: {src_config.config}")


def init():
    args = parse_args()
    os.makedirs(CACHE_TRAIN_DATA_URL, exist_ok=True)
    mox.file.copy_parallel(args.data_url, CACHE_TRAIN_DATA_URL)
    update_config(args)


init()