# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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
"""train deeplabv3."""

import argparse
import os
import subprocess

_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"


def _parse_args():
    parser = argparse.ArgumentParser('mindspore deeplabv3 training')
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')

    # dataset
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--crop_size', type=int, default=513, help='crop size')
    parser.add_argument('--min_scale', type=float, default=0.5,
                        help='minimum scale of data argumentation')
    parser.add_argument('--max_scale', type=float, default=2.0,
                        help='maximum scale of data argumentation')
    parser.add_argument('--ignore_label', type=int, default=255,
                        help='ignore label')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='number of classes')

    # optimizer
    parser.add_argument('--train_epochs', type=int, default=200, help='epoch')
    parser.add_argument('--lr_type', type=str, default='cos',
                        help='type of learning rate')
    parser.add_argument('--base_lr', type=float, default=0.015,
                        help='base learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=40000,
                        help='learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='learning rate decay rate')
    parser.add_argument('--loss_scale', type=float, default=3072.0,
                        help='loss scale')

    # model
    parser.add_argument('--model', type=str, default='deeplab_v3_s16',
                        help='select model')
    parser.add_argument('--freeze_bn', action='store_true', help='freeze bn')
    parser.add_argument('--ckpt_pre_trained', type=str, default='',
                        help='pretrained model')

    # train
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'CPU'],
                        help='device where the code will be implemented. '
                             '(Default: Ascend)')
    parser.add_argument('--is_distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--rank', type=int, default=0,
                        help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1,
                        help='world size of distributed')
    parser.add_argument('--save_steps', type=int, default=1500,
                        help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=200,
                        help='max checkpoint for saving')
    parser.add_argument('--filter_weight', type=str, default="",
                        help="filter weight")

    args, _ = parser.parse_known_args()
    return args


def _train(args, train_url, data_url, ckpt_pre_trained):
    mindrecord_path = os.path.join(data_url, "dataset.mindrecord")
    train_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "train.py")
    cmd = ["python", train_file,
           f"--train_dir={os.path.abspath(train_url)}",
           f"--data_file={os.path.abspath(mindrecord_path)}",
           f"--batch_size={args.batch_size}",
           f"--crop_size={args.crop_size}",
           f"--min_scale={args.min_scale}",
           f"--max_scale={args.max_scale}",
           f"--ignore_label={args.ignore_label}",
           f"--num_classes={args.num_classes}",
           f"--train_epochs={args.train_epochs}",
           f"--lr_type={args.lr_type}",
           f"--base_lr={args.base_lr}",
           f"--lr_decay_step={args.lr_decay_step}",
           f"--lr_decay_rate={args.lr_decay_rate}",
           f"--loss_scale={args.loss_scale}",
           f"--model={args.model}",
           f"--ckpt_pre_trained={ckpt_pre_trained}",
           f"--device_target={args.device_target}",
           f"--rank={args.rank}",
           f"--group_size={args.group_size}",
           f"--save_steps={args.save_steps}",
           f"--keep_checkpoint_max={args.keep_checkpoint_max}"]
    if args.freeze_bn:
        cmd.append('--freeze_bn')
    if args.is_distributed:
        cmd.append('--is_distributed')
    if args.filter_weight == "True":
        cmd.append('--filter_weight=True')
    print(' '.join(cmd))
    os.environ["DEVICE_ID"] = str(args.rank)
    process = subprocess.Popen(cmd, shell=False)
    return process.wait()


def main():
    args = _parse_args()
    try:
        import moxing as mox
        os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
        os.makedirs(_CACHE_DATA_URL, exist_ok=True)
        mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
        train_url = _CACHE_TRAIN_URL
        data_url = _CACHE_DATA_URL
        ckpt_pre_trained = os.path.join(_CACHE_DATA_URL,
                                        args.ckpt_pre_trained) \
            if args.ckpt_pre_trained else ""
        ret = _train(args, train_url, data_url, ckpt_pre_trained)
        mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)
    except ModuleNotFoundError:
        train_url = args.train_url
        data_url = args.data_url
        ckpt_pre_trained = args.ckpt_pre_trained
        ret = _train(args, train_url, data_url, ckpt_pre_trained)

    if ret != 0:
        exit(1)


if __name__ == '__main__':
    main()
