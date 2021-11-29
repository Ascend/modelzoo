# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

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
import os
import sys
import time
import moxing as mox
import importlib
import subprocess
import logging
import argparse
import json


parser = argparse.ArgumentParser("Vega")
parser.add_argument('--config_file', type=str, default='nas/backbone_nas/backbone_nas_tf.yml', help='config file name')
parser.add_argument('--base_dir', type=str, default='/home/work/user-job-dir/automl/examples', help='base dir')
parser.add_argument('--cifar10', action='store_true', default=False, help='cifar10 dataset')
parser.add_argument('--imagenet', action='store_true', default=False, help='imagenet dataset')
parser.add_argument('--VOC2012', action='store_true', default=False, help='VOC2012 dataset')
parser.add_argument('--prune_model', action='store_true', default=False, help='prune pretrained model')
args, unknown_args = parser.parse_known_args()


def copy_dataset():
    if args.cifar10:
        logging.info('downloading cifar10 dataset')
        mox.file.copy_parallel('s3://bucket-auto2/xiaozhi/Dataset/cifar-10_bin/cifar-10-batches-bin',
                               '/cache/datasets/cifar-10-batches-bin')
    if args.imagenet:
        logging.info('downloading imagenet dataset')
        mox.file.copy_parallel('s3://bucket-auto2/xiaozhi/Dataset/imagenet_tfrecord',
                               '/cache/datasets/imagenet_tfrecord')
        os.system('cd /cache/datasets/imagenet_tfrecord && tar -xf imagenet_tfrecord.tar.gz')
    if args.VOC2012:
        logging.info('downloading VOC2012 dataset')
        mox.file.copy_parallel('s3://bucket-auto2/xiaozhi/Dataset/VOC2012/',
                               '/cache/datasets/VOC2012')
    if args.prune_model:
        logging.info('downloading prune pretrained model')
        mox.file.copy_parallel('s3://bucket-auto2/xiaozhi/Models/prune_ea/', '/cache/models/prune/')
    # mox.file.copy_parallel('s3://bucket-auto2/xiaozhi/Models/resnet_nas',
    #                        '/cache/models/resnet_nas')
    # mox.file.copy_parallel('s3://bucket-auto2/xiaozhi/Results/vega/tf_trainer_d/resnet_variant/ft/0624.154232.420/workers/fully_train/4/3/',
    #                        '/cache/model/resnet_fully_train')
    # mox.file.copy_parallel('s3://bucket-auto2/xiaozhi/Results/vega/tf_trainer_d/resnet_variant/ft/0624.023812.753/workers/fully_train/2/0/',
    #                        '/cache/model/resnet_fully_train')


def first_device_start(config_file):
    os.system('mkdir -p /cache/workspace/device0')
    device_id = int(os.environ.get('DEVICE_ID'))
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    if device_id == 0:
        print(os.environ)
        copy_dataset()
        os.system('bash /home/work/user-job-dir/automl/examples/cloud_run.sh {}'.format(config_file))
        try:
            if args.config_file == 'nas/backbone_nas/backbone_nas_tf.yml':
                mox.file.copy_parallel('/device', 's3://bucket-auto2/xiaozhi/Debug/resnet/device/' + os.environ.get('BATCH_JOB_ID'))
            elif args.config_file == 'nas/darts_cnn/darts_tf.yml':
                mox.file.copy_parallel('/cache/workspace/device0', 's3://bucket-auto2/xiaozhi/Debug/darts_cnn/graph/' + os.environ.get('BATCH_JOB_ID'))
                mox.file.copy_parallel('/device', 's3://bucket-auto2/xiaozhi/Debug/darts_cnn/device/' + os.environ.get('BATCH_JOB_ID'))
            elif args.config_file == 'nas/adelaide_ea/adelaide_ea_tf.yml':
                mox.file.copy_parallel('/cache/workspace/device0', 's3://bucket-auto2/xiaozhi/Debug/adelaide/graph/' + os.environ.get('BATCH_JOB_ID'))
                mox.file.copy_parallel('/device', 's3://bucket-auto2/xiaozhi/Debug/adelaide/device/' + os.environ.get('BATCH_JOB_ID'))
        except Exception:
            pass


if __name__ == "__main__":
    # base_dir = '/home/work/user-job-dir/automl/examples/nas'
    # config_file = 'backbone_nas/backbone_nas_tf.yml'
    # config_file = 'simple_cnn/simple_cnn.yml'
    # config_file = 'darts_cnn/darts_tf.yml'
    base_dir = args.base_dir
    config_file = args.config_file
    config_file = os.path.join(base_dir, config_file)
    first_device_start(config_file)
