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
"""This is the founction of environment variables."""
import os
import argparse
from vega.core.run import env_args


def register_cluster_args():
    """Get the environment variables of the cluster."""
    parser = argparse.ArgumentParser("prune")
    parser.add_argument('--data_url', type=str, default=None,
                        help='s3 path of dataset')
    parser.add_argument('--train_url', type=str, default=None,
                        help='s3 path of outputs')
    parser.add_argument('--init_method', type=str, default=None,
                        help='master address')
    parser.add_argument('--rank', type=int, default=0,
                        help='Index of current task')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of tasks')
    parser.add_argument('--eval_count', type=int, default=10,
                        help='Total number of eval_count')
    args, _ = parser.parse_known_args()

    if 'NPU-VISIBLE-DEVICES' in os.environ:
        world_size = int(os.environ['BATCH_TASK_REPLICAS'])
        rank = int(os.environ['BATCH_TASK_INDEX'])
        batch_job_hosts = 'BATCH_' + os.environ['BATCH_GROUP_NAME'].upper() + '_HOSTS'
        master_address = os.environ[batch_job_hosts].split(',')[0]
        init_method = 'tcp://' + master_address
        args.init_method = init_method
        args.rank = rank
        args.world_size = world_size
    else:
        data_url = os.environ['DLS_DATA_URL']
        train_url = os.environ['DLS_TRAIN_URL']
        world_size = int(os.environ['DLS_TASK_NUMBER'])
        if world_size == 1:
            init_method = 'tcp://' + os.environ['BATCH_CURRENT_HOST']
            rank = 0
        else:
            init_method = 'tcp://' + os.environ['BATCH_CUSTOM0_HOSTS']
            rank = int(os.environ['DLS_TASK_INDEX'])
        args.data_url = data_url
        args.train_url = train_url
        args.init_method = init_method
        args.rank = rank
        args.world_size = world_size

    env_args(args)
