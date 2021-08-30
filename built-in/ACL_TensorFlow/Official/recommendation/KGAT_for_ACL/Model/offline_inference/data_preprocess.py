# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import numpy as np
import sys
sys.path.append('./')
from utility.loader_kgat import KGAT_loader
from utility.parser import parse_args

if __name__ == '__main__':

    args = parse_args()

    if os.path.isdir(args.output_path):
        os.system("rm -rf " + args.output_path)
    os.makedirs(args.output_path)
    os.makedirs(args.output_path+"/input1")
    os.makedirs(args.output_path+"/input2")
    os.makedirs(args.output_path+"/input3")

    BATCH_SIZE = args.batch_size

    u_batch_size = BATCH_SIZE * 2

    data_generator = KGAT_loader(args=args, path=args.data_path + args.dataset)

    users_to_test = list(data_generator.test_user_dict.keys())
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size  # + 1

    count = 0
    USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        item_batch = range(ITEM_NUM)

        # save to bin file
        np.array(user_batch, dtype=np.int32).tofile(args.output_path + '/input1/users_' + str(u_batch_id).zfill(5) + '.bin')
        np.array(item_batch, dtype=np.int32).tofile(args.output_path + '/input2/pos_items_' + str(u_batch_id).zfill(5) + '.bin')
        np.array([0.] * len(eval(args.layer_size)), dtype=np.float32).tofile(args.output_path + '/input3/node_dropout_' + str(u_batch_id).zfill(5) + '.bin')

    print("[info]  data bin ok")
