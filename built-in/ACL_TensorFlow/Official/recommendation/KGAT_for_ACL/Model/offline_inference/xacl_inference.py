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
import multiprocessing
import sys
sys.path.append('./')
from utility.batch_test import cores, Ks, BATCH_SIZE, ITEM_NUM, data_generator, test_one_user
from utility.parser import parse_args


def xaclPath(output_path, inference_path, model_path):
    """
    使用文件夹推理
    """
    if os.path.isdir(inference_path):
        os.system("rm -rf " + inference_path)
    os.makedirs(inference_path)
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path_lst = [output_path + "input1", output_path + "input2", output_path + "input3"]
    output_paths = ','.join(output_path_lst)
    print("xacl_fmk -m " + model_path + " -i " + output_paths +
          " -o " + inference_path + '/kgat_output_bin')
    os.system("xacl_fmk -m " + model_path + " -i " +
              output_paths + " -o " + inference_path + '/kgat_output_bin')
    print(inference_path)
    print("[INFO]    推理结果生成结束")


def inference_files(inference_path):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2

    users_to_test = list(data_generator.test_user_dict.keys())
    test_users = users_to_test
    n_test_users = len(test_users)

    files = sorted(os.listdir(inference_path))
    files = [inference_path + '/' + i for i in files]

    for u_batch_id, f in enumerate(files):
        if f.endswith(".bin"):
            rate_batch = np.fromfile(f, dtype='float32')
            rate_batch = rate_batch.reshape((-1, ITEM_NUM))

            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]

            user_batch_rating_uid = zip(rate_batch, user_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users
    pool.close()
    print(result)


if __name__ == '__main__':
    args = parse_args()

    output_path = args.output_path
    inference_path = args.inference_path
    model_path = args.model_path

    xaclPath(output_path, inference_path, model_path)
    inference_files(inference_path)
