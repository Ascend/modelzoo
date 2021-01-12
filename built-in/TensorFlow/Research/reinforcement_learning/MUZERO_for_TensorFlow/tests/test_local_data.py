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
# Copyright 2020 Huawei Technologies Co., Ltd
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
from xt.benchmark.tools.evaluate_xt import fetch_train_event
from xt.benchmark.tools.evaluate_xt import read_train_records
from xt.benchmark.tools.evaluate_xt import DEFAULT_ARCHIVE_DIR


def test_fetch_train_path():
    archive = os.path.join(
        os.path.expanduser("~"), "projects/rl/{}".format(DEFAULT_ARCHIVE_DIR)
    )
    _id = "xt_cartpole_0204"
    r = fetch_train_event(archive, _id)
    print("get", r)


def test_fetch_train_path_single():
    archive = os.path.join(
        os.path.expanduser("~"), "projects/rl/{}".format(DEFAULT_ARCHIVE_DIR)
    )
    _id = "xt_cartpole_0204"
    r = fetch_train_event(archive, _id, True)
    print("get", r)


def test_read_train_record():
    bm_args = {
        "archive_root": os.path.join(
            os.path.expanduser("~"), "projects/rl/{}".format(DEFAULT_ARCHIVE_DIR)
        ),
        "bm_id": "xt_cartpole_0204",
    }
    # import numpy as np
    d = read_train_records(bm_args)
    # d = np.array(d)
    print("\n", d)
