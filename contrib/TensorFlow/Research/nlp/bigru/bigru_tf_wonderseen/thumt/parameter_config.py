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
from thumt.npu_utils import *
  
##
dtype = tf.float32			# float16 or float32

## util settings
STOP_SEE_VARIABLE = False

## model parameters
REFERENCE_NUM = 1       	# the reference pair using in the interface module
TRAIN_DECODE_LENGTH = 61   	# the max sequence length of target sentences
TRAIN_ENCODE_LENGTH = 60	# the max sequence length of source sentences
EVAL_DECODE_LENGTH = 100	# the max sequence length of target sentences
EVAL_ENCODE_LENGTH = 100	# the max sequence length of source sentences
EVAL_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 128
TEST_INFERENCE = False
BOS_ID = 2

## NPU configuration
using_NPU = True			# CPU or NPU
using_dynamic = False		# using dynamic shape or static shape
session_config_fn = npu_config if using_NPU else old_session_config


## Data Path
data_dir = './'#'s3://bi-gru/scripts/thumt/data'
if 's3:' in data_dir:
    import moxing as mox
    mox.file.shift('os', 'mox')
