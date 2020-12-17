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

"""Default config variables, which may be overridden by a user config."""
import os.path as osp
import os

PYTHON_COMMAND = "python"

SUMO_SLEEP = 1.0  # Delay between initializing SUMO and connecting with TraCI

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = PROJECT_PATH + "/data"

# users set both of these in their bash_rc or bash_profile
# and also should run aws configure after installing awscli
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_S3_PATH = "s3://bucket_name"

# path to the Aimsun_Next main directory (required for Aimsun simulations)
AIMSUN_NEXT_PATH = os.environ.get("AIMSUN_NEXT_PATH", None)


# path to the aimsun_flow environment's main directory (required for Aimsun
# simulations)
AIMSUN_SITEPACKAGES = os.environ.get("AIMSUN_SITEPACKAGES", None)
