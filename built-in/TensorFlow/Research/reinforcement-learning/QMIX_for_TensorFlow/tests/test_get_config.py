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

import sys

from xt.benchmark.tools.get_config import *


def test_main():
    """
    test parse config file
    """
    config_file = sys.argv[1]
    ret_para = parse_xt_multi_case_paras(config_file)
    for para in ret_para:
        print(para)


def test_find_items():
    """
    test find key in dictionary
    """
    config_file = sys.argv[1]
    with open(config_file) as file_hander:
        yaml_obj = yaml.safe_load(file_hander)

    ret_obj = finditem(yaml_obj, "agent_config")
    print(ret_obj)


if __name__ == "__main__":
    test_main()
    # test_find_items()
