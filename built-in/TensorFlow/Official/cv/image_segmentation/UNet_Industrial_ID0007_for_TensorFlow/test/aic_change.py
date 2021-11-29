# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#
# ==============================================================================

import json
json_path = "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/config/ascend910/aic-ascend910-ops-info.json"
new_json_path = "new-aic-ascend910-ops-info.json"
with open(json_path,'r') as load_f:
    load_dict = json.load(load_f)

for key in load_dict.keys():
    if key in ["ReduceSumD", "Sigmoid"]:
        print(load_dict[key].keys())
        load_dict[key]["precision_reduce"] = {"flag": "false"}
        print(load_dict[key].keys())
        print(load_dict[key]["precision_reduce"])
with open(json_path, 'w') as dump_f:
    json.dump(load_dict, dump_f, indent=1, sort_keys=True)