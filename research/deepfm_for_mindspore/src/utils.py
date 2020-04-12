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
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn, Parameter, ParameterTuple, context
from mindspore.common.initializer import Uniform, initializer, Normal

np_type = np.float32
ms_type = mstype.float32
 
def add_write(file_path, out_str):
    with open(file_path, 'a+', encoding="utf-8") as file_out:
        file_out.write( out_str + "\n" )

def init_method( method, shape, name, max_val=0.01):
    if method in ['random', 'uniform']: 
        params = Parameter(initializer(Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, ms_type), name=name) 
    elif method == "normal":
        params = Parameter(initializer(Normal(max_val), shape, ms_type), name=name)
    return params
 
def init_var_dict(init_args, vars):
    var_map = {}
    _, _max_val= init_args
    for _i in range(len(vars)):
        key, shape, init_method = vars[_i]
        if key not in var_map.keys():
            if init_method in ['random', 'uniform']:
                var_map[key] = Parameter(initializer(Uniform(_max_val), shape, ms_type), name=key)
            elif init_method == "one":
                var_map[key] = Parameter(initializer("ones", shape, ms_type), name=key)    
            elif init_method == "zero":
                var_map[key] = Parameter(initializer("zeros", shape, ms_type), name=key) 
            elif init_method == 'normal': 
                var_map[key] = Parameter(initializer(Normal(_max_val), shape, ms_type), name=key)
    return var_map
# 