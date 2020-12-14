# -*- coding: utf-8 -*-

# Copyright [yyyy] [name of copyright owner]
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

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import torch
import torch.onnx

from collections import OrderedDict
from efficientnet_pytorch.model import EfficientNet

def proc_node_module(checkpoint,AttrName):
    new_state_dict = OrderedDict()
    for k,v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict
def convert():
    checkpoint = torch.load("./checkpoint.pth", map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
    model = EfficientNet.from_name("efficientnet-b0")
    model.set_swish(memory_efficient=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval();

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(10,3,224,224)
    torch.onnx.export(model,dummy_input,"efficientnet_npu_16.onnx", input_names = input_names, output_names = output_names, opset_version=11)
if __name__ == "__main__":
    convert()