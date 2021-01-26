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

import onnx

model = onnx.load("yolov5s.onnx")

prob_info = onnx.helper.make_tensor_value_info('images', onnx.TensorProto.FLOAT, [-1, 12, 320, 320])
model.graph.input.remove(model.graph.input[0])
model.graph.input.insert(0, prob_info)
model.graph.node[41].input[0] = 'images'

node_list = ["Concat_40"]

slice_node = ["Slice_4", "Slice_14", "Slice_24", "Slice_34", "Slice_9", "Slice_19", "Slice_29", "Slice_39", ]
node_list.extend(slice_node)

max_idx = len(model.graph.node)
rm_cnt = 0
for i in range(len(model.graph.node)):
    if i < max_idx:
        n = model.graph.node[i - rm_cnt]
        if n.op_type == "Transpose":
            print(n.name)
        if n.name in node_list:
            print("remove {} total {}".format(n.name, len(model.graph.node)))
            model.graph.node.remove(n)
            max_idx -= 1
            rm_cnt += 1


onnx.checker.check_model(model)

onnx.save(model, "modify_yolov5.onnx")
