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
import sys
import onnx

INT_MAX = sys.maxsize

if len(sys.argv) == 1:
    print("usage: python3.7 modify_yolov5.py model.onnx [n,c,h,w]")
    exit(0)

model_path = sys.argv[1]
model = onnx.load(model_path)

b, c, h, w = 1, 3, 640, 640
if len(sys.argv) == 3:
    b, c, h, w = list(map(int, sys.argv[2].split(',')))
    print(f"input shape: {b, c, h, w}")


def get_node_by_name(nodes, name):
    for n in nodes:
        if n.name == name:
            return n
    return -1

"""
before:                           after:
            input                             input
    /      /      \      \                /           \
slice4 slice14 slice24 slice34        slice4        slice14
   |      |       |       |              |             |
slice9 slice19 slice29 slice39           t              t
    \      \      /       /           /    \         /     \
            concat                slice9 slice29 slice19 slice39
                                     |      |       |       |
                                     t      t       t       t
                                      \      \      /       /
                                              concat
"""

model.graph.node.remove(get_node_by_name(model.graph.node, "Slice_24"))
model.graph.node.remove(get_node_by_name(model.graph.node, "Slice_34"))

prob_info1 = onnx.helper.make_tensor_value_info('to_slice9', onnx.TensorProto.FLOAT, [b, c, h, w//2])
prob_info3 = onnx.helper.make_tensor_value_info('to_slice19', onnx.TensorProto.FLOAT, [b, c, h, w//2])
prob_info5 = onnx.helper.make_tensor_value_info('from_slice9', onnx.TensorProto.FLOAT, [b, c, h//2, w//2])
prob_info6 = onnx.helper.make_tensor_value_info('from_slice19', onnx.TensorProto.FLOAT, [b, c, h//2, w//2])
prob_info7 = onnx.helper.make_tensor_value_info('from_slice29', onnx.TensorProto.FLOAT, [b, c, h//2, w//2])
prob_info8 = onnx.helper.make_tensor_value_info('from_slice39', onnx.TensorProto.FLOAT, [b, c, h//2, w//2])

# slice4 slice24 后的Transpose,若自定义模型，inputs中的编号需根据onnx图中实际编号修改
node1 = onnx.helper.make_node(
    'Transpose',
    inputs=[get_node_by_name(model.graph.node, "Slice_4").output[0]],
    outputs=['to_slice9'],
    perm=[0, 1, 3, 2]
)
node3 = onnx.helper.make_node(
    'Transpose',
    inputs=[get_node_by_name(model.graph.node, "Slice_14").output[0]],
    outputs=['to_slice19'],
    perm=[0, 1, 3, 2]
)
# slice9 slice19 slice29 slice39后的Transpose
node5 = onnx.helper.make_node(
    'Transpose',
    inputs=['from_slice9'],
    outputs=[get_node_by_name(model.graph.node, "Slice_9").output[0]],
    perm=[0, 1, 3, 2]
)
node6 = onnx.helper.make_node(
    'Transpose',
    inputs=['from_slice19'],
    outputs=[get_node_by_name(model.graph.node, "Slice_19").output[0]],
    perm=[0, 1, 3, 2]
)
node7 = onnx.helper.make_node(
    'Transpose',
    inputs=['from_slice29'],
    outputs=[get_node_by_name(model.graph.node, "Slice_29").output[0]],
    perm=[0, 1, 3, 2]
)
node8 = onnx.helper.make_node(
    'Transpose',
    inputs=['from_slice39'],
    outputs=[get_node_by_name(model.graph.node, "Slice_39").output[0]],
    perm=[0, 1, 3, 2]
)
model.graph.node.append(node1)
model.graph.node.append(node3)
model.graph.node.append(node5)
model.graph.node.append(node6)
model.graph.node.append(node7)
model.graph.node.append(node8)

# slice9 slice19 换轴
model.graph.initializer.append(onnx.helper.make_tensor('starts_9', onnx.TensorProto.INT64, [1], [0]))
model.graph.initializer.append(onnx.helper.make_tensor('ends_9', onnx.TensorProto.INT64, [1], [INT_MAX]))
model.graph.initializer.append(onnx.helper.make_tensor('axes_9', onnx.TensorProto.INT64, [1], [2]))
model.graph.initializer.append(onnx.helper.make_tensor('steps_9', onnx.TensorProto.INT64, [1], [2]))
newnode1 = onnx.helper.make_node(
    'Slice',
    name='Slice_9',
    inputs=['to_slice9', 'starts_9', 'ends_9', 'axes_9', 'steps_9'],
    outputs=['from_slice9']
)
model.graph.node.remove(get_node_by_name(model.graph.node, "Slice_9"))
model.graph.node.insert(9, newnode1)
newnode2 = onnx.helper.make_node(
    'Slice',
    name='Slice_19',
    inputs=['to_slice19', 'starts_9', 'ends_9', 'axes_9', 'steps_9'],
    outputs=['from_slice19']
)
model.graph.node.remove(get_node_by_name(model.graph.node, "Slice_19"))
model.graph.node.insert(19, newnode2)

# slice29 slice39 换轴
model.graph.initializer.append(onnx.helper.make_tensor('starts_29', onnx.TensorProto.INT64, [1], [1]))
model.graph.initializer.append(onnx.helper.make_tensor('ends_29', onnx.TensorProto.INT64, [1], [INT_MAX]))
model.graph.initializer.append(onnx.helper.make_tensor('axes_29', onnx.TensorProto.INT64, [1], [2]))
model.graph.initializer.append(onnx.helper.make_tensor('steps_29', onnx.TensorProto.INT64, [1], [2]))
newnode3 = onnx.helper.make_node(
    'Slice',
    name='Slice_29',
    inputs=['to_slice9', 'starts_29', 'ends_29', 'axes_29', 'steps_29'],
    outputs=['from_slice29']
)
model.graph.node.remove(get_node_by_name(model.graph.node, "Slice_29"))
model.graph.node.insert(29, newnode3)
newnode4 = onnx.helper.make_node(
    'Slice',
    name='Slice_39',
    inputs=['to_slice19', 'starts_29', 'ends_29', 'axes_29', 'steps_29'],
    outputs=['from_slice39']
)
model.graph.node.remove(get_node_by_name(model.graph.node, "Slice_39"))
model.graph.node.insert(39, newnode4)

onnx.save(model, model_path.split('.')[0] + "_t.onnx")
print("success")
