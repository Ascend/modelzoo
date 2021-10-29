#
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
#
from npu_bridge.npu_init import *
import argparse
import numpy as np
from resnet3d import Resnet3DBuilder

# add args
parser = argparse.ArgumentParser(description="train data")
parser.add_argument('--X_train_file', type=str, help='X_train_data_path')
parser.add_argument('--labels_file', type=str, help='labels_file')
parser.add_argument('--y_train_file', type=str, help='y_train_file')
parser.add_argument('--train_epochs', type=int, help='train_epochs', default=20)
parser.add_argument('--batch_size', type=int, help='batch_size', default=10)
args = parser.parse_args()

#############npu modify start###############
# npu_keras_sess = set_keras_session_npu_config()
global_config = tf.ConfigProto(log_device_placement=False)
custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
npu_keras_sess = set_keras_session_npu_config(config=global_config)
#############npu modify end###############

# pseudo volumetric data
# X_train = np.random.rand(1000, 64, 64, 32, 1)
# labels = np.random.randint(0, 2, size=[1000])
# y_train = np.eye(2)[labels]
# np.save(file="X_train.npy", arr=X_train)
# np.save(file="labels.npy", arr=labels)
# np.save(file="y_train.npy", arr=y_train)

X_train = np.load(file=args.X_train_file)
labels = np.load(file=args.labels_file)
y_train = np.load(file=args.y_train_file)

# train
model = Resnet3DBuilder.build_resnet_18((64, 64, 32, 1), 2)
#model.compile(loss="categorical_crossentropy", optimizer=npu_keras_optimizer(tf.keras.optimizers.SGD()))
model.compile(loss="categorical_crossentropy",optimizer="sgd")
model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.train_epochs)
model.save("resnet-3d.h5")
close_session(npu_keras_sess)

