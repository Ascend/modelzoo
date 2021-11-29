# coding=utf-8
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
import tensorflow as tf


class Layers: 
    def get_accuracy(self, labels, predicted_classes, logits, args):
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes) 
        top5acc = tf.metrics.mean(
            tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
        if args.rank_size == 1:
            newaccuracy = (accuracy[0], accuracy[1])
            newtop5acc = (top5acc[0], top5acc[1])
        else:
            from npu_bridge.hccl import hccl_ops
            newaccuracy = (hccl_ops.allreduce(accuracy[0],"sum")/args.rank_size, accuracy[1])
            newtop5acc = (hccl_ops.allreduce(top5acc[0],"sum")/args.rank_size, top5acc[1])
        metrics = {'val-top1acc': newaccuracy, 'val-top5acc': newtop5acc}
        return metrics




