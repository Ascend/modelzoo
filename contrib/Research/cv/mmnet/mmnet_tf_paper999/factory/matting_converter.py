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
from abc import ABC
from abc import abstractmethod
import tensorflow as tf


class ConverterBase(ABC):
    @classmethod
    @abstractmethod
    def convert(
        cls,
        logits: tf.Tensor,
        output_name: str,
        num_classes: int,
    ):
        raise NotImplementedError(f"convert() not defined in {cls.__class__.__name__}")


class ProbConverter(ConverterBase):
    @classmethod
    def convert(
        cls,
        logits: tf.Tensor,
        output_name: str,
        num_classes: int,
    ):
        assert num_classes == 2

        softmax_scores = tf.contrib.layers.softmax(logits, scope="output/softmax")
        # tf.identity to assign output_name
        output = tf.identity(softmax_scores, name=output_name)
        return output

