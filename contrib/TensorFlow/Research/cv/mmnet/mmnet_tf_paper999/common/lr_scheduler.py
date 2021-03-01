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
def factory(args, log, global_step_from_checkpoint=None, global_step=None, dataset=None):
    learning_rate_scheduler = FixedLR(args, log)
    return learning_rate_scheduler


def add_arguments(parser):
    FixedLR.add_arguments(parser)


class FixedLR:
    def __init__(
        self, args, logger
    ):
        self.args = args
        self.logger = logger
        assert hasattr(self.args, "learning_rate") and isinstance(self.args.learning_rate, float)
        self.learning_rate = self.args.learning_rate

        self.placeholder = self.learning_rate
        self.should_feed_dict = False

    @staticmethod
    def add_arguments(parser):
        g_lr = parser.add_argument_group("Learning Rate Arguments")
        g_lr.add_argument("--learning_rate", default=1e-4, type=float, help="Initial learning rate for gradient update")
