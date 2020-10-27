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
"""hub config."""
from src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2

def create_network(name, *args, **kwargs):
    if name == "mobilenetv2":
        backbone_net = MobileNetV2Backbone()
        include_top = kwargs["include_top"]
        if include_top is None:
            include_top = True
        if include_top:
            activation = kwargs["activation"]
            head_net = MobileNetV2Head(input_channel=backbone_net.out_channels,
                                       num_classes=int(kwargs["num_classes"]),
                                       activation=activation)
            net = mobilenet_v2(backbone_net, head_net)
            return net
        return backbone_net
    raise NotImplementedError(f"{name} is not implemented in the repo")
