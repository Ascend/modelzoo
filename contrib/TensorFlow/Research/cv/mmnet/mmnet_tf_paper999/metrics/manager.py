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
import metrics.ops as mops
import metrics.parser as parser
from metrics.base import MetricManagerBase
from metrics.summaries import Summaries


class MattingMetricManager(MetricManagerBase):
    _metric_input_data_parser = parser.MattingDataParser

    def __init__(self,
                 is_training: bool,
                 save_evaluation_image: bool,
                 exclude_metric_names: list,
                 summary: Summaries):
        super().__init__(exclude_metric_names, summary)
        self.register_metrics([
            # misc
            mops.InferenceTimeMetricOp(),
            # tensor ops
            mops.LossesMetricOp(),
            mops.ImageSummaryOp(),

            mops.MADMetricOp(),
            mops.GaussianGradMetricOp(),
        ])

        if not is_training and save_evaluation_image:
            self.register_metrics([
                mops.MiscImageRetrieveOp(),
                mops.MiscImageSaveOp(),
            ])
