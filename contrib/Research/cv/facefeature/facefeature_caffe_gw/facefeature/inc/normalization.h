/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once
#include <cmath>
#include <cfloat>
#include "utils.h"

float norm_sum(const float* vectorBuf, int size);
void normalization(float* vectorBuf, int size);
float cal_similarity(const float* x, const float* y, const int size);
int search_feature_lib(std::vector<FaceInfo> featureLib, float* featureVector, int featureSize, float& similarity);
