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

#include "normalization.h"
#include "utils.h"

float norm_sum(const float* vectorBuf, int size)
{
    float a = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        a = pow(vectorBuf[i], 2.0f);
        sum += a;
    }

    return sqrt(sum);
}

void normalization(float* vectorBuf, int size)
{
    float a = norm_sum(vectorBuf, size);
    a = (a > DBL_EPSILON) ? a : DBL_EPSILON;

    for (int i = 0; i < size; i++) {
        vectorBuf[i] = vectorBuf[i] / a;
    }
}

// the bigger the better
float cal_similarity(const float* x, const float* y, int size)
{
    int i = 0;
    float a = 0.0f;
    float b = 0.0f;
    float similarity = 0.0f;

    for (i = 0; i < size; i++) {
        similarity += x[i] * y[i];
    }
    //return similarity;
    return (1 + similarity) / 2.0;
}

int search_feature_lib(std::vector<FaceInfo> featureLib, float* featureVector, int featureSize, float& similarity)
{
    normalization(featureVector, featureSize);

    float maxSimilarity = 0.0f;
    int objectId = -1;
    for(int i = 0; i < featureLib.size(); i++) {
        float simi = cal_similarity(featureVector, featureLib[i].feature, featureSize);
        if (simi > maxSimilarity) {
            maxSimilarity = simi;
            objectId = i;
        }
    }
    similarity = maxSimilarity;
    INFO_LOG("recognition maxSimilarity is %f", maxSimilarity);
    const float SIMILAR_THREHOLD = 0.8;
    if (maxSimilarity  >= SIMILAR_THREHOLD) {
        return objectId;
    } else {
        return -1;
    }
}