#pragma once
#include <cmath>
#include <cfloat>
#include "utils.h"

float L2NormSum(const float* vectorBuf, int size);
void L2Normalization(float* vectorBuf, int size);
float CalSimilarity(const float* x, const float* y, const int size);
int SearchFeatureLib(std::vector<FaceInfo> featureLib, float* featureVector, int featureSize, float& similarity);
