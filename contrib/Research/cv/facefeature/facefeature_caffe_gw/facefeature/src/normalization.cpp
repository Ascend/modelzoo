#include "normalization.h"
#include "utils.h"

float L2NormSum(const float* vectorBuf, int size)
{
    float a;
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        a = pow(vectorBuf[i], 2.0f);
        sum += a;
    }

    return sqrt(sum);
}

void L2Normalization(float* vectorBuf, int size)
{
    float a = L2NormSum(vectorBuf, size);
    a = (a > DBL_EPSILON) ? a : DBL_EPSILON;

    for (int i = 0; i < size; i++) {
        vectorBuf[i] = vectorBuf[i] / a;
    }
}

// the bigger the better
float CalSimilarity(const float* x, const float* y, int size)
{
    int i;
    float a = 0.0f;
    float b = 0.0f;
    float similarity = 0.0f;

    for (i = 0; i < size; i++) {
        similarity += x[i] * y[i];
    }
    //return similarity;
    return (1 + similarity) / 2.0;
}

int SearchFeatureLib(std::vector<FaceInfo> featureLib, float* featureVector, int featureSize, float& similarity)
{
    L2Normalization(featureVector, featureSize);

    float maxSimilarity = 0.0f;
    int objectId = -1;
    for(int i = 0; i < featureLib.size(); i++) {
        float simi = CalSimilarity(featureVector, featureLib[i].feature, featureSize);
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