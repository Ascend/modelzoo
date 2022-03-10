/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <vector>
#include "Yolov3Detection.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void SplitString(const std::string &s, std::vector<std::string> &v, const std::string &c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }

    if (pos1 != s.length()) {
        v.push_back(s.substr(pos1));
    }
}

void InitYolov3Param(InitParam &initParam)
{
    initParam.deviceId = 0;
    initParam.labelPath = "../data/config/coco.names";
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/yolov3_tf_aipp.om";
    initParam.classNum = 80;
    initParam.biasesNum = 18;
    initParam.biases = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
    initParam.objectnessThresh = "0.3";
    initParam.iouThresh = "0.45";
    initParam.scoreThresh = "0.3";
    initParam.yoloType = 3;
    initParam.modelType = 0;
    initParam.inputType = 0;
    initParam.anchorDim = 3;
}

APP_ERROR ReadImagesPath(const std::string &path, std::vector<std::string> &imagesPath)
{
    std::ifstream inFile;
    inFile.open(path, std::ios_base::in);
    std::string line;
    // Check images path file validity
    if (inFile.fail()) {
        LogError << "Failed to open label file: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::vector<std::string> vectorStr;
    std::string splitStr = " ";
    // construct label map
    while (std::getline(inFile, line)) {
        if (line.find('#') <= 1) {
            continue;
        }
        vectorStr.clear();
        SplitString(line, vectorStr, splitStr);
        imagesPath.push_back(vectorStr[1]);
    }

    inFile.close();
    return APP_ERR_OK;
}

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './yolov3 infer.txt'.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitYolov3Param(initParam);
    auto yolov3 = std::make_shared<Yolov3DetectionOpencv>();
    APP_ERROR ret = yolov3->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Yolov3DetectionOpencv init failed, ret=" << ret << ".";
        return ret;
    }

    std::string inferText = argv[1];
    std::vector<std::string> imagesPath;
    ret = ReadImagesPath(inferText, imagesPath);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImagesPath failed, ret=" << ret << ".";
        return ret;
    }
    for (uint32_t i = 0; i < imagesPath.size(); i++) {
        LogInfo << "read image path " << imagesPath[i];
        ret = yolov3->Process(imagesPath[i]);
        if (ret != APP_ERR_OK) {
            LogError << "Yolov3DetectionOpencv process failed, ret=" << ret << ".";
            yolov3->DeInit();
            return ret;
        }
    }

    yolov3->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}