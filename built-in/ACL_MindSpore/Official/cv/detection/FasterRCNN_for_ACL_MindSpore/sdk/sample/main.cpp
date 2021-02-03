/*
 * Copyright(C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <getopt.h>

#include <boost/property_tree/json_parser.hpp>
#include <cstring>
#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "MxBase/Log/Log.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"

namespace {
const std::string STREAM_NAME = "im_fasterrcnn";
}  // namespace

namespace {
APP_ERROR ReadFile(const std::string &filePath, MxStream::MxstDataInput &dataBuffer) {
    char c[PATH_MAX + 1] = {0x00};
    size_t count = filePath.copy(c, PATH_MAX + 1);
    if (count != filePath.length()) {
        LogError << "Failed to copy file path(" << c << ").";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the absolute path of input file
    char path[PATH_MAX + 1] = {0x00};
    if ((strlen(c) > PATH_MAX) || (realpath(c, path) == nullptr)) {
        LogError << "Failed to get image, the image path is (" << filePath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    // Open file with reading mode
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file (" << path << ").";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // Get the length of input file
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // If file not empty, read it into FileInfo and return it
    if (fileSize > 0) {
        dataBuffer.dataSize = fileSize;
        dataBuffer.dataPtr = new (std::nothrow) uint32_t[fileSize];
        if (dataBuffer.dataPtr == nullptr) {
            LogError << "allocate memory with \"new uint32_t\" failed.";
            fclose(fp);
            return APP_ERR_COMM_FAILURE;
        }

        uint32_t readRet = fread(dataBuffer.dataPtr, 1, fileSize, fp);
        if (readRet <= 0) {
            fclose(fp);
            return APP_ERR_COMM_READ_FAIL;
        }
        fclose(fp);
        return APP_ERR_OK;
    }
    fclose(fp);
    return APP_ERR_COMM_FAILURE;
}

std::string ReadPipelineConfig(const std::string &pipelineConfigPath) {
    std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
    if (!file) {
        LogError << pipelineConfigPath << " file dose not exist.";
        return "";
    }
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    std::unique_ptr<char[]> data(new char[fileSize]);
    file.read(data.get(), fileSize);
    file.close();
    std::string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}

}  // namespace

APP_ERROR GetRealPath(std::string &srcPath, std::string &realPath) {
    char path[PATH_MAX + 1] = {0};
    if ((srcPath.size() > PATH_MAX) || (realpath(srcPath.c_str(), path) == nullptr)) {
        LogError << "Failed to get realpath: (" << srcPath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    realPath = path;
    return APP_ERR_OK;
}

APP_ERROR GetOptions(int argc, char *argv[], std::string &imagePath, std::string &configPath) {
    std::string srcImage;
    std::string srcConfig;

    int opt = 0;
    int option_index = 0;
    std::string optString = "hi:c:f:";
    static struct option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                           {"image", required_argument, nullptr, 'i'},
                                           {"config", required_argument, nullptr, 'c'},
                                           {nullptr, 0, nullptr, 0}};

    while ((opt = getopt_long(argc, argv, optString.c_str(), long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                srcImage = optarg;
                break;
            case 'c':
                srcConfig = optarg;
                break;
            case 'h':
            default:
                std::cout << "Usages: \n"
                             "    imagedemo [options] \n"
                             "Args:\n"
                             "    -h --help         help message\n"
                             "    -i --image        The image used for inference.\n"
                             "    -c --config       The path of the pipeline file, "
                             "default is ../pipeline/Sample.pipeline.\n";
                return APP_ERR_COMM_FAILURE;
        }
    }

    if (GetRealPath(srcConfig, configPath) != APP_ERR_OK or GetRealPath(srcImage, imagePath) != APP_ERR_OK) {
        LogError << "[Main args] The configuration file(" << srcConfig << ") or image is invalid, path(" << srcImage
                 << ").";
        return APP_ERR_COMM_NO_EXIST;
    }

    LogInfo << "[Main args] The configuration file:             " << configPath;
    LogInfo << "[Main args] The image which used for inference: " << imagePath;

    return APP_ERR_OK;
}

struct ObjectInfo {
    float x0;
    float y0;
    float x1;
    float y1;
    float confidence;
    int classId;
    std::string label;
};

APP_ERROR DrawRectangle(std::string &imgPath, std::vector<ObjectInfo> &detects) {
    cv::Mat srcImg = cv::imread(imgPath);
    cv::RNG rng(5004);
    for (auto &det : detects) {
        cv::Rect rect(det.x0, det.y0, det.x1 - det.x0, det.y1 - det.y0);
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(100, 255));
        cv::rectangle(srcImg, rect, color, 2);
        std::string labelText(det.label + ", " + std::to_string(det.confidence));
        cv::putText(srcImg, labelText, cv::Point(det.x0 + 5, std::max<int>(det.y0 - 5, 0)), 2, 0.5, color, 1);
    }
    cv::imwrite(imgPath.substr(0, imgPath.rfind(".")) + "-label.jpg", srcImg);
    return APP_ERR_OK;
}

APP_ERROR WriteResult(std::string &imgPath, std::stringstream &result) {
    namespace pt = boost::property_tree;
    pt::ptree root;
    pt::json_parser::read_json(result, root);
    if (!root.get_child_optional("MxpiObject")) {
        LogError << "The predict result is null.";
        return APP_ERR_OK;
    }

    root = root.get_child("MxpiObject");
    std::vector<ObjectInfo> detects;
    std::stringstream echoStr;
    echoStr << "\nThe predicted results are as follows:\n";
    size_t predictNum = 0;
    for (auto it = root.begin(); it != root.end(); ++it) {
        ObjectInfo detObj = {
            it->second.get<float>("x0"),
            it->second.get<float>("y0"),
            it->second.get<float>("x1"),
            it->second.get<float>("y1"),
        };
        pt::ptree clsPt = it->second.get_child("classVec");
        for (auto &subPt : clsPt) {
            detObj.classId = subPt.second.get<int>("classId");
            detObj.label = subPt.second.get<std::string>("className");
            detObj.confidence = subPt.second.get<float>("confidence");
        }
        echoStr << "Object: " << ++predictNum << "\n"
                << "    classId: " << detObj.classId << "\n"
                << "    label: " << detObj.label << "\n"
                << "    confidence: " << detObj.confidence << "\n"
                << "    bbox: [" << detObj.x0 << ", " << detObj.y1 << ", " << detObj.x0 << ", " << detObj.y0 << "]\n"
                << "-----------------------\n";
        detects.push_back(detObj);
    }

    LogInfo << echoStr.str();
    DrawRectangle(imgPath, detects);
    return APP_ERR_OK;
}

int main(int argc, char *argv[]) {
    // read image file and build stream input
    MxStream::MxstDataInput dataBuffer;
    std::string imgPath, pipelinePath;
    APP_ERROR ret = GetOptions(argc, argv, imgPath, pipelinePath);
    if (ret != APP_ERR_OK) {
        LogError << "The input parameter error.";
        return ret;
    }
    ret = ReadFile(imgPath, dataBuffer);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to read image file, ret = " << ret << ".";
        return ret;
    }
    // read pipeline config file
    std::string pipelineConfig = ReadPipelineConfig(pipelinePath);
    if (pipelinePath == "") {
        LogError << "Read pipeline failed.";
        return APP_ERR_COMM_INIT_FAIL;
    }
    // init stream manager
    MxStream::MxStreamManager mxStreamManager;
    ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init Stream manager, ret = " << ret << ".";
        return ret;
    }
    // create stream by pipeline config file
    ret = mxStreamManager.CreateMultipleStreams(pipelineConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create Stream, ret = " << ret << ".";
        return ret;
    }
    std::string streamName = STREAM_NAME;
    int inPluginId = 0;
    // send data into stream
    ret = mxStreamManager.SendData(streamName, inPluginId, dataBuffer);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to send data to stream, ret = " << ret << ".";
        return ret;
    }
    // get stream output
    MxStream::MxstDataOutput *output = mxStreamManager.GetResult(streamName, inPluginId);
    if (output == nullptr) {
        LogError << "Failed to get pipeline output.";
        return ret;
    }

    std::string((char *)output->dataPtr, output->dataSize);
    std::stringstream result(std::string((char *)output->dataPtr, output->dataSize));
    // LogInfo << "Results:" << result.get();

    ret = WriteResult(imgPath, result);

    // destroy streams
    mxStreamManager.DestroyAllStreams();
    delete dataBuffer.dataPtr;
    dataBuffer.dataPtr = nullptr;

    delete output;
    return 0;
}
