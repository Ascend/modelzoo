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

* File sample_process.cpp
* Description: handle acl resource
*/
#include "crowd_count.h"
#include <iostream>
#include "acl/acl.h"
#include "model_process.h"
#include "utils.h"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/opencv.hpp"
#define MAX_PIXEL_255 (255)

using namespace std;
CrowdCount::CrowdCount(const char* modelPath,
uint32_t modelWidth,
uint32_t modelHeight)
:deviceId_(0), context_(nullptr), stream_(nullptr), modelWidth_(modelWidth),
modelHeight_(modelHeight), isInited_(false){
    imageInfoSize_ = 0;
    imageInfoBuf_ = nullptr;
    modelPath_ = modelPath;
}

CrowdCount::~CrowdCount() {
    DestroyResource();
}

Result CrowdCount::InitResource() {
    // ACL init
    const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    // open device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }

    return SUCCESS;
}

Result CrowdCount::InitModel(const char* omModelPath) {
    Result ret = model_.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = model_.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = model_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    return SUCCESS;
}

Result CrowdCount::Init() {
    if (isInited_) {
        INFO_LOG("Classify instance is initied already!");
        return SUCCESS;
    }

    Result ret = InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("Init acl resource failed");
        return FAILED;
    }

    ret = InitModel(modelPath_);
    if (ret != SUCCESS) {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    ret = dvpp_.InitResource(stream_);
    if (ret != SUCCESS) {
        ERROR_LOG("Init dvpp failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}


Result CrowdCount::Preprocess(ImageData& resizedImage, ImageData& srcImage) {
    ImageData imageDevice;
    Utils::CopyImageDataToDevice(imageDevice, srcImage, runMode_);
    ImageData yuvImage;
    Result ret = dvpp_.CvtJpegToYuv420sp(yuvImage, imageDevice);
    if (ret == FAILED) {
        ERROR_LOG("Convert jpeg to yuv failed");
        return FAILED;
    }

    //resize
    ret = dvpp_.CropAndPaste(resizedImage, yuvImage, modelWidth_, modelHeight_);

    if (ret == FAILED) {
        ERROR_LOG("Resize image failed");
        return FAILED;
    }
    return SUCCESS;
}




Result CrowdCount::Inference(aclmdlDataset*& inferenceOutput,
ImageData& resizedImage) {

    Result ret = model_.CreateInput(resizedImage.data.get(),
    resizedImage.size);
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed\n");
        return FAILED;
    }
    ret = model_.Execute();
    if (ret != SUCCESS) {
        model_.DestroyInput();
        ERROR_LOG("Execute model inference failed\n");
        return FAILED;
    }
    model_.DestroyInput();
    inferenceOutput = model_.GetModelOutputData();
    return SUCCESS;
}

Result CrowdCount::Postprocess(ImageData& image, aclmdlDataset* modelOutput,
const string& origImagePath) {
    uint32_t dataSize = 0;
    float* detectData = (float *)GetInferenceOutputItem(dataSize, modelOutput,0);
    float sum = 0;
    for(int i = 0; i < dataSize / sizeof(float); i++) {
        sum += detectData[i];
    }
    sum = sum / 1000.0f;
    INFO_LOG("sum  %f",sum);

    cv::Mat dst, heatMap;
    cv::Mat orig = cv::imread(origImagePath, CV_LOAD_IMAGE_COLOR);
    cv::resize(orig, orig, cv::Size(image.width, image.height));

    float *data = nullptr;
    if(image.width < modelWidth_) {
        data = new float[image.height * image.width]();
        for(int i = 0; i < image.height; i++) {
            for(int j = 0; j < image.width; j ++) {
                data[i * image.width +j] = detectData[i * modelWidth_ + j];
            }
        }
        heatMap = cv::Mat(image.height,image.width,CV_32FC1, data);

    }else {
        heatMap = cv::Mat(image.height,image.width,CV_32FC1, detectData);
    }

    cv::GaussianBlur(heatMap, heatMap,cv::Size(0,0), 5, 5,cv::BORDER_DEFAULT);
    cv::normalize(heatMap, heatMap, 0,255, cv::NORM_MINMAX,CV_8UC1);
    cv::applyColorMap(heatMap,heatMap,cv::COLORMAP_JET) ;
    cv::addWeighted(orig, 1, heatMap, 0.5, 0.0, dst);

    std::stringstream stream;
    stream << Utils::Round(sum) ;
    cv::putText(dst,stream.str(),cv::Point(30, 60),
    cv::FONT_HERSHEY_PLAIN,5,cv::Scalar( 0,0, MAX_PIXEL_255 ),8);

    int pos = origImagePath.find_last_of("/");
    string filename(origImagePath.substr(pos + 1));
    stringstream outputName;
    outputName.str("");
    outputName << "./output/out_" << filename;
    cv::imwrite(outputName.str(), dst);

    if(data != nullptr) {
        delete[] data;
    }

    return SUCCESS;
}

void* CrowdCount::GetInferenceOutputItem(uint32_t& itemDataSize,
aclmdlDataset* inferenceOutput,uint32_t idx) {
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, idx);
    if (dataBuffer == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer from model "
        "inference output failed", idx);
        return nullptr;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer address "
        "from model inference output failed", idx);
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0) {
        ERROR_LOG("The %dth dataset buffer size of "
        "model inference output is 0", idx);
        return nullptr;
    }

    void* data = nullptr;
    if (runMode_ == ACL_HOST) {
        data = Utils::CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        if (data == nullptr) {
            ERROR_LOG("Copy inference output to host failed");
            return nullptr;
        }
    }
    else {
        data = dataBufferDev;
    }

    itemDataSize = bufferSize;
    return data;
}

void CrowdCount::DestroyResource()
{
    aclrtFree(imageInfoBuf_);
    model_.DestroyResource();
    dvpp_.DestroyResource();
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");
    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}
