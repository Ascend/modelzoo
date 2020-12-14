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
#include "face_feature.h"
#include <iostream>

#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/opencv.hpp"
#include "acl/acl.h"
#include "model_process.h"
#include "utils.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

FaceFeature::FaceFeature(const char* modelPath, uint32_t modelWidth, uint32_t modelHeight)
    :deviceId_(0), context_(nullptr), stream_(nullptr), isInited_(false),
    modelWidth_(modelWidth), modelHeight_(modelHeight){
    imageInfoSize_ = 0;
    imageInfoBuf_ = nullptr;
    modelPath_ = modelPath;
}

FaceFeature::~FaceFeature() {
    destroy_resource();
}

Result FaceFeature::init_resource() {
    // ACL init
    //const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(nullptr);
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

Result FaceFeature::init_model(const char* omModelPath) {
    Result ret = model_.load_model_from_file_with_mem(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = model_.create_desc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = model_.create_output();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }
    return SUCCESS;
}

Result FaceFeature::create_image_info_buffer()
{
    const float imageInfo[4] = {(float)modelWidth_, (float)modelHeight_,
    (float)modelWidth_, (float)modelHeight_};
    imageInfoSize_ = sizeof(imageInfo);
    if (runMode_ == ACL_HOST) {
        imageInfoBuf_ = Utils::copy_data_host_to_device((void *)imageInfo, imageInfoSize_);
    }
    else {
        imageInfoBuf_ = Utils::copy_data_device_to_device((void *)imageInfo, imageInfoSize_);
    }
    if (imageInfoBuf_ == nullptr) {
        ERROR_LOG("Copy image info to device failed");
        return FAILED;
    }

    return SUCCESS;
}

Result FaceFeature::init() {
    if (isInited_) {
        INFO_LOG("Classify instance is initied already!");
        return SUCCESS;
    }

    Result ret = init_resource();
    if (ret != SUCCESS) {
        ERROR_LOG("Init acl resource failed");
        return FAILED;
    }

    ret = init_model(modelPath_);
    if (ret != SUCCESS) {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    ret = create_image_info_buffer();
    if (ret != SUCCESS) {
        ERROR_LOG("Create image info buf failed");
        return FAILED;
    }
    isInited_ = true;
    return SUCCESS;
}

Result FaceFeature::pre_process(const string& imageFile,ImageData& rgbImage) {

    cv::Mat mat = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);
    if (mat.empty()) {
        return FAILED;
    }
    int width = mat.cols;
    int height = mat.rows;

    if(width != 112 || height != 112) {
        return FAILED;
    }

    rgbImage.width = width;
    rgbImage.height = height;
    rgbImage.alignWidth = width;
    rgbImage.alignHeight = height;
    rgbImage.size =width*height*3;

    /* use new to allocate memeory */
    std::shared_ptr<uint8_t> tmp(new uint8_t[width * height * 3]);
    rgbImage.data = tmp;
    memcpy(rgbImage.data.get(), mat.data, width * height * 3);

    return SUCCESS;
}

Result FaceFeature::inference(aclmdlDataset*& inferenceOutput,
    ImageData& resizedImage) {

    Result ret = model_.create_input(resizedImage.data.get(),
                resizedImage.size);
    if (ret != SUCCESS) {
        ERROR_LOG("feature Create mode input dataset failed\n");
        return FAILED;
    }

    ret = model_.execute();
    if (ret != SUCCESS) {
        model_.destroy_input();
        ERROR_LOG("feature Execute model inference failed\n");
        return FAILED;
    }
    model_.destroy_input();

    inferenceOutput = model_.get_model_output_data();
    return SUCCESS;
}

Result FaceFeature::post_process(aclmdlDataset* modelOutput,FaceInfo& faceInfo) {
    uint32_t dataSize = 0;
    float* detectData = (float *)get_inference_output_item(dataSize, modelOutput, 0);

    INFO_LOG("Postprocess dataSize %d ", dataSize);

    /*for(int i = 0; i < dataSize / sizeof(float); i++) {
        INFO_LOG("Postprocess i %d, %f", i, detectData[i]);
    }*/

    memcpy(faceInfo.feature, detectData, dataSize);
    faceInfo.featureSize = dataSize / sizeof(float);

    return SUCCESS;
}

void* FaceFeature::get_inference_output_item(uint32_t& itemDataSize,
            aclmdlDataset* inferenceOutput,uint32_t idx) {

    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, idx);
    if (dataBuffer == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer from model "
        "inference output failed\n", idx);
        return nullptr;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer address "
        "from model inference output failed\n", idx);
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0) {
        ERROR_LOG("The %d   th dataset buffer size of "
        "model inference output is 0\n", idx);
        return nullptr;
    }

    void* data = nullptr;
    data = dataBufferDev;

    itemDataSize = bufferSize;
    return data;
}

void FaceFeature::destroy_resource()
{
    aclrtFree(imageInfoBuf_);
    model_.destroy_resource();
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
