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
#include "crowd_count.h"
#include <iostream>
#include "acl/acl.h"
#include "model_process.h"
#include "utils.h"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/opencv.hpp"
#define MAX_PIXEL_255 (255)

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
    destroy_resource();
}

Result CrowdCount::init_resource() {
    // ACL init
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

Result CrowdCount::init_model(const char* omModelPath) {
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

Result CrowdCount::init() {
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

    ret = dvpp_.init_resource(stream_);
    if (ret != SUCCESS) {
        ERROR_LOG("Init dvpp failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}


Result CrowdCount::pre_process(ImageData& resizedImage, ImageData& srcImage) {
    ImageData imageDevice;
    Utils::copy_image_data_to_device(imageDevice, srcImage, runMode_);
    ImageData yuvImage;
    Result ret = dvpp_.cvt_jpeg_to_yuv420sp(yuvImage, imageDevice);
    if (ret == FAILED) {
        ERROR_LOG("Convert jpeg to yuv failed");
        return FAILED;
    }

    //resize
    ret = dvpp_.crop_and_paste(resizedImage, yuvImage, modelWidth_, modelHeight_);

    if (ret == FAILED) {
        ERROR_LOG("Resize image failed");
        return FAILED;
    }
    return SUCCESS;
}




Result CrowdCount::inference(aclmdlDataset*& inferenceOutput,
ImageData& resizedImage) {

    Result ret = model_.create_input(resizedImage.data.get(),
    resizedImage.size);
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed\n");
        return FAILED;
    }
    ret = model_.execute();
    if (ret != SUCCESS) {
        model_.destroy_input();
        ERROR_LOG("Execute model inference failed\n");
        return FAILED;
    }
    model_.destroy_input();
    inferenceOutput = model_.get_model_output_data();
    return SUCCESS;
}

Result CrowdCount::post_process(ImageData& image, aclmdlDataset* modelOutput,
const string& origImagePath) {
    uint32_t dataSize = 0;
    float* detectData = (float *)get_inference_output_item(dataSize, modelOutput,0);
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
    stream << Utils::round(sum) ;
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

void* CrowdCount::get_inference_output_item(uint32_t& itemDataSize,
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
        data = Utils::copy_data_device_to_local(dataBufferDev, bufferSize);
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

void CrowdCount::destroy_resource()
{
    aclrtFree(imageInfoBuf_);
    model_.destroy_resource();
    dvpp_.destroy_resource();
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
