/* *
* Copyright 2020 Huawei Technologies Co., Ltd
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
* */

/* *
 * @file dvpp_process.cpp
 *
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "dvpp_process.h"
#include <iostream>
#include "acl/acl.h"
using namespace std;

const int g_w = 16;
const int g_h = 2;

DvppProcess::DvppProcess(aclrtStream &stream)
    : stream_(stream),
      dvppChannelDesc_(nullptr),
      resizeConfig_(nullptr),
      decodeOutDevBuffer_(nullptr),
      decodeOutputDesc_(nullptr),
      resizeInputDesc_(nullptr),
      resizeOutputDesc_(nullptr),
      inDevBuffer_(nullptr),
      inDevBufferSize_(0),
      inputWidth_(0),
      inputHeight_(0),
      resizeOutBufferDev_(nullptr),
      resizeOutBufferSize_(0),
      modelInputWidth_(0),
      modelInputHeight_(0)
{}

DvppProcess::~DvppProcess()
{
    DestroyResource();
    DestroyOutputPara();
}

Result DvppProcess::InitResource()
{
    dvppChannelDesc_ = acldvppCreateChannelDesc();
    if (dvppChannelDesc_ == nullptr) {
        ERROR_LOG("acldvppCreateChannelDesc failed");
        return FAILED;
    }

    aclError ret = acldvppCreateChannel(dvppChannelDesc_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acldvppCreateChannelAsync failed, ret = %d", ret);
        return FAILED;
    }

    resizeConfig_ = acldvppCreateResizeConfig();
    if (resizeConfig_ == nullptr) {
        ERROR_LOG("acldvppCreateResizeConfig failed");
        return FAILED;
    }

    INFO_LOG("dvpp init resource success");
    return SUCCESS;
}

void DvppProcess::SetInput(void *inDevBuffer, int inDevBufferSize, int inputWidth, int inputHeight)
{
    inDevBuffer_ = inDevBuffer;
    inDevBufferSize_ = inDevBufferSize;
    inputWidth_ = inputWidth;
    inputHeight_ = inputHeight;
}

void DvppProcess::GetOutput(void **outputBuffer, int &outputSize)
{
    *outputBuffer = resizeOutBufferDev_;
    outputSize = resizeOutBufferSize_;
    resizeOutBufferDev_ = nullptr;
    resizeOutBufferSize_ = 0;
}

Result DvppProcess::InitOutputPara(int modelInputWidth, int modelInputHeight)
{
    if ((modelInputWidth <= 0) || (modelInputHeight <= 0)) {
        ERROR_LOG("InitInput para invalid");
        return FAILED;
    }


    modelInputWidth_ = modelInputWidth;
    modelInputHeight_ = modelInputHeight;

    // output buffer, adjust the value based on the actual model
    int resizeOutWidth = modelInputWidth_;
    int resizeOutHeight = modelInputHeight_;
    int resizeOutWidthStride = (modelInputWidth_ + (g_w - 1)) / g_w * g_w;
    int resizeOutHeightStride = (modelInputHeight_ + (g_h - 1)) / g_h * g_h;
    resizeOutBufferSize_ = resizeOutWidthStride * resizeOutHeightStride * 3 / 2;
    aclError ret = acldvppMalloc(&resizeOutBufferDev_, resizeOutBufferSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acldvppMalloc resizeOutBuffer failed, ret = %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

void DvppProcess::DestroyOutputPara()
{
    if (resizeOutBufferDev_ != nullptr) {
        acldvppFree(resizeOutBufferDev_);
        resizeOutBufferDev_ = nullptr;
    }
}

Result DvppProcess::InitDecodeOutputDesc()
{
    uint32_t decodeOutWidthStride = (inputWidth_ + 127) / 128 * 128;                     // 128-byte alignment
    uint32_t decodeOutHeightStride = (inputHeight_ + 15) / 16 * 16;                      // 16-byte alignment
    uint32_t decodeOutBufferSize = decodeOutWidthStride * decodeOutHeightStride * 3 / 2; // yuv format size
    aclError ret = acldvppMalloc(&decodeOutDevBuffer_, decodeOutBufferSize);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acldvppMalloc jpegOutBufferDev failed, ret = %d", ret);
        return FAILED;
    }

    decodeOutputDesc_ = acldvppCreatePicDesc();
    if (decodeOutputDesc_ == nullptr) {
        ERROR_LOG("acldvppCreatePicDesc decodeOutputDesc failed");
        return FAILED;
    }

    acldvppSetPicDescData(decodeOutputDesc_, decodeOutDevBuffer_);
    acldvppSetPicDescFormat(decodeOutputDesc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(decodeOutputDesc_, inputWidth_);
    acldvppSetPicDescHeight(decodeOutputDesc_, inputHeight_);
    acldvppSetPicDescWidthStride(decodeOutputDesc_, decodeOutWidthStride);
    acldvppSetPicDescHeightStride(decodeOutputDesc_, decodeOutHeightStride);
    acldvppSetPicDescSize(decodeOutputDesc_, decodeOutBufferSize);
    return SUCCESS;
}

Result DvppProcess::ProcessDecode()
{
    // decode to yuv format
    aclError ret = acldvppJpegDecodeAsync(dvppChannelDesc_, inDevBuffer_, inDevBufferSize_, decodeOutputDesc_, stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acldvppJpegDecodeAsync failed, ret = %d", ret);
        return FAILED;
    }

    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtSynchronizeStream failed");
        return FAILED;
    }
    return SUCCESS;
}

void DvppProcess::DestroyDecodeResource()
{
    if (decodeOutputDesc_ != nullptr) {
        acldvppDestroyPicDesc(decodeOutputDesc_);
        decodeOutputDesc_ = nullptr;
    }
}

Result DvppProcess::InitResizeInputDesc()
{
    uint32_t jpegOutWidthStride = (inputWidth_ + 127) / 128 * 128; // 128-byte alignment
    uint32_t jpegOutHeightStride = (inputHeight_ + 15) / 16 * 16;  // 16-byte alignment
    uint32_t jpegOutBufferSize = jpegOutWidthStride * jpegOutHeightStride * 3 / 2;
    resizeInputDesc_ = acldvppCreatePicDesc();
    if (resizeInputDesc_ == nullptr) {
        ERROR_LOG("InitResizeInputDesc failed");
        return FAILED;
    }

    acldvppSetPicDescData(resizeInputDesc_, decodeOutDevBuffer_);
    acldvppSetPicDescFormat(resizeInputDesc_, PIXEL_FORMAT_YVU_SEMIPLANAR_420);
    acldvppSetPicDescWidth(resizeInputDesc_, inputWidth_);
    acldvppSetPicDescHeight(resizeInputDesc_, inputHeight_);
    acldvppSetPicDescWidthStride(resizeInputDesc_, jpegOutWidthStride);
    acldvppSetPicDescHeightStride(resizeInputDesc_, jpegOutHeightStride);
    acldvppSetPicDescSize(resizeInputDesc_, jpegOutBufferSize);
    return SUCCESS;
}

Result DvppProcess::InitResizeOutputDesc()
{
    // adjust based on the actual model
    int resizeOutputWidthStride = (modelInputWidth_ + (g_w - 1)) / g_w * g_w;
    int resizeOutputHeightStride = (modelInputHeight_ + (g_h - 1)) / g_h * g_h;

    resizeOutputDesc_ = acldvppCreatePicDesc();
    if (resizeOutputDesc_ == nullptr) {
        ERROR_LOG("acldvppCreatePicDesc failed");
        return FAILED;
    }

    acldvppSetPicDescData(resizeOutputDesc_, resizeOutBufferDev_);
    acldvppSetPicDescFormat(resizeOutputDesc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(resizeOutputDesc_, modelInputWidth_);
    acldvppSetPicDescHeight(resizeOutputDesc_, modelInputHeight_);
    acldvppSetPicDescWidthStride(resizeOutputDesc_, resizeOutputWidthStride);
    acldvppSetPicDescHeightStride(resizeOutputDesc_, resizeOutputHeightStride);
    acldvppSetPicDescSize(resizeOutputDesc_, resizeOutBufferSize_);
    return SUCCESS;
}

Result DvppProcess::ProcessResize()
{
    // resize pic size
    aclError ret = acldvppVpcResizeAsync(dvppChannelDesc_, resizeInputDesc_, resizeOutputDesc_, resizeConfig_, stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acldvppVpcResizeAsync failed, ret = %d", ret);
        return FAILED;
    }

    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtSynchronizeStream failed, ret = %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

void DvppProcess::DestroyResizeResource()
{
    if (decodeOutDevBuffer_ != nullptr) {
        (void)acldvppFree(decodeOutDevBuffer_);
        decodeOutDevBuffer_ = nullptr;
    }

    if (resizeInputDesc_ != nullptr) {
        acldvppDestroyPicDesc(resizeInputDesc_);
        resizeInputDesc_ = nullptr;
    }

    if (resizeOutputDesc_ != nullptr) {
        acldvppDestroyPicDesc(resizeOutputDesc_);
        resizeOutputDesc_ = nullptr;
    }
}

void DvppProcess::DestroyResource()
{
    if (resizeConfig_ != nullptr) {
        acldvppDestroyResizeConfig(resizeConfig_);
        resizeConfig_ = nullptr;
    }

    if (dvppChannelDesc_ != nullptr) {
        aclError ret = acldvppDestroyChannel(dvppChannelDesc_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("acldvppDestroyChannel failed, ret = %d", ret);
        }

        acldvppDestroyChannelDesc(dvppChannelDesc_);
        dvppChannelDesc_ = nullptr;
    }
}

Result DvppProcess::Process()
{
    // pic decode
    Result ret = InitDecodeOutputDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("InitDecodeOutputDesc failed");
        DestroyDecodeResource();
        return FAILED;
    }

    ret = ProcessDecode();
    if (ret != SUCCESS) {
        ERROR_LOG("ProcessDecode failed");
        DestroyDecodeResource();
        return FAILED;
    }

    DestroyDecodeResource();

    // pic resize
    ret = InitResizeInputDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("InitResizeInputDesc failed");
        DestroyResizeResource();
        return FAILED;
    }

    ret = InitResizeOutputDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("InitResizeOutputDesc failed");
        DestroyResizeResource();
        return FAILED;
    }

    ret = ProcessResize();
    if (ret != SUCCESS) {
        ERROR_LOG("ProcessResize failed");
        DestroyResizeResource();
        return FAILED;
    }

    DestroyResizeResource();

    return SUCCESS;
}
