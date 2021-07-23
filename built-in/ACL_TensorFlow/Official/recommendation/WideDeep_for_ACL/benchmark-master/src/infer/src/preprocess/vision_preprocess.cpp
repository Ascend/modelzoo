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

#include "vision_preprocess.h"
#include "string_hash.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sys/time.h>
#include "securec.h"
#define STR_MAX_LEN 128

const int TIMES_SECOND_MICROSECOND = 1000000;
VisionPreProcess::VisionPreProcess() : context_(nullptr), stream_(nullptr), processDvpp(nullptr) {}

VisionPreProcess::~VisionPreProcess()
{
    DestroyResource();
}

int VisionPreProcess::Init(int input_width, int input_height)
{
    modelInputWidth = input_width;
    modelInputHeight = input_height;
    context_ = context;//from glibale variable
    // create stream
    aclrtSetCurrentContext(context_);
    int ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    // alloc new dvpp device
    processDvpp = new DvppProcess(stream_);
    // Result
    ret = processDvpp->InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("init dvpp resource failed");
        return FAILED;
    }

	perfInfo_ = std::make_shared<PerfInfo>();
	
	initialTimeStamp = GetCurentTimeStamp();
    INFO_LOG("Init success");

    return SUCCESS;
}

void VisionPreProcess::DeInit()
{
    isStop_ = true;
    inputQueuePtr_->Stop();
    processThr_.join();
    INFO_LOG("DeInit success");
}


void *VisionPreProcess::GetDeviceBufferOfPicture(const char *inputHostBuff, const uint32_t inputHostBuffSize,
    uint32_t &devPicBufferSize)
{
    if (inputHostBuff == nullptr) {
        return nullptr;
    }

    void *inBufferDev = nullptr;
    uint32_t inBufferSize = inputHostBuffSize;
    aclError ret = acldvppMalloc(&inBufferDev, inBufferSize);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc device buffer failed. size is %u", inBufferSize);
        // delete[] inputHostBuff;
        return nullptr;
    }

    ret = aclrtMemcpy(inBufferDev, inBufferSize, inputHostBuff, inputHostBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("memcpy failed. device buffer size is %u, input host buffer size is %u", inBufferSize,
            inputHostBuffSize);
        acldvppFree(inBufferDev);
        // delete[] inputHostBuff;
        return nullptr;
    }

    // delete[] inputHostBuff;
    devPicBufferSize = inBufferSize;
    return inBufferDev;
}

uint64_t VisionPreProcess::GetCurentTimeStamp()
{
	gettimeofday(&currentTimeval, NULL);
	return currentTimeval.tv_sec * TIMES_SECOND_MICROSECOND + currentTimeval.tv_usec;
}

void VisionPreProcess::ProcessThread()
{
    aclrtSetCurrentContext(context_);
    while (!isStop_) {
        std::shared_ptr<RawData> inputRawData = nullptr;
        inputQueuePtr_->Pop(inputRawData);
        if (!inputRawData) {
            continue;
        }

        uint32_t width = inputRawData->img.width;
        uint32_t height = inputRawData->img.height;

        char *inputHostBuff = (char *)inputRawData->img.data.buf.get();
        uint32_t inputHostBuffSize = inputRawData->img.data.len;

        uint32_t devPicBufferSize;
        void *picDevBuffer = GetDeviceBufferOfPicture(inputHostBuff, inputHostBuffSize, devPicBufferSize);
        if (picDevBuffer == nullptr) {
            ERROR_LOG("get pic device buffer failed");
            continue;
        }

        // set the deivce input buff width*height
        processDvpp->SetInput(picDevBuffer, devPicBufferSize, width, height);

        // init the dvpp output deivce buff modelInputWidth*modelInputHeight
        int ret;
        ret = processDvpp->InitOutputPara(modelInputWidth, modelInputHeight);
        if (ret != SUCCESS) {
            ERROR_LOG("init dvpp output para failed");
            acldvppFree(picDevBuffer);
            picDevBuffer = nullptr;
            continue;
        }

        ret = processDvpp->Process();
        if (ret != SUCCESS) {
            ERROR_LOG("dvpp process failed");
            acldvppFree(picDevBuffer);
            picDevBuffer = nullptr;
            continue;
        }
        
        // release the picDevBuffer
        (void)acldvppFree(picDevBuffer);
        picDevBuffer = nullptr;
        // get the dvpp out put mem
        void *dvppOutputBuffer = nullptr;
        int dvppOutputSize;
        processDvpp->GetOutput(&dvppOutputBuffer, dvppOutputSize);

        // init ModelInputData
        std::shared_ptr<ModelInputData> imgnetData = std::make_shared<ModelInputData>();
        imgnetData->dataId = inputRawData->dataId;
        imgnetData->modelType = inputRawData->modelType;
        // init the text data
        imgnetData->img.height = height;
        imgnetData->img.width = width;
        imgnetData->finish = inputRawData->finish;
        imgnetData->img.data.buf.reset((uint8_t *)dvppOutputBuffer, [](uint8_t *p) { aclrtFree(p); });
        imgnetData->img.data.len = dvppOutputSize;

        outputQueuePtr_->Push(imgnetData);

        // perf info
        perfInfo_->throughputRate = inputRawData->dataId / (1.0 * (GetCurentTimeStamp() - initialTimeStamp) / 
            TIMES_SECOND_MICROSECOND);
        perfInfo_->moduleLantency = 1.0 / perfInfo_->throughputRate * 1000; // ms
    }
}

void VisionPreProcess::Process(BlockingQueue<std::shared_ptr<RawData>> *inputQueuePtr,
    BlockingQueue<std::shared_ptr<ModelInputData>> *outputQueuePtr)
{
    inputQueuePtr_ = inputQueuePtr;
    outputQueuePtr_ = outputQueuePtr;
    processThr_ = std::thread(&VisionPreProcess::ProcessThread, this);
}

void VisionPreProcess::DestroyResource()
{
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    // delete dvpp device
    delete processDvpp;
}

std::shared_ptr<PerfInfo> VisionPreProcess::GetPerfInfo()
{
	return perfInfo_;
}
