/**
* @file sample_process.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "sample_process.h"
#include <iostream>
#include "acl/acl.h"
#include "utils.h"
using namespace std;
extern bool g_isDevice;


SampleProcess::SampleProcess() :deviceId_(0), context_(nullptr), stream_(nullptr)
{
}

SampleProcess::~SampleProcess()
{
}

Result SampleProcess::InitResource(char* omModelPath)
{
    // ACL init
    const char *aclConfigPath;
    if((aclConfigPath = getenv("ACL_CONFIG_PATH"))) {
        INFO_LOG("acl init with config file:%s", aclConfigPath);
    } else {
        aclConfigPath = "";
    }
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

    // get run mode
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("get run mode success");
    Result r_ret = processModel.LoadModelFromFileWithMem(omModelPath);
    if (r_ret != SUCCESS) {
        ERROR_LOG("load model from file failed");
        return FAILED;
    }

    ret = processModel.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    ret = processModel.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("create model output failed");
        return FAILED;
    }

    return SUCCESS;
}

Result SampleProcess::Process(void* binfile,int len)
{
    //for (size_t index = 0; index < sizeof(testFile) / sizeof(testFile[0]); ++index) {
        //INFO_LOG("start to process file:%s", binfile.c_str());
        // model process
        //INFO_LOG("start memcpy is %d", Utils::getCurrentTime());
        uint32_t devBufferSize = len;
        void *picDevBuffer = Utils::GetDeviceBufferOfptr(binfile,len);
        //ERROR_LOG("devBufferSize is %d", len);
        //ERROR_LOG("picDevBuffer:%f.", *((float*)picDevBuffer));
        if (picDevBuffer == nullptr) {
            ERROR_LOG("get pic device buffer failed,index is %zu", 0);
            return FAILED;
        }
        //INFO_LOG("end memcpy is %d", Utils::getCurrentTime());

        //INFO_LOG("start CreateInput is %d", Utils::getCurrentTime());
        Result ret = processModel.CreateInput(picDevBuffer, devBufferSize);
        if (ret != SUCCESS) {
            ERROR_LOG("model create input failed");
            aclrtFree(picDevBuffer);
            return FAILED;
        }
        //INFO_LOG("end CreateInput is %d", Utils::getCurrentTime());

        INFO_LOG("start Execute is %d", Utils::getCurrentTime());
        ret = processModel.Execute();
        if (ret != SUCCESS) {
            ERROR_LOG("model execute failed");
            aclrtFree(picDevBuffer);
            return FAILED;
        }
        INFO_LOG("end Execute is %d", Utils::getCurrentTime());

        //INFO_LOG("start aclrtFree is %d", Utils::getCurrentTime());
        // release model input buffer
        aclrtFree(picDevBuffer);
        //INFO_LOG("end aclrtFree is %d", Utils::getCurrentTime());

        // print the top 5 confidence values with indexes.use function DumpModelOutputResult
        // if want to dump output result to file in the current directory
        processModel.OutputModelResult();

    //}
    // loop end


    return SUCCESS;
}

Result SampleProcess::Process(void* binfile,int len, vector<Output_buf> &output, long &npuTime)
{
    //for (size_t index = 0; index < sizeof(testFile) / sizeof(testFile[0]); ++index) {
        //INFO_LOG("start to process file:%s", binfile.c_str());
        // model process
        //INFO_LOG("start memcpy is %d", Utils::getCurrentTime());
        uint32_t devBufferSize = len;
        void *picDevBuffer = Utils::GetDeviceBufferOfptr(binfile,len);
        //ERROR_LOG("devBufferSize is %d", len);
        //ERROR_LOG("picDevBuffer:%f.", *((float*)picDevBuffer));
        if (picDevBuffer == nullptr) {
            ERROR_LOG("get pic device buffer failed,index is %zu", 0);
            return FAILED;
        }
        //INFO_LOG("end memcpy is %d", Utils::getCurrentTime());

        //INFO_LOG("start CreateInput is %d", Utils::getCurrentTime());
        Result ret = processModel.CreateInput(picDevBuffer, devBufferSize);
        if (ret != SUCCESS) {
            ERROR_LOG("model create input failed");
            aclrtFree(picDevBuffer);
            return FAILED;
        }
        //INFO_LOG("end CreateInput is %d", Utils::getCurrentTime());

        INFO_LOG("start Execute is %d", Utils::getCurrentTime());
        long start = Utils::getCurrentTime();
        ret = processModel.Execute();
        if (ret != SUCCESS) {
            ERROR_LOG("model execute failed");
            aclrtFree(picDevBuffer);
            return FAILED;
        }
        long end = Utils::getCurrentTime();
        INFO_LOG("end Execute is %d", end);
        
        npuTime = end - start;
        if (npuTime < 0) {
            ERROR_LOG("NPU time not correct: %d start: %d end: %d", npuTime, start, end);
            aclrtFree(picDevBuffer);
            return FAILED;
        }
        INFO_LOG("npu compute cost %f ms", 1.0*(npuTime)/1000.0);

        //INFO_LOG("start aclrtFree is %d", Utils::getCurrentTime());
        // release model input buffer
        aclrtFree(picDevBuffer);
        //INFO_LOG("end aclrtFree is %d", Utils::getCurrentTime());

        // print the top 5 confidence values with indexes.use function DumpModelOutputResult
        // if want to dump output result to file in the current directory
        processModel.OutputModelResult(output);

    //}
    // loop end


    return SUCCESS;
}

Result SampleProcess::Unload()
{
    processModel.Unload();
    processModel.DestroyDesc();
    processModel.DestroyInput();
    processModel.DestroyOutput();
    DestroyResource();
    return SUCCESS;
}

void SampleProcess::DestroyResource()
{
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
        INFO_LOG("end to destroy stream");
    }

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
        INFO_LOG("end to destroy context");
    }

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
