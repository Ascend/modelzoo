/**
* @file model_process.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "model_process.h"
#include <iostream>
#include <map>
#include <sstream>
#include <algorithm>
#include "utils.h"
using namespace std;
extern bool g_isDevice;
//extern Config configSettings;
extern std::map<aclDataType, std::string> ACLdt;
extern std::map<aclDataType, int> ACLdt_size;

ModelProcess::ModelProcess() :modelId_(0), modelMemSize_(0), modelWeightSize_(0), modelMemPtr_(nullptr),
modelWeightPtr_(nullptr), loadFlag_(false), modelDesc_(nullptr), input_(nullptr), output_(nullptr)
{
}

ModelProcess::~ModelProcess()
{
}



Result ModelProcess::LoadModelFromFileWithMem(const char *modelPath)
{
    if (loadFlag_) {
        ERROR_LOG("has already loaded a model");
        return FAILED;
    }

    aclError ret = aclmdlQuerySize(modelPath, &modelMemSize_, &modelWeightSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("query model failed, model file is %s", modelPath);
        return FAILED;
    }
    //int mem_malloc_type=configSettings.Read("aclrtMemMallocPolicy", 0);
    ret = aclrtMalloc(&modelMemPtr_, modelMemSize_, (aclrtMemMallocPolicy)(Config::getInstance()->Read("aclrtMemMallocPolicy", 0)));
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc buffer for mem failed, require size is %zu", modelMemSize_);
        return FAILED;
    }

    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, (aclrtMemMallocPolicy)(Config::getInstance()->Read("aclrtMemMallocPolicy", 0)));
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu", modelWeightSize_);
        return FAILED;
    }

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelMemPtr_,
        modelMemSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("load model from file failed, model file is %s", modelPath);
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath);
    return SUCCESS;
}

Result ModelProcess::CreateDesc()
{
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("get model description failed");
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyDesc()
{
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
}


Result ModelProcess::CreateInput(void *inputDataBuffer, size_t bufferSize)
{
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    //INFO_LOG("start CreateInput::aclCreateDataBuffer is %d", Utils::getCurrentTime());
    aclDataBuffer* inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    //INFO_LOG("end CreateInput::aclCreateDataBuffer is %d", Utils::getCurrentTime());
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }
    //INFO_LOG("start CreateInput::aclmdlAddDatasetBuffer is %d", Utils::getCurrentTime());
    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    //INFO_LOG("end CreateInput::aclmdlAddDatasetBuffer is %d", Utils::getCurrentTime());
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("add input dataset buffer failed");
        aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }

    return SUCCESS;
}

void ModelProcess::DestroyInput()
{
    if (input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        aclDestroyDataBuffer(dataBuffer);
    }
    aclmdlDestroyDataset(input_);
    input_ = nullptr;
}

Result ModelProcess::CreateOutput()
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create ouput failed");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(modelDesc_, i);

        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, buffer_size, (aclrtMemMallocPolicy)(Config::getInstance()->Read("aclrtMemMallocPolicy", 0)));
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", buffer_size);
            return FAILED;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't create data buffer, create output failed");
            aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't add data buffer, create output failed");
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    INFO_LOG("create model output success");
    return SUCCESS;
}

void ModelProcess::DumpModelOutputResult()
{
    stringstream ss;
    size_t outputNum = aclmdlGetDatasetNumBuffers(output_);
    static int executeNum = 0;
    for (size_t i = 0; i < outputNum; ++i) {
        ss << "output" << ++executeNum << "_" << i << ".bin";
        string outputFileName = ss.str();
        FILE *outputFile = fopen(outputFileName.c_str(), "wb");
        if (outputFile) {
            aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
            void* data = aclGetDataBufferAddr(dataBuffer);
            uint32_t len = aclGetDataBufferSize(dataBuffer);

            void* outHostData = NULL;
            aclError ret = ACL_ERROR_NONE;
            if (!g_isDevice) {
                ret = aclrtMallocHost(&outHostData, len);
                if (ret != ACL_ERROR_NONE) {
                    ERROR_LOG("aclrtMallocHost failed, ret[%d]", ret);
                    return;
                }

                ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_ERROR_NONE) {
                    ERROR_LOG("aclrtMemcpy failed, ret[%d]", ret);
                    (void)aclrtFreeHost(outHostData);
                    return;
                }

                fwrite(outHostData, len, sizeof(char), outputFile);

                ret = aclrtFreeHost(outHostData);
                if (ret != ACL_ERROR_NONE) {
                    ERROR_LOG("aclrtFreeHost failed, ret[%d]", ret);
                    return;
                }
            } else {
                fwrite(data, len, sizeof(char), outputFile);
            }
            fclose(outputFile);
            outputFile = nullptr;
        } else {
            ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
            return;
        }
    }

    INFO_LOG("dump data success");
    return;
}

void ModelProcess::OutputModelResult()
{

    aclError ret = ACL_ERROR_NONE;

    if(outputdata.size()!=0)
    {
        for(auto it=outputdata.begin();it!=outputdata.end();++it){
                ret = aclrtFreeHost(*it);
            }
            outputdata.clear();
    }
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSize(dataBuffer);
        //INFO_LOG("OutputModelResult::len %d", len);
        //void *outHostData = NULL;
        outputdata.push_back(NULL);
         ret = ACL_ERROR_NONE;
        float *outData = NULL;
        if (!g_isDevice) {
            //INFO_LOG("start OutputModelResult::aclrtMallocHost is %d", Utils::getCurrentTime());
            aclError ret = aclrtMallocHost(&outputdata[i], len);
            //INFO_LOG("end OutputModelResult::aclrtMallocHost is %d", Utils::getCurrentTime());
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtMallocHost failed, ret[%d]", ret);
                return;
            }
            //INFO_LOG("start OutputModelResult::aclrtMemcpy is %d", Utils::getCurrentTime());
            ret = aclrtMemcpy(outputdata[i], len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            //INFO_LOG("end OutputModelResult::aclrtMemcpy is %d", Utils::getCurrentTime());
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtMemcpy failed, ret[%d]", ret);
                return;
            }

            outData = reinterpret_cast<float*>(outputdata[i]);
            //outputdata.push_back(outHostData);
        } else {
            outData = reinterpret_cast<float*>(data);
            outputdata.push_back(data);
        }
        map<float, unsigned int, greater<float> > resultMap;
        for (unsigned int j = 0; j < len / sizeof(float); ++j) {
            resultMap[*outData] = j;
            outData++;
        }

        int cnt = 0;
        for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
            // print top 5
            if (++cnt > 5) {
                break;
            }

            //INFO_LOG("top %d: index[%d] value[%lf]", cnt, it->second, it->first);
        }
    }

    INFO_LOG("1.output data success");
    return;
}

void ModelProcess::OutputModelResult(vector<Output_buf> &output)
{

    OutputModelResult();

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);


    for (size_t i = 0; i < outputSize; ++i) {
        Output_buf tmp_output;
        //ptr
        tmp_output.ptr = outputdata[i];
        //size
        tmp_output.size = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        //ndim & shape
        aclmdlIODims d={{'a'},0,{1,2,3}};
        aclmdlIODims *dims=&d;
        aclError ret = aclmdlGetOutputDims((const aclmdlDesc *)modelDesc_, i, dims);
        tmp_output.ndim = (*dims).dimCount;
        for(int j=0;j<(*dims).dimCount;j++)
            tmp_output.shape.push_back((*dims).dims[j]);

        //itemsize & format
        aclDataType dt =  aclmdlGetOutputDataType(modelDesc_, i);
        tmp_output.itemsize=ACLdt_size[dt];
        tmp_output.format=ACLdt[dt];
//        if(dt==ACL_FLOAT)
//        {
//            tmp_output.itemsize=4;
//            tmp_output.format="float";
//        }
//        if(dt==ACL_FLOAT16)
//        {
//            tmp_output.itemsize=2;
//            tmp_output.format="float16";
//        }
//        if(dt==ACL_UINT8)
//        {
//            tmp_output.itemsize=1;
//            tmp_output.format="uint8";
//        }
//        if(dt==ACL_INT8)
//        {
//            tmp_output.itemsize=1;
//            tmp_output.format="int8";
//        }

        //strides
        int strides=tmp_output.itemsize;
        std::vector<int64_t> tmp_strides;
        std::vector<int64_t> tmp1_strides;
        //vector<int64_t>::reverse_iterator it;//声明一个迭代器，来访问vector容器，作用：遍历或者指向vector容器的元素
        for(auto it=tmp_output.shape.rbegin();it!=tmp_output.shape.rend();++it)
        {
            tmp_strides.push_back(strides);
            strides = strides * (*it);
        }
        //for (int index = 0 ; index < tmp_output.ndim; index++) {
        //   tmp_strides.push_back(strides);
        //   strides = strides * tmp_output.shape[index];
        //}
        for(auto it=tmp_strides.rbegin();it!=tmp_strides.rend();++it){
        //for (int index=0;index < tmp_strides.size(); index++) {
           tmp1_strides.push_back(*it);
        }
        tmp_output.strides = tmp1_strides;
        output.push_back(tmp_output);
    }
    INFO_LOG("2.output data success");
}
void ModelProcess::DestroyOutput()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
}

Result ModelProcess::Execute()
{
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }

    INFO_LOG("model execute success");
    return SUCCESS;
}

void ModelProcess::Unload()
{
    aclError ret;
    if (!g_isDevice) {
            //INFO_LOG("start Unload::aclrtFreeHost is %d", Utils::getCurrentTime());
            //for (int index = 0 ; index < outputdata.size(); index++) {
            //    ret = aclrtFreeHost(outputdata[index]);
            //}
            for(auto it=outputdata.begin();it!=outputdata.end();++it){
                ret = aclrtFreeHost(*it);
            }
            //INFO_LOG("end Unload::aclrtFreeHost is %d", Utils::getCurrentTime());
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtFreeHost failed, ret[%d]", ret);
                return;
            }
        }
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    ret = aclmdlUnload(modelId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelMemPtr_ != nullptr) {
        aclrtFree(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
}
