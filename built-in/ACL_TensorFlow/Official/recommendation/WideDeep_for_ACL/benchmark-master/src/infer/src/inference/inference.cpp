/*
 * License: Copyright (c) Huawei Technologies Co., Ltd. 2012-2019. All rights reserved.
 * Description: davinci inference module
 * Date: 2020-03-19 14:06:22
 * LastEditTime : 2020-04-17 11:51:44
 */
#include <vector>
#include <iostream>
#include <sstream>
#include <sys/time.h>
#include "inference.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl/acl_mdl.h"

using namespace std;

const float CONVERT_USEC = 1000000.0f;
const float CONVERT_MS = 1000.0f;

// default deviceID
InferBase::InferBase()
    : batchsize_(1),
      deviceId_(0),
      alive_(false),
      curTimeCost_(0.0),
      curExcuteCost_(0.0),
      curSampleNum_(0),
      modelId_(0),
      modelDesc_(nullptr),
      inputNum_(0),
      outputNum_(0),
      realNum_(0),
      wordNum_(0),
      finish_(false),
      inputQueue_(nullptr),
      outputQueue_(nullptr),
      moduleProcessTotalTime_(0),
      seqLenCount_(0)
{}

InferBase::InferBase(uint32_t batchsize, uint32_t deviceId, bool alive)
    : batchsize_(batchsize),
      deviceId_(deviceId),
      alive_(alive),
      curTimeCost_(0.0),
      curExcuteCost_(0.0),
      curSampleNum_(0),
      modelId_(0),
      modelDesc_(nullptr),
      inputNum_(0),
      outputNum_(0),
      realNum_(0),
      wordNum_(0),
      finish_(false),
      inputQueue_(nullptr),
      outputQueue_(nullptr),
      moduleProcessTotalTime_(0),
      seqLenCount_(0)
{}

InferBase::~InferBase() {}

bool InferBase::InitEnv(const string &cfgPath)
{
    context_ = context;

    return true;
}

bool InferBase::LoadModel(const string &omPath)
{
    aclError status;
    status = aclrtSetCurrentContext(context_);
    if (status != ACL_ERROR_NONE) {
        return false;
    }

    status = aclmdlLoadFromFile(omPath.c_str(), &modelId_);
    if (status != ACL_ERROR_NONE) {
        return false;
    }

    modelDesc_ = aclmdlCreateDesc();
    status = aclmdlGetDesc(modelDesc_, modelId_);
    if (status != ACL_ERROR_NONE) {
        return false;
    }

    inputNum_ = aclmdlGetNumInputs(modelDesc_);
    outputNum_ = aclmdlGetNumOutputs(modelDesc_);

    return true;
}

void InferBase::UnLoadModel()
{
    aclmdlUnload(modelId_);
    aclmdlDestroyDesc(modelDesc_);
}

bool InferBase::Init(const string &omPath, const string &cfgPath)
{
    bool ret;
    ret = InitEnv(cfgPath);
    if (ret == false) {
        cout << "[ERROR][Inference] InitEnv failed!" << endl;
        return false;
    }

    // load model
    ret = LoadModel(omPath);
    if (ret == false) {
        cout << "[ERROR][Inference] load model failed!" << endl;
        return false;
    }

    alive_ = true;
    finish_ = false;
    perfInfo_.reset(new PerfInfo);

    cout << "[INFO][Inference] Init SUCCESS" << endl;

    return true;
}

void InferBase::UnInit()
{
    alive_ = false;
    inputQueue_->Stop();
    job_.join();
    UnLoadModel();
    cout << "[INFO][Inference] UnInit SUCCESS" << endl;
}

// free dataset
void FreeDatasetMemory(aclmdlDataset *dataset)
{
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        aclrtFree(data);
        data = nullptr;
    }

    aclmdlDestroyDataset(dataset);
}

// difference model difference input Dataset, data must match the model batchsize! TODO:batchsize
bool InferBase::CreateInDatasetBatchsize(const vector<shared_ptr<ModelInputData>> &inDataBatchsize,
    shared_ptr<aclmdlDataset> &inDataset)
{
    aclError ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }

    aclmdlDataset *input = aclmdlCreateDataset();
    // k input tensor
    for (int k = 0; k < inputNum_; ++k) {
        uint32_t modelInputSize =
            aclmdlGetInputSizeByIndex(modelDesc_, k); // mutil input tensor, the order must be the same as omg
        uint32_t singleSize = modelInputSize / batchsize_;

        void *dst; // dst memory will be released by FreeDatasetMemory!
        ret = aclrtMalloc(&dst, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            return false;
        }

        ModelType mt;
        char *ptr = (char *)dst;
        for (int i = 0; i < inDataBatchsize.size(); ++i) {
            mt = inDataBatchsize[i]->modelType;
            uint32_t len;
            uint8_t *buf;
            if (mt == MT_VISION) {
                len = inDataBatchsize[i]->img.data.len;
                buf = inDataBatchsize[i]->img.data.buf.get(); // device memory
            } else if (mt == MT_NMT || mt == MT_WIDEDEEP) {
                len = inDataBatchsize[i]->text.textRawData[k].len;
                buf = inDataBatchsize[i]->text.textRawData[k].buf.get();
            } else {
                cout << "[ERROR][Inference] unknown model type!" << endl;
                return false;
            }

            if (len != singleSize) {
                cout << "[ERROR][Inference] input data size don't match the model input size!" << endl;
                return false;
            }

            void *dstTmp = (void *)ptr;
            ret = aclrtMemcpy(dstTmp, len, buf, len, ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (ret != ACL_ERROR_NONE) {
                aclrtFree(dst);
                dst = nullptr;
                return false;
            }
            ptr += len;
        }

        aclDataBuffer *inputData = nullptr;
        inputData = aclCreateDataBuffer(dst, modelInputSize);
        if (inputData == nullptr) {
            return false;
        }

        ret = aclmdlAddDatasetBuffer(input, inputData);
        if (ret != ACL_ERROR_NONE) {
            aclmdlDestroyDataset(input);
            return false;
        }
    }

    inDataset.reset(input, FreeDatasetMemory);

    return true;
}

bool InferBase::CreateOutDataset(shared_ptr<aclmdlDataset> &outDataset)
{
    aclError ret;
    aclmdlDataset *dataset = aclmdlCreateDataset();
    bool flag = false;
    vector<void *> outputDevPtrs;
    for (int i = 0; i < outputNum_; ++i) {
        uint64_t bufSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        void *outputBuf = nullptr;
        ret = aclrtMalloc(&outputBuf, (size_t)bufSize, ACL_MEM_MALLOC_HUGE_ONLY);
        if (ret != ACL_ERROR_NONE) {
            flag = true;
            break;
        }
        outputDevPtrs.push_back(outputBuf);

        aclDataBuffer *outputData = nullptr;
        outputData = aclCreateDataBuffer(outputBuf, bufSize);
        if (outputData == nullptr) {
            flag = true;
            break;
        }

        ret = aclmdlAddDatasetBuffer(dataset, outputData);
        if (ret != ACL_ERROR_NONE) {
            flag = true;
            break;
        }
    }

    if (flag == true) {
        for (auto ptr : outputDevPtrs) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
        aclmdlDestroyDataset(dataset);
        return false;
    }

    outDataset.reset(dataset, FreeDatasetMemory);

    return true;
}

bool InferBase::Process(shared_ptr<aclmdlDataset> inDataset, shared_ptr<aclmdlDataset> &outDataset)
{
    aclError ret;
    // infer execute
    struct timeval start, end;
    gettimeofday(&start, NULL);
    ret = aclmdlExecute(modelId_, inDataset.get(), outDataset.get());
    gettimeofday(&end, NULL);
    if (ret != ACL_ERROR_NONE) {
        return false;
    }
    float escapedTime = (end.tv_sec - start.tv_sec) * CONVERT_USEC + (end.tv_usec - start.tv_usec);
    escapedTime /= CONVERT_MS; // ms

    // the inference cost time of all samples
    curExcuteCost_ += escapedTime;

    // the cost time from thread start
    curTimeCost_ = (end.tv_sec - threadStart_.tv_sec) * CONVERT_USEC + (end.tv_usec - threadStart_.tv_usec);
    curTimeCost_ /= CONVERT_USEC; // second

    perfInfo_->inferLantency = curExcuteCost_ / curSampleNum_; // ms/image
    perfInfo_->throughputRate = curSampleNum_ / curTimeCost_;  // images/s

    return true;
}

bool InferBase::PushOutput(const shared_ptr<aclmdlDataset> &dataset)
{
    aclError ret;
    shared_ptr<ModelOutputData> out = make_shared<ModelOutputData>(); // output data of batch
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset.get()); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset.get(), i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSize(dataBuffer);
        void *outHostData = nullptr;
        ret = aclrtMallocHost(&outHostData, len);
        if (ret != ACL_ERROR_NONE) {
            continue;
        }

        ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            aclrtFreeHost(outHostData);
            continue;
        }

        stringstream ss;
        ss << i;
        string keyName;
        ss >> keyName;
        out->modelOutputData[keyName].buf.reset((uint8_t *)outHostData, [](uint8_t *p) { aclrtFreeHost(p); });
        out->modelOutputData[keyName].len = len;
    }
    out->realNum = realNum_;
    out->vDataId = vDataId_;
    out->finish = finish_;

    outputQueue_->Push(out);

    return true;
}

void InferBase::ThreadFunc()
{
    aclrtSetCurrentContext(context_);
    bool ret = false;
    vector<shared_ptr<ModelInputData>> inDataBatchsize;
    shared_ptr<aclmdlDataset> inDataset, outDataset;
    while (alive_) {
        shared_ptr<ModelInputData> input = nullptr;
        if (inDataBatchsize.size() < batchsize_) {
            ret = inputQueue_->Pop(input);
            if (!input) {
                continue;
            }
            if (input->modelType == MT_NMT) {
                void *seqLen = nullptr;
                int aret = aclrtMallocHost(&seqLen, sizeof(uint32_t));
                if (aret != ACL_ERROR_NONE) {
                    cout << "[ERROR][Inference] aclrtMallocHost failed!" << endl;
                    continue;
                }
                aret = aclrtMemcpy(seqLen, sizeof(uint32_t), input->text.textRawData[1].buf.get(), sizeof(uint32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
                if (aret != ACL_ERROR_NONE) {
                    aclrtFreeHost(seqLen);
                    cout << "[ERROR][Inference] aclrtMemcpy failed!" << endl;
                    continue;
                }
                seqLenCount_ += *((uint32_t *)seqLen);
                aclrtFreeHost(seqLen);
            }
            inDataBatchsize.push_back(input);
            vDataId_.push_back(input->dataId);
            if (!(input->finish)) {
                continue;
            }
            finish_ = input->finish;
        }
        // cal the module average process time start, the blocking queue waiting time is not included.
        struct timeval moduleProcessStart, moduleProcessEnd;
        gettimeofday(&moduleProcessStart, NULL);

        realNum_ = inDataBatchsize.size();
        if (seqLenCount_ > 0) {
            curSampleNum_ += seqLenCount_;
            seqLenCount_ = 0;
        } else {
            curSampleNum_ += realNum_;
        }

        ret = CreateInDatasetBatchsize(inDataBatchsize, inDataset);
        if (ret == false) {
            inDataBatchsize.clear();
            continue;
        }

        ret = CreateOutDataset(outDataset);
        if (ret == false) {
            inDataBatchsize.clear();
            continue;
        }

        ret = Process(inDataset, outDataset);
        if (ret == false) {
            inDataBatchsize.clear();
            continue;
        }

        // push the outDataset to queue, need to create the output queue!
        PushOutput(outDataset);

        gettimeofday(&moduleProcessEnd, NULL);
        moduleProcessTotalTime_ += (moduleProcessEnd.tv_sec - moduleProcessStart.tv_sec) * CONVERT_MS +
            (moduleProcessEnd.tv_usec - moduleProcessStart.tv_usec) / CONVERT_MS;
        perfInfo_->moduleLantency = moduleProcessTotalTime_ / *(vDataId_.end() - 1); // ms

        // clear all elements!
        vDataId_.clear();
        inDataBatchsize.clear();
    }
}

void InferBase::Run(BlockInputQueue *inQueue, BlockOutputQueue *&outQueue)
{
    inputQueue_ = inQueue;
    outputQueue_ = outQueue;

    gettimeofday(&threadStart_, NULL);
    job_ = thread(&InferBase::ThreadFunc, this);
}

shared_ptr<PerfInfo> InferBase::GetPerfInfo()
{
    return perfInfo_;
}