/*
 * License: Copyright (c) Huawei Technologies Co., Ltd. 2012-2019. All rights reserved.
 * Description: davinci inference module
 * Date: 2020-03-19 11:32:40
 * LastEditTime : 2020-04-17 11:22:03
 */
#ifndef BANCHMARK_INFERENCE_H
#define BANCHMARK_INFERENCE_H

#include <string>
#include <thread>
#include <memory>
#include <sys/time.h>
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "../common/block_queue.h"
#include "../common/data_struct.h"

using namespace std;

using BlockInputQueue = BlockingQueue<shared_ptr<ModelInputData>>;
using BlockOutputQueue = BlockingQueue<shared_ptr<ModelOutputData>>;

class InferBase {
public:
    InferBase();
    InferBase(uint32_t batchsize, uint32_t deviceId, bool alive = false);
    ~InferBase();
    bool Init(const string& omPath, const string& cfgPath = "");
    void Run(BlockInputQueue *inQueue, BlockOutputQueue *&outQueue);
    void UnInit();

    shared_ptr<PerfInfo> GetPerfInfo();

protected:
    bool InitEnv(const string& cfgPath = "");
    bool LoadModel(const string& omPath);
    void UnLoadModel();
    bool CreateOutDataset(shared_ptr<aclmdlDataset>& outDataset);
    // diffence model diffence input
    bool CreateInDatasetBatchsize(const vector<shared_ptr<ModelInputData>>& inDataBatchsize, shared_ptr<aclmdlDataset>& inDataset);
    bool PushOutput(const shared_ptr<aclmdlDataset>& dataset);
    bool Process(shared_ptr<aclmdlDataset> inDataset, shared_ptr<aclmdlDataset>& outDataset);
    void ThreadFunc(); // thread func calling Run()

    uint32_t deviceId_;
    aclrtContext context_;
    uint32_t modelId_;
    aclmdlDesc* modelDesc_;
    uint32_t inputNum_;
    uint32_t outputNum_;
    uint32_t batchsize_;
    uint32_t realNum_;
    uint32_t wordNum_;
    vector<uint64_t> vDataId_;
    bool finish_;

    BlockInputQueue* inputQueue_;
    BlockOutputQueue* outputQueue_;
    bool alive_;
    thread job_;

    // performce
    double curTimeCost_;
    double curExcuteCost_;
    uint64_t curSampleNum_;
    shared_ptr<PerfInfo> perfInfo_;
    struct timeval threadStart_;
    double moduleProcessTotalTime_;
    uint64_t seqLenCount_;
};



#endif // BANCHMARK_INFERENCE_H