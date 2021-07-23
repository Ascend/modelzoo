/*
 * License: Copyright (c) Huawei Technologies Co., Ltd. 2012-2019. All rights reserved.
 * Description: davinci postprocess module
 * Date: 2020-03-19
 * LastEditTime : 2020-04-18
 */
#ifndef BANCHMARK_POSTPROCESS_H
#define BANCHMARK_POSTPROCESS_H

#include <vector>
#include <map>
#include <utility>
#include <memory>
#include <thread>
#include "HashTable.h"
#include "common/block_queue.h"
#include "common/data_struct.h"

namespace {
    using PostBlockQueue = BlockingQueue<std::shared_ptr<ModelOutputData>>;
    const float CONVERT_USEC = 1000000.0f;
    const float CONVERT_MS = 1000.0f;
    const int32_t EOS_VALUE = 2;
    const std::string EOS = "</s>";
}

class PostProcess {
public:
    PostProcess();
    ~PostProcess();
    bool Init(const uint32_t batchsize, const ModelType modelType, const std::string& cfgFile, const bool flag);
    bool DeInit();
    bool Run(PostBlockQueue* deviceInput, PostBlockQueue* hostInput);
    std::shared_ptr<PerfInfo> GetPerfInfo();
    bool GetFinishFlag();

private:
    bool InitVision(const std::string& cfgFile);
    bool InitNLP(const std::string& cfgFile);
    bool InitFastRCNN(const std::string& cfgFile);
    bool InitNMT(const std::string& cfgFile);
    bool InitWideDeep(const std::string& cfgFile);
    void ThreadProc();
    bool ProcessVision();
    bool VisionCore(std::shared_ptr<ModelOutputData>& pDeviceInputInfo);
    bool VisionDumpResult(const std::string &fileName, const float* dataBuf, const uint32_t& len);
    bool ProcessNLP();
    bool ProcessFastRCNN();
    bool ProcessNMT();
    bool ProcessWideDeep();
    bool WideDeepCore(std::shared_ptr<ModelOutputData>& pDeviceInputInfo);
    bool FuseWideDeep(uint32_t dataLen, float* pDeviceBuf, float* pHostBuf);
    bool SaveToTxtFile(const std::string& fileName, const std::string& context, const bool isAppend);
    bool SaveToBinaryFile(const std::string& fileName, const void* context, size_t size, const bool isAppend);

private:
    using ProcessFunc = bool(PostProcess::*)();

private:
    uint32_t batchSize_;
    ModelType modelType_;
    bool isKeepAlive_;
    std::thread processThr_;
    PostBlockQueue* deviceInputQue_;
    PostBlockQueue* hostInputQue_;
    std::string cfgFile_;
    ProcessFunc processFunc_;
    std::map<uint64_t, std::string> frameInfoMap_;
    std::unordered_map<size_t, std::string> hashTab_;
    uint32_t totalCnt_;
    uint32_t rightCnt_;
    std::vector<uint8_t> vecFlag_;
    double curTimeCost_;
    uint64_t curSampleNum_;
    struct timeval threadStart_;
    std::shared_ptr<PerfInfo> perfInfo_;
    bool isFinished_;
    bool isOutputBinary_;
};

#endif