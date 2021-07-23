/*
 * License: Copyright (c) Huawei Technologies Co., Ltd. 2012-2019. All rights reserved.
 * Description: davinci postprocess module
 * Date: 2020-03-19
 * LastEditTime : 2020-04-18
 */
#include "postprocess.h"
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>

PostProcess::PostProcess()
    : batchSize_(0), modelType_(MT_INVALID), isKeepAlive_(true), deviceInputQue_(nullptr),
      hostInputQue_(nullptr), cfgFile_(""), processFunc_(nullptr), totalCnt_(0), rightCnt_(0),
      curTimeCost_(0.0), curSampleNum_(0), isFinished_(false), isOutputBinary_(false) {}

PostProcess::~PostProcess() {}

bool PostProcess::Init(const uint32_t batchsize, const ModelType modelType, const std::string& cfgFile, const bool flag)
{
    if (batchsize <= 0 || modelType >= MT_INVALID || cfgFile == "") {
        std::cout << "[PostProcess] Init params is error." << std::endl;
        return false;
    }

    batchSize_ = batchsize;
    modelType_ = modelType;
    cfgFile_ = cfgFile;
    perfInfo_ = std::make_shared<PerfInfo>();
    perfInfo_->moduleLantency = 0.0;
    perfInfo_->throughputRate = 0.0;
    isOutputBinary_ = flag;

    bool status = false;

    switch (modelType_) {
        case MT_VISION:
            processFunc_ = &PostProcess::ProcessVision;
            status = InitVision(cfgFile_);
            break;

        case MT_NLP:
            processFunc_ = &PostProcess::ProcessNLP;
            status = InitNLP(cfgFile_);
            break;

        case MT_FASTERRCNN:
            processFunc_ = &PostProcess::ProcessFastRCNN;
            status = InitFastRCNN(cfgFile_);
            break;

        case MT_NMT:
            processFunc_ = &PostProcess::ProcessNMT;
            status = InitNMT(cfgFile_);
            break;

        case MT_WIDEDEEP:
            processFunc_ = &PostProcess::ProcessWideDeep;
            status = InitWideDeep(cfgFile_);
            break;

        default:
            std::cout << "[PostProcess] Do not support model type(" << modelType_ << ")." << std::endl;
            status = false;
    }

    if (status) {
        std::cout << "[INFO][PostProcess] init SUCCESS" << std::endl;
    } else {
        std::cout << "[ERROR][PostProcess] init failed." << std::endl; 
    }

    return status;
}

bool PostProcess::DeInit()
{
    if (hostInputQue_) {
        hostInputQue_->Stop();
    }
    if (deviceInputQue_) {
        deviceInputQue_->Stop();
    }

    isKeepAlive_ = false;
    processThr_.join();

    std::cout << "[INFO][PostProcess] Deinit SUCCESS" << std::endl;

    return true;
}

bool PostProcess::InitVision(const std::string& cfgFile)
{
    std::ifstream in(cfgFile_);
    if (!in.is_open()) {
        std::cout << "[ERROR][PostProcess] Can not open " << cfgFile_ << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line == "") {
            continue; // 跳过空行
        }

        uint64_t dataId;
        std::string dataName = "";
        uint32_t width = 0;
        uint32_t height = 0; 
        std::istringstream strm(line);
        strm >> dataId >> dataName >> width >> height;
        size_t pos = dataName.find_last_of('.');
        std::string fileName = dataName.substr(0, pos);
        auto ret = frameInfoMap_.insert(std::make_pair(dataId, fileName));
        if (!ret.second) {
            std::cout << "[ERROR][PostProcess] frameInfoMap insert error." << std::endl;
            return false;
        }
    }

    return true;
}

bool PostProcess::InitNLP(const std::string& cfgFile)
{
    return true;
}

bool PostProcess::InitFastRCNN(const std::string& cfgFile)
{
    return true;
}

bool PostProcess::InitNMT(const std::string& cfgFile)
{
    std::ifstream in(cfgFile);
    if (!in.is_open()) {
        std::cout << "[ERROR][PostProcess] Can not open " << cfgFile_ << std::endl;
        return false;
    }

    size_t index = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line == "") {
            continue; // 跳过空行
        }

        hashTab_[index] = line;
        ++index;
    }

    auto iter = hashTab_.find(EOS_VALUE);
    if (iter == hashTab_.end() || iter->second != EOS) {
        std::cout << "[ERROR][PostProcess] parse nmt cfgFile failed." << std::endl;
        return false;
    }

    // 清空已存在的nmt_output_file.txt
    std::ofstream out("nmt_output_file.txt");
    if (!out.is_open()) {
        std::cout << "[ERROR][PostProcess] Can not open nmt_output_file.txt" << std::endl;
        return false;
    }
    out.close();

    return true;
}

bool PostProcess::InitWideDeep(const std::string& cfgFile)
{
    const std::string moreThan50K = ">50K";
    const std::string lessThan50K = "<=50K";
    std::ifstream in(cfgFile_);
    if (!in.is_open()) {
        std::cout << "[ERROR][PostProcess] Can not open " << cfgFile_ << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line == "") {
            continue; // 跳过空行
        }

        size_t pos = line.find_last_of(',');
        std::string str = line.substr(pos + 1);
        if (str.find(moreThan50K) != std::string::npos) {
            vecFlag_.push_back(1);
        } else if (str.find(lessThan50K) != std::string::npos) {
            vecFlag_.push_back(0);
        } else {
            std::cout << "[ERROR][PostProcess] parse wide deep cfgFile failed." << std::endl;
            vecFlag_.clear();
            return false;
        }
    }

    return true;
}

bool PostProcess::ProcessVision()
{
    std::shared_ptr<ModelOutputData> pInputInfo = nullptr;
    deviceInputQue_->Pop(pInputInfo);
    if (!pInputInfo) {
        return true;
    }

    // 校验输入数据的合法性
    if (pInputInfo->realNum <= 0 || pInputInfo->realNum != pInputInfo->vDataId.size() ||
        pInputInfo->modelOutputData.size() < 1) {
        std::cout << "[ERROR][PostProcess] vision_postprocess input is error." << std::endl;
        return false;
    }

    // 性能统计使用
    curSampleNum_ += pInputInfo->realNum;
    std::cout << "[INFO][PostProcess] curSampleNum is " << curSampleNum_ << std::endl;

    for (auto iter = pInputInfo->modelOutputData.cbegin(); iter != pInputInfo->modelOutputData.cend(); ++iter) {
        uint32_t byteSize = iter->second.len;
        uint8_t* byteBuf = iter->second.buf.get();

        // 深入校验输入数据的合法性
        if (byteSize % batchSize_ || byteSize % sizeof(float) || byteBuf == nullptr) {
            std::cout << "[ERROR][PostProcess] vision_postprocess data is error." << std::endl;
            return false;
        }
    }

    if (!VisionCore(pInputInfo)) {
        std::cout << "[ERROR][PostProcess] VisionCore failed." << std::endl;
        return false;
    }

    isFinished_ = pInputInfo->finish;

    return true;
}

bool PostProcess::VisionCore(std::shared_ptr<ModelOutputData>& pInputInfo)
{
    uint32_t index = 1;
    std::map<std::string, DataBuf> &mapOut = pInputInfo->modelOutputData;
    for (auto iter = mapOut.cbegin(); iter != mapOut.cend(); ++iter) {
        uint32_t byteSize = iter->second.len;
        uint8_t* byteBuf = iter->second.buf.get();
        uint32_t frameLen = (byteSize / sizeof(float)) / batchSize_; // 推理输出的单帧数据长度
        auto *dataBuf = (float*)byteBuf;

        for (int32_t i = 0; i < pInputInfo->realNum; ++i) {
            auto frameIter = frameInfoMap_.find(pInputInfo->vDataId[i]);
            if (frameIter == frameInfoMap_.end()) {
                std::cout << "[ERROR][PostProcess] Can not find dataId(" << pInputInfo->vDataId[i] << ")." << std::endl;
                return false;
            }

            std::string fileName = frameIter->second + "_" + std::to_string(index);
            if (!VisionDumpResult(fileName, dataBuf + i * frameLen, frameLen)) {
                std::cout << "[ERROR][PostProcess] SaveToTxtFile failed." << std::endl;
                return false;
            }
        }

        index++;
    }

    return true;
}

bool PostProcess::VisionDumpResult(const std::string &fileName, const float* pBuf, const uint32_t& len)
{
    if (isOutputBinary_ == false) {
        std::string context = "";
        for (int32_t j = 0; j < len; ++j) {
            context = context + std::to_string(pBuf[j]) + " ";
        }
        context.append("\n");
        if (!SaveToTxtFile(fileName + ".txt", context, false)) {
            std::cout << "[ERROR][PostProcess] SaveToTxtFile failed." << std::endl;
            return false;
        }
    } else {
        if (!SaveToBinaryFile(fileName + ".bin", pBuf, len * sizeof(float), false)) {
            std::cout << "[ERROR][PostProcess] SaveToTxtFile failed." << std::endl;
            return false;
        }
    }

    return true;
}

bool PostProcess::ProcessNLP()
{
    return true;
}

bool PostProcess::ProcessFastRCNN()
{
    return true;
}

bool PostProcess::ProcessNMT()
{
    std::shared_ptr<ModelOutputData> pInputInfo = nullptr;
    deviceInputQue_->Pop(pInputInfo);
    if (!pInputInfo) {
        return true;
    }

    // 校验输入数据的合法性
    if (pInputInfo->realNum <= 0 || pInputInfo->realNum != pInputInfo->vDataId.size() ||
        pInputInfo->modelOutputData.size() != 1) {
        std::cout << "[ERROR][PostProcess] nmt_postprocess input is error." << std::endl;
        return false;
    }

    // 性能统计使用
    curSampleNum_ += pInputInfo->realNum;
    std::cout << "[INFO][PostProcess] curSampleNum is " << curSampleNum_ << std::endl;

    auto iter = pInputInfo->modelOutputData.cbegin();
    uint32_t byteSize = iter->second.len;
    uint8_t* byteBuf = iter->second.buf.get();
    // 深入校验输入数据的合法性
    if (byteSize % batchSize_ || byteSize % sizeof(int32_t) || byteBuf == nullptr) {
        std::cout << "[ERROR][PostProcess] nmt_postprocess data is error." << std::endl;
        return false;
    }

    uint32_t dataLen = byteSize / sizeof(int32_t);
    uint32_t frameLen = dataLen / batchSize_;
    auto* dataBuf = (int32_t*)byteBuf;
    std::string context = "";
    for (int32_t i = 0; i < pInputInfo->realNum; ++i) {
        for (int32_t j = 0; j < frameLen; ++j) {
            if (dataBuf[i * frameLen + j] == EOS_VALUE) {
                break;
            }
            auto hashIter = hashTab_.find(dataBuf[frameLen * i + j]);
            if (hashIter != hashTab_.end()) {
                context.append(hashIter->second + " ");
            } else {
                context.append(" ");
            }
        }

        if (context.length() > 0) {
            // 移除最后一个空格
            context.pop_back();
        }

        context.append("\n");
    }

    if (!SaveToTxtFile("nmt_output_file.txt", context, true)) {
        std::cout << "[ERROR][PostProcess] SaveToTxtFile failed." << std::endl;
        return false;
    }

    isFinished_ = pInputInfo->finish;

    return true;
}

bool PostProcess::ProcessWideDeep()
{
    std::shared_ptr<ModelOutputData> pDeviceInputInfo = nullptr;
    deviceInputQue_->Pop(pDeviceInputInfo);
    if (!pDeviceInputInfo) {
        return true;
    }

    // 校验device侧的数据的合法性,设备端一次性送一批数据
    if (pDeviceInputInfo->realNum <= 0 || pDeviceInputInfo->realNum != pDeviceInputInfo->vDataId.size() ||
        pDeviceInputInfo->modelOutputData.size() != 1) {
        std::cout << "[ERROR][PostProcess] widtDeep_postprocess deviceInput is error." << std::endl;
        return false;
    }

    // 性能统计使用
    curSampleNum_ += pDeviceInputInfo->realNum;

    auto deviceIter = pDeviceInputInfo->modelOutputData.cbegin();
    uint32_t deviceByteSize = deviceIter->second.len;
    uint8_t* pDeviceByteBuf = deviceIter->second.buf.get();
    // 深入校验输入数据的合法性
    if (deviceByteSize % batchSize_ || deviceByteSize % sizeof(float) || pDeviceByteBuf == nullptr) {
        std::cout << "[ERROR][PostProcess] widtDeep_postprocess deviceByteSize is error." << std::endl;
        return false;
    }

    if (WideDeepCore(pDeviceInputInfo) == false) {
        std::cout << "[ERROR][PostProcess] WideDeepCore failed." << std::endl;
        return false;
    }

    isFinished_ = pDeviceInputInfo->finish;

    if (isFinished_) {
        std::string context = "";
        context += "rightCount: ";
        context += std::to_string(rightCnt_);
        context += "\n";
        context += "totalCount: ";
        context += std::to_string(totalCnt_);
        context += "\n";
        context += "accuracy: ";
        context += std::to_string((rightCnt_ * 1.0) / totalCnt_);
        context += "\n";

        if (!SaveToTxtFile("widedeep_outputfile.txt", context, false)) {
            std::cout << "[ERROR][PostProcess] SaveToTxtFile failed." << std::endl;
            return false;
        }
    }

    return true;
}

bool PostProcess::WideDeepCore(std::shared_ptr<ModelOutputData>& pDeviceInputInfo)
{
    auto deviceIter = pDeviceInputInfo->modelOutputData.cbegin();
    uint32_t deviceByteSize = deviceIter->second.len;
    uint8_t* pDeviceByteBuf = deviceIter->second.buf.get();

    for (int i = 0; i < pDeviceInputInfo->realNum; ++i) {
        std::shared_ptr<ModelOutputData> pHostInputInfo = nullptr;
        hostInputQue_->Pop(pHostInputInfo);
        if (!pHostInputInfo) {
            break;
        }

        // 校验host侧的数据的合法性(主机端一次性送单个数据),且校验device侧与host侧的数据ID是否匹配
        if (pHostInputInfo->realNum != 1 || pHostInputInfo->vDataId.size() != 1 ||
            pHostInputInfo->modelOutputData.size() != 1 || pDeviceInputInfo->vDataId[i] != pHostInputInfo->vDataId[0]) {
            std::cout << "[ERROR][PostProcess] widtDeep_postprocess hostInput is error." << std::endl;
            return false;
        }

        auto hostIter = pHostInputInfo->modelOutputData.cbegin();
        uint32_t hostByteSize = hostIter->second.len;
        uint8_t* pHostByteBuf = hostIter->second.buf.get();
        // 校验device侧与host侧的数据长度是否匹配
        if (hostByteSize * batchSize_ != deviceByteSize) {
            std::cout << "[ERROR][PostProcess] The data length of deviceInput and hostInput do not match." << std::endl;
            return false;
        }

        // 主机侧数据类型转换
        uint32_t hostDataLen = hostByteSize / sizeof(float);
        float* pHostDataBuf = (float*)pHostByteBuf;
        float* pDeviceDataBuf = (float*)pDeviceByteBuf + hostDataLen * i;
        if (!FuseWideDeep(hostDataLen, pDeviceDataBuf, pHostDataBuf)) {
            std::cout << "[ERROR][PostProcess] FuseWideDeep failed." << std::endl;
            return false;
        }
    }

    return true;
}

bool PostProcess::FuseWideDeep(uint32_t dataLen, float* pDeviceBuf, float* pHostBuf)
{
    if (dataLen != 1 || pDeviceBuf == nullptr || pHostBuf == nullptr) {
        std::cout << "[ERROR][PostProcess] FuseWideDeep input is error." << std::endl;
        return false;
    }

    uint8_t isMoreThan50K = vecFlag_[totalCnt_];
    totalCnt_++;

    float sumWideDeep = *pDeviceBuf + *pHostBuf;
    float softmaxExp = exp(0) + exp(sumWideDeep);
    const uint32_t resultSize = 2;
    float softmaxResult[resultSize];
    softmaxResult[0] = exp(0) / softmaxExp;
    softmaxResult[1] = exp(sumWideDeep) / softmaxExp;

    if ((softmaxResult[1] > softmaxResult[0] && isMoreThan50K) ||
        (softmaxResult[1] <= softmaxResult[0] && !isMoreThan50K)) {
        rightCnt_++;
    }

    return true;
}

bool PostProcess::SaveToTxtFile(const std::string& fileName, const std::string& context, const bool isAppend)
{
    if (fileName == "") {
        std::cout << "[ERROR][PostProcess] fileName can not be empty." << std::endl;
        return false;
    }

    std::ofstream out;
    if (isAppend == true) {
        out.open(fileName, std::ofstream::app);
    } else {
        out.open(fileName);
    }

    if (!out.is_open()) {
        std::cout << "[ERROR][PostProcess] Can not open " << fileName << std::endl;
        return false;
    }

    out << context;
    out.close();

    return true;
}

bool PostProcess::SaveToBinaryFile(const std::string& fileName, const void* context, size_t size, const bool isAppend)
{
    if (fileName == "" || context == nullptr || size <= 0) {
        std::cout << "[ERROR][PostProcess] The params of SaveToBinaryFile error." << std::endl;
        return false;
    }

    std::ofstream out;
    if (isAppend == true) {
        out.open(fileName, std::ofstream::binary | std::ofstream::app);
    } else {
        out.open(fileName, std::ofstream::binary);
    }

    if (!out.is_open()) {
        std::cout << "[ERROR][PostProcess] Can not open " << fileName << std::endl;
        return false;
    }

    out.write((char*)context, size);
    out.close();

    return true;
}

void PostProcess::ThreadProc()
{
    bool status = false;
    while (isKeepAlive_) {
        status = (this->*processFunc_)();
        if (!status) {
            std::cout << "[ERROR][PostProcess] Fail to process data." << std::endl;
        } else {
            struct timeval end;
            end.tv_sec = 0;
            end.tv_usec = 0;
            gettimeofday(&end, nullptr);
            curTimeCost_ = (end.tv_sec - threadStart_.tv_sec) + (end.tv_usec - threadStart_.tv_usec) / CONVERT_USEC;
            perfInfo_->throughputRate = curSampleNum_ / curTimeCost_;
            perfInfo_->moduleLantency = CONVERT_MS / perfInfo_->throughputRate; // ms
        }
    }
}

bool PostProcess::Run(PostBlockQueue* deviceInput, PostBlockQueue* hostInput)
{
    if ((deviceInput == nullptr) || (modelType_ == MT_WIDEDEEP && hostInput == nullptr)) {
        std::cout << "[ERROR][PostProcess] param is error." << std::endl;
        return false;
    }

    deviceInputQue_ = deviceInput;
    hostInputQue_ = hostInput;
    gettimeofday(&threadStart_, nullptr);
    processThr_ = std::thread(&PostProcess::ThreadProc, this);

    return true;
}

std::shared_ptr<PerfInfo> PostProcess::GetPerfInfo()
{
    return perfInfo_;
}

bool PostProcess::GetFinishFlag()
{
    // 虽然该标志会被多线程访问,但由于旧值影响小于加锁的影响,因此此处不加互斥锁
    return isFinished_;
}