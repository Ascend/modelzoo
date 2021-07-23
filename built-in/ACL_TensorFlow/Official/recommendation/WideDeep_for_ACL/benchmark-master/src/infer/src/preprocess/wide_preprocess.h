/**
 * ============================================================================
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2020. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#ifndef WIDE_PREPROCESS_H
#define WIDE_PREPROCESS_H

#include <vector>
#include <string>
#include <thread>
#include <sys/time.h>
#include "common/data_struct.h"
class WidePreProcess {
public:
    WidePreProcess();
    ~WidePreProcess();
    int Init();
    void DeInit();
    void Process(BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr, BlockingQueue<std::shared_ptr<ModelOutputData>>*
    outputQueuePtr);
    std::shared_ptr<PerfInfo> GetPerfInfo();

private:

    float* ageBucketsWideWeight;
    float* ageXEduXOccWeight;
    float* eduXOccWeight;
    float* educationWideWeight;
    float* maritalStatusWideWeight;
    float* occupationWideWeight;
    float* relationshipWideWeight;
    float* workclassWideWeight;
    float* wideBiasWeight;

    std::vector<std::string> educationArrayWide;
    std::vector<std::string> maritalStatusArrayWide;
    std::vector<std::string> relationshipArrayWide;
    std::vector<std::string> workclassArrayWide;
    std::vector<std::uint32_t> ageBucketsArrayWide;

    std::thread processThr_;
    void ProcessThread();
    uint32_t Ptr2Int(uint8_t* ptr);

    bool isStop_ = false;

    BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr_ = nullptr;
    BlockingQueue<std::shared_ptr<ModelOutputData>>* outputQueuePtr_ = nullptr;

    // for perf
    std::shared_ptr<PerfInfo> perfInfo_;
    uint64_t GetCurentTimeStamp();
    struct timeval currentTimeval;
    uint64_t initialTimeStamp;
};

#endif

