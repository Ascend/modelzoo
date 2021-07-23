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

#ifndef NMT_PREPROCESS_H
#define NMT_PREPROCESS_H

#include <memory>
#include <thread>
#include <string>
#include <sys/time.h>
#include "HashTable.h"
#include "common/data_struct.h"
#include "acl/acl.h"

class NmtPreprocess {
public:
    NmtPreprocess() {}
    ~NmtPreprocess();
    int Init(std::string vocabTabFile);
    void DeInit();
    void Process(BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr, BlockingQueue<std::shared_ptr<ModelInputData>>*
	outputQueuePtr);
    std::shared_ptr<PerfInfo> GetPerfInfo();

private:
    HashTable* hashTable;
    size_t eosValue;
    uint32_t maxLineLen = 0;
    std::thread processThr_;
    bool isStop_ = false;
    void ProcessThread();
    uint32_t count = 0;

    BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr_ = nullptr;
    BlockingQueue<std::shared_ptr<ModelInputData>>* outputQueuePtr_ = nullptr;

    // for perf
    std::shared_ptr<PerfInfo> perfInfo_;
    uint64_t GetCurentTimeStamp();
    struct timeval currentTimeval;
    uint64_t initialTimeStamp;
   
};

#endif