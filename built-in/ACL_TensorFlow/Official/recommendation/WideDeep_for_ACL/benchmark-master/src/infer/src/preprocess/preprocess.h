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

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <thread>
#include <vector>
#include <utility>
#include <string>

#include "common/block_queue.h"
#include "common/data_struct.h"
#include "wide_preprocess.h"
#include "deep_preprocess.h"
#include "nmt_preprocess.h"
#include "vision_preprocess.h"

class Preprocess {
public:
    Preprocess(){}
    ~Preprocess(){}
    bool Init(ModelType m, std::string vocabTabFile);
    bool Init(ModelType m, int input_width, int input_height);
    int DeInit();
    int Run(BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr, BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr1,
    BlockingQueue<std::shared_ptr<ModelInputData>>* 
    outputQueuePtr, BlockingQueue<std::shared_ptr<ModelOutputData>>* outputQueuePtr1);
    std::vector<std::shared_ptr<PerfInfo>> GetPerfInfo();

private:
    ModelType modelType_ = MT_WIDEDEEP;
    std::thread processThr_;
    void ProcessThread();

    WidePreProcess* widePreProcessInstance = nullptr;
    DeepPreprocess* deepPreprocessInstance = nullptr;
    NmtPreprocess* nmtPreprocessInstance = nullptr;
    VisionPreProcess* visionPreprocessInstance = nullptr;

    BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr_ = nullptr;
    BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr1_ = nullptr;
    BlockingQueue<std::shared_ptr<ModelInputData>>* outputQueuePtr_ = nullptr;
    BlockingQueue<std::shared_ptr<ModelOutputData>>* outputQueuePtr1_ = nullptr; // only for widedeep model.

    std::string vocabTabFile_;
    int input_width_ = 224;
    int input_height_ = 224;

};

#endif