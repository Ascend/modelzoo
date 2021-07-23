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

#include "preprocess.h"
#include <iostream>

bool Preprocess::Init(ModelType m, std::string vocabTabFile)
{
    modelType_ = m;
    vocabTabFile_ = vocabTabFile;
    std::cout << "[INFO][Preprocess] Init SUCCESS" << std::endl;
    return true;
}

bool Preprocess::Init(ModelType m, int input_width, int input_height)
{
    modelType_ = m;
    input_width_ = input_width;
    input_height_ = input_height;
    std::cout << "[INFO][Preprocess] Init SUCCESS" << std::endl;
    return true;
}

int Preprocess::DeInit()
{
    switch (modelType_) {
        case MT_VISION:
            visionPreprocessInstance->DeInit();
            delete visionPreprocessInstance;
            break;

        case MT_WIDEDEEP: {
            widePreProcessInstance->DeInit();
            deepPreprocessInstance->DeInit();
            delete widePreProcessInstance;
            delete deepPreprocessInstance;
            break;
        }
        case MT_NMT: {
            nmtPreprocessInstance->DeInit();
            delete nmtPreprocessInstance;
            break;
        }
        
        default:
            break;
    }
    std::cout << "[INFO][Preprocess] DeInit SUCCESS" << std::endl;
}

std::vector<std::shared_ptr<PerfInfo>> Preprocess::GetPerfInfo()
{
    std::vector<std::shared_ptr<PerfInfo>> perfInfoVec;
    switch (modelType_) {
        case MT_VISION:
            perfInfoVec.push_back(visionPreprocessInstance->GetPerfInfo());
            break;
        case MT_WIDEDEEP:
            perfInfoVec.push_back(widePreProcessInstance->GetPerfInfo());
            perfInfoVec.push_back(deepPreprocessInstance->GetPerfInfo());
            break;
        case MT_NMT:
            perfInfoVec.push_back(nmtPreprocessInstance->GetPerfInfo());
            break;
        default:
            break;
    }
    return perfInfoVec;

}

void Preprocess::ProcessThread()
{
    switch (modelType_) {
    case MT_VISION:
        visionPreprocessInstance = new VisionPreProcess;
        visionPreprocessInstance->Init(input_width_, input_height_);
        visionPreprocessInstance->Process(inputQueuePtr_, outputQueuePtr_);
        break;
    case MT_WIDEDEEP:
        {
            widePreProcessInstance = new WidePreProcess;
            widePreProcessInstance->Init();
            widePreProcessInstance->Process(inputQueuePtr1_, outputQueuePtr1_);

            deepPreprocessInstance = new DeepPreprocess;
            deepPreprocessInstance->Init();
            deepPreprocessInstance->Process(inputQueuePtr_, outputQueuePtr_);
            
        }
        break;
    case MT_NMT:
        {
            nmtPreprocessInstance = new NmtPreprocess;
            nmtPreprocessInstance->Init(vocabTabFile_);
            nmtPreprocessInstance->Process(inputQueuePtr_, outputQueuePtr_);
        }
        break;
    default:
        break;
    }
}

int Preprocess::Run(BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr, 
                    BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr1,
                    BlockingQueue<std::shared_ptr<ModelInputData>>* outputQueuePtr, 
                    BlockingQueue<std::shared_ptr<ModelOutputData>>* outputQueuePtr1)
{
    inputQueuePtr_ = inputQueuePtr;
    inputQueuePtr1_ = inputQueuePtr1;
    outputQueuePtr_ = outputQueuePtr;
    outputQueuePtr1_ = outputQueuePtr1;
    ProcessThread();
    return 0;
}
