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

#include "nmt_preprocess.h"
#include <iostream>
#include "securec.h"
const int TIMES_SECOND_MICROSECOND = 1000000;
const int MAX_LINE_LEN = 64;
int NmtPreprocess::Init(std::string vocabTabFile)
{
    hashTable = new HashTable;
    if (hashTable->Init(vocabTabFile) != 0) {
        std::cout << "[ERROR][NmtPreprocess] Init FAILED" << std::endl;
        return -1;
    }
    eosValue = hashTable->LookUp("<unk>");
    
	perfInfo_ = std::make_shared<PerfInfo>();
	
	initialTimeStamp = GetCurentTimeStamp();

    std::cout << "[INFO][NmtPreprocess] Init SUCCESS" << std::endl;

    return 0;
}

void NmtPreprocess::DeInit()
{
    isStop_ = true;
    inputQueuePtr_->Stop();
    processThr_.join();
    std::cout << "[INFO][NmtPreprocess] DeInit SUCCESS" << std::endl;
}

NmtPreprocess::~NmtPreprocess() 
{
    
    if (hashTable) {
        delete hashTable;
    }
}

uint64_t NmtPreprocess::GetCurentTimeStamp()
{
	gettimeofday(&currentTimeval, nullptr);
	return currentTimeval.tv_sec * TIMES_SECOND_MICROSECOND + currentTimeval.tv_usec;
}

void NmtPreprocess::ProcessThread()
{
    aclrtSetCurrentContext(context);
    while (!isStop_) {
        std::vector<uint32_t> result;
        std::shared_ptr<RawData> inputRawData = nullptr;
        inputQueuePtr_->Pop(inputRawData);
		if (!inputRawData) {
			continue;
		}
        
        for (auto it = inputRawData->text.textRawData.begin(); it != inputRawData->text.textRawData.end(); it++) {
            std::string token((char*)((*it).buf.get()));
            result.push_back((uint32_t)hashTable->LookUp(token));
            if (result.size() >= MAX_LINE_LEN) {
                break;
            }
        }

        uint32_t realLen = result.size();
        int32_t paddingLen = MAX_LINE_LEN - result.size(); // 剩下的进行padding
        while (paddingLen > 0) {
            result.push_back(eosValue);
            paddingLen--;
        }
		void* resultBufferDevice = nullptr;
		int ret = aclrtMalloc(&resultBufferDevice, sizeof(uint32_t) * result.size(), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            std::cout << "[ERROR][NmtPreprocesss] aclrtMalloc error!" << std::endl;
            continue;
        }
		ret = aclrtMemcpy(resultBufferDevice, sizeof(uint32_t) * result.size(), &result[0], sizeof(uint32_t) * result.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != 0) {
            std::cout << "[ERROR][NmtPreprocesss] aclrtMemcpy error!" << std::endl;
            aclrtFree(resultBufferDevice);
            continue;
        }
        
		std::shared_ptr<ModelInputData> nmtPreData = std::make_shared<ModelInputData>();
        nmtPreData->dataId = inputRawData->dataId;
        nmtPreData->finish = inputRawData->finish;
		nmtPreData->modelType = MT_NMT;
        DataBuf seqBuf;
        seqBuf.buf.reset((uint8_t*)resultBufferDevice, [](uint8_t *p){aclrtFree(p);});
        seqBuf.len = sizeof(uint32_t) * result.size();
        nmtPreData->text.textRawData.push_back(seqBuf);

        uint32_t* tmpLen = new uint32_t[1];
        *tmpLen = realLen;
		void* tmpLenDevice = nullptr;
		ret = aclrtMalloc(&tmpLenDevice, sizeof(uint32_t), ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != 0) {
            std::cout << "[ERROR][NmtPreprocesss] aclrtMalloc error!" << std::endl;
            aclrtFree(resultBufferDevice);
            delete(tmpLen);
            continue;
        }
		ret = aclrtMemcpy(tmpLenDevice, sizeof(uint32_t), tmpLen, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != 0) {
            std::cout << "[ERROR][NmtPreprocesss] aclrtMemcpy error!" << std::endl;
            aclrtFree(resultBufferDevice);
            delete(tmpLen);
            aclrtFree(tmpLenDevice);
            continue;
        }
        DataBuf seqLenBuf;
        seqLenBuf.buf.reset((uint8_t*)tmpLenDevice, [](uint8_t *p){aclrtFree(p);});
        seqLenBuf.len = sizeof(uint32_t);
        nmtPreData->text.textRawData.push_back(seqLenBuf);

		outputQueuePtr_->Push(nmtPreData);

		// perf info
        const float MULTI = 1000.0;
		perfInfo_->throughputRate = inputRawData->dataId / (1.0 * (GetCurentTimeStamp() - initialTimeStamp) / TIMES_SECOND_MICROSECOND);
        perfInfo_->moduleLantency = MULTI / perfInfo_->throughputRate; // ms
    }
}

std::shared_ptr<PerfInfo> NmtPreprocess::GetPerfInfo()
{
	return perfInfo_;
}

void NmtPreprocess::Process(BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr, 
    BlockingQueue<std::shared_ptr<ModelInputData>>* outputQueuePtr)
{
	inputQueuePtr_ = inputQueuePtr;
	outputQueuePtr_ = outputQueuePtr;
    processThr_ = std::thread(&NmtPreprocess::ProcessThread, this);
}
