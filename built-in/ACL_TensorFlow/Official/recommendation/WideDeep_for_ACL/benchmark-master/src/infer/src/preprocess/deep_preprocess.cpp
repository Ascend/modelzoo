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

#include "deep_preprocess.h"
#include "string_hash.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <sys/time.h>
#include "securec.h"
using namespace std;
const int STR_MAX_LEN = 128;
const int DEEP_INFO_COUNT = 51;
const int TIMES_SECOND_MICROSECOND = 1000000;
const int OCCUPATION_EMBEDDING_LEN = 8000;
const int EDUCATION_ONE_HOT = 16;
const int MARITAL_STATUS_ONE_HOT = 7;
const int RELATIONSHIP_ONE_HOT = 6;
const int WORK_CLASS_ONE_HOT = 9;
const int OCCUPATION_EMBEDDING = 8;
DeepPreprocess::DeepPreprocess()
{
	processSize = 0;
	bufferSize = 0;
	occupationEmbeddingWeight = nullptr;
}

DeepPreprocess::~DeepPreprocess()
{

	if (occupationEmbeddingWeight != nullptr) {
		delete[] occupationEmbeddingWeight;
	}
}

int DeepPreprocess::Init()
{	

	processSize = 1;
	bufferSize = processSize * (DEEP_INFO_COUNT) * sizeof(float);
	
	uint32_t i;
	string education[] = {"Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",\
		"Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",\
		"5th-6th", "10th", "1st-4th", "Preschool", "12th"};
	string maritalStatus[] = {"Married-civ-spouse", "Divorced", "Married-spouse-absent",\
		"Never-married", "Separated", "Married-AF-spouse", "Widowed"};                          
	string relationship[]  = {"Husband", "Not-in-family", "Wife", "Own-child", "Unmarried", "Other-relative"};      
	string workclass[] = {"Self-emp-not-inc", "Private", "State-gov", "Federal-gov",\
		"Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"};

	for (i = 0; i < EDUCATION_ONE_HOT; i++) {
		educationArray.push_back(education[i]);
	}
	for (i = 0; i < MARITAL_STATUS_ONE_HOT; i++) {
		maritalStatusArray.push_back(maritalStatus[i]);
	}
	for (i = 0; i < RELATIONSHIP_ONE_HOT; i++) {
		relationshipArray.push_back(relationship[i]);
	}
	for (i = 0; i < WORK_CLASS_ONE_HOT; i++) {
		workclassArray.push_back(workclass[i]);
	}
	
	float *occupationEmbedding = new float[OCCUPATION_EMBEDDING_LEN];
	if (!occupationEmbedding) {
		std::cout << "[INFO][DeepPreprocess] Init failed" << std::endl;
		return -1;
	}
	occupationEmbeddingWeight = occupationEmbedding;
	
	ifstream oEWeights("acl/bin/occupation_embedding_weights.bin", std::ios::binary | std::ios::in);
	if (!oEWeights.is_open()) {
		std::cout << "[INFO][DeepPreprocess] failed to open occupation_embedding_weights" << std::endl;
		return -1;
	}
	oEWeights.read((char*)occupationEmbedding, sizeof(float) * OCCUPATION_EMBEDDING_LEN);
	oEWeights.close();

	perfInfo_ = std::make_shared<PerfInfo>();
	initialTimeStamp = GetCurentTimeStamp();

	std::cout << "[INFO][DeepPreprocess] Init SUCCESS" << std::endl;
	return 0;
	
}

void DeepPreprocess::DeInit()
{
	isStop_ = true;
	inputQueuePtr_->Stop();
	processThr_.join();
	std::cout << "[INFO][DeepPreprocess] DeInit SUCCESS" << std::endl;
}

uint32_t DeepPreprocess::Ptr2Int(uint8_t* ptr) 
{
	std::stringstream tmpSstr((char*)ptr);
	uint32_t tmpInt;
	tmpSstr >> tmpInt;
	return tmpInt;
}

uint64_t DeepPreprocess::GetCurentTimeStamp()
{
	gettimeofday(&currentTimeval, NULL);
	return currentTimeval.tv_sec * TIMES_SECOND_MICROSECOND + currentTimeval.tv_usec;
}

void DeepPreprocess::ProcessThread()
{
	//struct timeval time1, time2, time3, time4, time5, time6, time7, time8, time9;
	aclrtSetCurrentContext(context);
	while (!isStop_) {
		//gettimeofday(&time3, nullptr);
		std::shared_ptr<RawData> inputRawData = nullptr;
		inputQueuePtr_->Pop(inputRawData);
		if (!inputRawData) {
			continue;
		}
		
		float educationOneHot[EDUCATION_ONE_HOT] = {0};
		float maritalStatusOneHot[MARITAL_STATUS_ONE_HOT] = {0};
		float occupationEmbedding[OCCUPATION_EMBEDDING] = {0};
		float relationshipOneHot[RELATIONSHIP_ONE_HOT] = {0};
		float workclassOneHot[WORK_CLASS_ONE_HOT] = {0};

		// age
		uint32_t age = Ptr2Int(inputRawData->text.textRawData[0].buf.get());
		
		// workclass
		std::string workclassStr((char*)inputRawData->text.textRawData[1].buf.get());
		auto result = std::find(workclassArray.begin(), workclassArray.end(), workclassStr.c_str());
		if (result == workclassArray.end()) {
			std::cout << "[ERROR][DeepPreprocess] Invalid workclass: " << workclassStr << " ,length: " << workclassStr.length() << std::endl;
		}
		workclassOneHot[result - workclassArray.begin()] = 1;

		//  education
		std::string educationStr((char*)inputRawData->text.textRawData[3].buf.get());
		result = std::find(educationArray.begin(), educationArray.end(), educationStr.c_str());
		if (result == educationArray.end()) {
			std::cout << "[ERROR][DeepPreprocess] Invalid education: " << educationStr << " ,length: " << educationStr.length() << std::endl;
		}
		educationOneHot[result - educationArray.begin()] = 1;

		// educationNum
		uint32_t educationNum = Ptr2Int(inputRawData->text.textRawData[4].buf.get());
		
		// maritalStatus
		std::string maritalStatusStr((char*)inputRawData->text.textRawData[5].buf.get());
		result = std::find(maritalStatusArray.begin(), maritalStatusArray.end(), maritalStatusStr.c_str());
		if (result == maritalStatusArray.end()) {
			std::cout << "[ERROR][DeepPreprocess] Invalid maritalStatus: " << maritalStatusStr << " ,length: " << maritalStatusStr.length() << std::endl;
		}
		maritalStatusOneHot[result - maritalStatusArray.begin()] = 1;

		// occupation
		std::string occupationStr((char*)inputRawData->text.textRawData[6].buf.get());
		uint64_t occHash64 = Fingerprint64(occupationStr);
		uint64_t occHash64Idx = occHash64 % 1000;
		for (uint32_t i = 0; i < OCCUPATION_EMBEDDING; i++) {
			occupationEmbedding[i] = occupationEmbeddingWeight[occHash64Idx * OCCUPATION_EMBEDDING + i];
		}

		// relationship
		std::string relationshipStr((char*)inputRawData->text.textRawData[7].buf.get());
		result = std::find(relationshipArray.begin(), relationshipArray.end(), relationshipStr.c_str());
		if (result == relationshipArray.end()) {
			std::cout << "[ERROR][DeepPreprocess] Invalid relationship: " << relationshipStr << " ,length: " << relationshipStr.length() << std::endl;
		}
		relationshipOneHot[result - relationshipArray.begin()] = 1;	

		// capitalGain
		uint32_t capitalGain = Ptr2Int(inputRawData->text.textRawData[10].buf.get());

		// capitalLoss
		uint32_t capitalLoss = Ptr2Int(inputRawData->text.textRawData[11].buf.get());

		// hoursPerWeek
		uint32_t hoursPerWeek = Ptr2Int(inputRawData->text.textRawData[12].buf.get());

		//gettimeofday(&time4, nullptr);
		// calculate the feature
		void* alignBufferHost = nullptr;
		int ret = aclrtMallocHost(&alignBufferHost, bufferSize);
		if (ret != 0) {
			std::cout << "[ERROR][DeepPreprocess] aclrtMallocHost error, buffer size = " << bufferSize << std::endl;
		}
		//gettimeofday(&time5, nullptr);
		float *feature = (float*)alignBufferHost;
		
		feature[0] = float(age);
		feature[1] = float(capitalGain);
		feature[2] = float(capitalLoss);
		for (uint32_t i = 0; i < EDUCATION_ONE_HOT; i++) {
			feature[i + 3] = educationOneHot[i];	
		}
		feature[19] = float(educationNum);
		feature[20] = float(hoursPerWeek);
		for (uint32_t i = 0; i < MARITAL_STATUS_ONE_HOT; i++) {
			feature[i + 21] = maritalStatusOneHot[i];		
		}
		for (uint32_t i = 0; i < OCCUPATION_EMBEDDING; i++) {
			feature[i + 28] = occupationEmbedding[i];
		}
		for (uint32_t i = 0; i < RELATIONSHIP_ONE_HOT; i++) {
			feature[i + 36] = relationshipOneHot[i];
		}
		for (uint32_t i = 0; i < WORK_CLASS_ONE_HOT; i++) {
			feature[i + 42] = workclassOneHot[i];
		}
		//ACL_APP_LOG(ACL_ERROR, "%d", 12311);
		//std::cout << "zzzzzzzzzzzzzzzzzzzzzz" << bufferSize << std::endl;
		//gettimeofday(&time6, nullptr);
		void* alignBufferDevice = nullptr;
		
		ret = aclrtMalloc(&alignBufferDevice, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
		if (ret != 0) {
			std::cout << "[ERROR][DeepPreprocess] aclrtMalloc error!" << std::endl;
			aclrtFreeHost(alignBufferHost);
			continue;
		}
		//gettimeofday(&time7, nullptr);
		//ACL_APP_LOG(ACL_ERROR, "%d", 123222);
		ret = aclrtMemcpy(alignBufferDevice, bufferSize, alignBufferHost, bufferSize, ACL_MEMCPY_HOST_TO_DEVICE);
		if (ret != 0) {
			std::cout << "[ERROR][DeepPreprocess] aclrtMemcpy error!" << std::endl;
			aclrtFree(alignBufferDevice);
			aclrtFreeHost(alignBufferHost);
			continue;
		}
		//gettimeofday(&time8, nullptr);
		aclrtFreeHost(alignBufferHost);
		std::shared_ptr<ModelInputData> deepData = std::shared_ptr<ModelInputData> (new ModelInputData);
		deepData->dataId = inputRawData->dataId;
		deepData->finish = inputRawData->finish;
		deepData->modelType = MT_WIDEDEEP;
		DataBuf dataBuf;
		dataBuf.buf.reset((uint8_t*)alignBufferDevice, [](uint8_t *p){aclrtFree(p);});
		dataBuf.len = bufferSize;
		deepData->text.textRawData.push_back(dataBuf);

		outputQueuePtr_->Push(deepData);
		//gettimeofday(&time9, nullptr);

		//double e2eCost9 = (time9.tv_sec - time3.tv_sec) * 1000.0 + (time9.tv_usec - time3.tv_usec) / 1000.0;
		//double e2eCost8 = (time8.tv_sec - time3.tv_sec) * 1000.0 + (time8.tv_usec - time3.tv_usec) / 1000.0;
		//double e2eCost7 = (time7.tv_sec - time3.tv_sec) * 1000.0 + (time7.tv_usec - time3.tv_usec) / 1000.0;
		//double e2eCost6 = (time6.tv_sec - time3.tv_sec) * 1000.0 + (time6.tv_usec - time3.tv_usec) / 1000.0;
		//double e2eCost5 = (time5.tv_sec - time3.tv_sec) * 1000.0 + (time5.tv_usec - time3.tv_usec) / 1000.0;
		//double e2eCost4 = (time4.tv_sec - time3.tv_sec) * 1000.0 + (time4.tv_usec - time3.tv_usec) / 1000.0;
		//std::cout << "xxxxxxxxxxxxxxxxxxxxe2eCost9:" << e2eCost9 << std::endl;
		//std::cout << "xxxxxxxxxxxxxxxxxxxxe2eCost8:" << e2eCost8 << std::endl;
		//std::cout << "xxxxxxxxxxxxxxxxxxxxe2eCost7:" << e2eCost7 << std::endl;
		//std::cout << "xxxxxxxxxxxxxxxxxxxxe2eCost6:" << e2eCost6 << std::endl;
		//std::cout << "xxxxxxxxxxxxxxxxxxxxe2eCost5:" << e2eCost5 << std::endl;
		//std::cout << "xxxxxxxxxxxxxxxxxxxxe2eCost4:" << e2eCost4 << std::endl;


		// perf info
		perfInfo_->throughputRate = inputRawData->dataId / (1.0 * (GetCurentTimeStamp() - initialTimeStamp) / TIMES_SECOND_MICROSECOND);
		perfInfo_->moduleLantency = 1.0 / perfInfo_->throughputRate * 1000; // ms
	}
}

std::shared_ptr<PerfInfo> DeepPreprocess::GetPerfInfo()
{
	return perfInfo_;
}

void DeepPreprocess::Process(BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr, 
	BlockingQueue<std::shared_ptr<ModelInputData>>* outputQueuePtr)
{
	inputQueuePtr_ = inputQueuePtr;
	outputQueuePtr_ = outputQueuePtr;
	processThr_ = std::thread(&DeepPreprocess::ProcessThread, this);
}




