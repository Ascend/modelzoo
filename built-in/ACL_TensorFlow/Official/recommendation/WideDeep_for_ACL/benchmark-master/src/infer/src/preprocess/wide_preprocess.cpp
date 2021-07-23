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

#include "wide_preprocess.h"
#include "string_hash.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <sys/time.h>
#include "securec.h"
const int STR_MAX_LEN = 128;
const int TIMES_SECOND_MICROSECOND = 1000000;

const int EDUCATION_ELT = 16;
const int MARITAL_STATUS_ELT = 7;
const int RELATIONSHIP_ELT = 6;
const int WORK_CLASS_ELT = 9;
const int AGE_BUCKETS = 10;
WidePreProcess::WidePreProcess()
{

	ageBucketsWideWeight = nullptr;
	ageXEduXOccWeight = nullptr;
	eduXOccWeight = nullptr;
	educationWideWeight = nullptr;
	maritalStatusWideWeight = nullptr;
	occupationWideWeight = nullptr;
	relationshipWideWeight = nullptr;
	workclassWideWeight = nullptr;
	wideBiasWeight = nullptr;

	educationArrayWide.clear();
	maritalStatusArrayWide.clear();
	relationshipArrayWide.clear();
	workclassArrayWide.clear();
	ageBucketsArrayWide.clear();
}

WidePreProcess::~WidePreProcess()
{


	if (ageBucketsWideWeight != nullptr) {
		delete[]ageBucketsWideWeight;
	}

	if (ageXEduXOccWeight != nullptr) {
		delete[]ageXEduXOccWeight;
	}

	if (eduXOccWeight != nullptr) {
		delete[]eduXOccWeight;
	}

	if (educationWideWeight != nullptr) {
		delete[]educationWideWeight;
	}

	if (maritalStatusWideWeight != nullptr) {
		delete[]maritalStatusWideWeight;
	}

	if (occupationWideWeight != nullptr) {
		delete[]occupationWideWeight;
	}

	if (relationshipWideWeight != nullptr) {
		delete[]relationshipWideWeight;
	}

	if (workclassWideWeight != nullptr) {
		delete[]workclassWideWeight;
	}

	if (wideBiasWeight != nullptr) {
		delete[]wideBiasWeight;
	}		
}

int WidePreProcess::Init()
{

	uint32_t i;

	string education[]     = {"Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",\
		"Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",\
		"5th-6th", "10th", "1st-4th", "Preschool", "12th"};
	string maritalStatus[] = {"Married-civ-spouse", "Divorced", "Married-spouse-absent",\
		"Never-married", "Separated", "Married-AF-spouse", "Widowed"};                          
	string relationship[]  = {"Husband", "Not-in-family", "Wife", "Own-child", "Unmarried", "Other-relative"};      
	string workclass[]     = {"Self-emp-not-inc", "Private", "State-gov", "Federal-gov",\
		"Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"};
	uint32_t ageBuckets[]    = {18, 25, 30, 35, 40, 45, 50, 55, 60, 65};

	for (i = 0; i < EDUCATION_ELT; i++) {
		educationArrayWide.push_back(education[i]);
	}
	for (i = 0; i < MARITAL_STATUS_ELT; i++) {
		maritalStatusArrayWide.push_back(maritalStatus[i]);
	}
	for (i = 0; i < RELATIONSHIP_ELT; i++) {
		relationshipArrayWide.push_back(relationship[i]);
	}
	for (i = 0; i < WORK_CLASS_ELT; i++) {
		workclassArrayWide.push_back(workclass[i]);
	}
	for (i = 0; i < AGE_BUCKETS; i++) {
		ageBucketsArrayWide.push_back(ageBuckets[i]);
	}

	float *occupationWideWeightBin = new float[1000];
	occupationWideWeight = occupationWideWeightBin;
	ifstream oWWeights("acl/bin/OccupationWideWeight.bin", std::ios::binary | std::ios::in);
	oWWeights.read((char*)occupationWideWeightBin, sizeof(float) * 1000);
	oWWeights.close();

	float *ageXEduXOccWeightBin = new float[1000];
	ageXEduXOccWeight = ageXEduXOccWeightBin;
	ifstream aEOWeights("acl/bin/AgeXEduXOccWeight.bin", std::ios::binary | std::ios::in);
	aEOWeights.read((char*)ageXEduXOccWeightBin, sizeof(float) * 1000);
	aEOWeights.close();

	float *eduXOccWeightBin = new float[1000];
	eduXOccWeight = eduXOccWeightBin;
	ifstream eOWeights("acl/bin/EduXOccWeight.bin", std::ios::binary | std::ios::in);
	eOWeights.read((char*)eduXOccWeightBin, sizeof(float) * 1000);
	eOWeights.close();

	float *educationWideWeightBin = new float[EDUCATION_ELT];
	educationWideWeight = educationWideWeightBin;
	ifstream eWWeights("acl/bin/EducationWideWeight.bin", std::ios::binary | std::ios::in);
	eWWeights.read((char*)educationWideWeightBin, sizeof(float) * EDUCATION_ELT);
	eWWeights.close();

	float *maritalStatusWideWeightBin = new float[MARITAL_STATUS_ELT];
	maritalStatusWideWeight = maritalStatusWideWeightBin;
	ifstream mSWWeights("acl/bin/MaritalStatusWideWeight.bin", std::ios::binary | std::ios::in);
	mSWWeights.read((char*)maritalStatusWideWeightBin, sizeof(float) * MARITAL_STATUS_ELT);
	mSWWeights.close();

	float *relationshipWideWeightBin = new float[RELATIONSHIP_ELT];
	relationshipWideWeight = relationshipWideWeightBin;
	ifstream rWWeights("acl/bin/RelationshipWideWeight.bin", std::ios::binary | std::ios::in);
	rWWeights.read((char*)relationshipWideWeightBin, sizeof(float) * RELATIONSHIP_ELT);
	rWWeights.close();

	float *workclassWideWeightBin = new float[WORK_CLASS_ELT];
	workclassWideWeight = workclassWideWeightBin;
	ifstream wWWeights("acl/bin/WorkclassWideWeight.bin", std::ios::binary | std::ios::in);
	wWWeights.read((char*)workclassWideWeightBin, sizeof(float) * WORK_CLASS_ELT);
	wWWeights.close();

	float *ageBucketsWideWeightBin = new float[11];
	ageBucketsWideWeight = ageBucketsWideWeightBin;
	ifstream aBWWeights("acl/bin/AgeBucketsWideWeight.bin", std::ios::binary | std::ios::in);
	aBWWeights.read((char*)ageBucketsWideWeightBin, sizeof(float) * 11);
	aBWWeights.close();

	float *wideBiasBin = new float[1];
	wideBiasWeight = wideBiasBin;
	ifstream biasWeights("acl/bin/WideBiasWeight.bin", std::ios::binary | std::ios::in);
	biasWeights.read((char*)wideBiasBin, sizeof(float) * 1);
	biasWeights.close();

	perfInfo_ = std::make_shared<PerfInfo>();
	
	initialTimeStamp = GetCurentTimeStamp();

	std::cout << "[INFO][WidePreProcess] Init SUCCESS" << std::endl;
	return 0;
}

void WidePreProcess::DeInit()
{
	isStop_ = true;
	inputQueuePtr_->Stop();
	processThr_.join();
	std::cout << "[INFO][WidePreProcess] DeInit SUCCESS" << std::endl;
}

uint32_t WidePreProcess::Ptr2Int(uint8_t* ptr) 
{
	std::stringstream tmpSstr((char*)ptr);
	uint32_t tmpInt;
	tmpSstr >> tmpInt;
	return tmpInt;
}

uint64_t WidePreProcess::GetCurentTimeStamp()
{
	gettimeofday(&currentTimeval, NULL);
	return currentTimeval.tv_sec * TIMES_SECOND_MICROSECOND + currentTimeval.tv_usec;
}

void WidePreProcess::ProcessThread()
{
	while (!isStop_) {
		std::shared_ptr<RawData> inputRawData = nullptr;
		inputQueuePtr_->Pop(inputRawData);
		if (!inputRawData) {
			continue;
		}

		uint32_t iD = inputRawData->dataId;

		// age
		uint32_t age = Ptr2Int(inputRawData->text.textRawData[0].buf.get());

		// workclass
		std::string workclassStr((char*)inputRawData->text.textRawData[1].buf.get());

		// education
		std::string educationStr((char*)inputRawData->text.textRawData[3].buf.get());

		// 	educationNum
		uint32_t educationNum = Ptr2Int(inputRawData->text.textRawData[4].buf.get());

		// maritalStatus
		std::string maritalStatusStr((char*)inputRawData->text.textRawData[5].buf.get());

		// occupation
		std::string occupationStr((char*)inputRawData->text.textRawData[6].buf.get());

		// relation
		std::string relationshipStr((char*)inputRawData->text.textRawData[7].buf.get());

		// capitalGain
		uint32_t capitalGain = Ptr2Int(inputRawData->text.textRawData[10].buf.get());

		// capitalLoss
		uint32_t capitalLoss = Ptr2Int(inputRawData->text.textRawData[11].buf.get());

		// hoursPerWeek
		uint32_t hoursPerWeek = Ptr2Int(inputRawData->text.textRawData[12].buf.get());
	
		float wideResult = 0.0;
		float wideOutPut[8] = {0};

		// 1.age_bucketized
		uint32_t ageBucket = 0;
		sort(ageBucketsArrayWide.begin(), ageBucketsArrayWide.end());

		if ( age >= 0 && age < ageBucketsArrayWide[0]) {
			ageBucket = 0;
		} else if (age >= ageBucketsArrayWide[ageBucketsArrayWide.size()-1]) {
			ageBucket = ageBucketsArrayWide.size();
		} else {
			for (uint32_t i = 0; i < ageBucketsArrayWide.size(); i++) {	
				if ( age >= ageBucketsArrayWide[i] && age < ageBucketsArrayWide[i + 1]) {
					ageBucket = i + 1;
					break;
				}
			}
		}	
		wideOutPut[0] = float(ageBucketsWideWeight[ageBucket]);
		wideResult += wideOutPut[0];

		// 3.education_X_occupation
		uint64_t eduXOccHash64Key = 956888297470;
		uint64_t eduXOccHash64;
		uint64_t eduHash64 = Fingerprint64(educationStr);
		uint64_t occHash64 = Fingerprint64(occupationStr);
		eduXOccHash64 = FingerprintCat64(eduXOccHash64Key, eduHash64);
		eduXOccHash64 = FingerprintCat64(eduXOccHash64, occHash64);
		uint64_t eduXOccHash64Idx = eduXOccHash64 % 1000;
		wideOutPut[2] = eduXOccWeight[eduXOccHash64Idx];
		wideResult += wideOutPut[2];

		// 2.age_bucketized_X_education_X_occupation (after 3.education_X_occupation)
		uint64_t ageXEduXOccHash64;      	
		ageXEduXOccHash64 = FingerprintCat64(eduXOccHash64Key, ageBucket);
		ageXEduXOccHash64 = FingerprintCat64(ageXEduXOccHash64, eduHash64);
		ageXEduXOccHash64 = FingerprintCat64(ageXEduXOccHash64, occHash64);
		uint64_t ageXEduXOccHash644Idx = ageXEduXOccHash64 % 1000;
		wideOutPut[1] = ageXEduXOccWeight[ageXEduXOccHash644Idx];
		wideResult += wideOutPut[1];		

		// 4.education
		vector<string>::iterator result = find(educationArrayWide.begin(), educationArrayWide.end(), educationStr.c_str());
		if (result == educationArrayWide.end()) {
			std::cout << "[ERROR][WidePreProcess] Invalid education: " << educationStr << std::endl;
		}
		wideOutPut[3] = educationWideWeight[int(result - educationArrayWide.begin())];
		wideResult += wideOutPut[3];

		// 5.marital_status
		result = find(maritalStatusArrayWide.begin(), maritalStatusArrayWide.end(), maritalStatusStr.c_str());
		if (result == maritalStatusArrayWide.end()) {
			std::cout << "[ERROR][WidePreProcess] Invalid maritalStatus: " << maritalStatusStr << std::endl;
		}
		wideOutPut[4] = maritalStatusWideWeight[int(result - maritalStatusArrayWide.begin())];
		wideResult += wideOutPut[4];

		// 6.occupation
		uint64_t occHash64Idx = occHash64 % 1000;
		wideOutPut[5] = occupationWideWeight[occHash64Idx];
		wideResult += wideOutPut[5];

		// 7.relationship
		result = find(relationshipArrayWide.begin(), relationshipArrayWide.end(), relationshipStr.c_str());
		if (result == relationshipArrayWide.end()) {
			std::cout << "[ERROR][WidePreProcess] Invalid relationship: " << relationshipStr << std::endl;
		}
		wideOutPut[6] = relationshipWideWeight[int(result - relationshipArrayWide.begin())];
		wideResult += wideOutPut[6];

		// 8.workclass
		result = find(workclassArrayWide.begin(), workclassArrayWide.end(), workclassStr.c_str());
		if (result == workclassArrayWide.end()) {
			std::cout << "[ERROR][WidePreProcess] Invalid workclass: " << workclassStr << std::endl;
		}
		wideOutPut[7] = workclassWideWeight[int(result - workclassArrayWide.begin())];
		wideResult += wideOutPut[7];

		// Wide result
		wideResult += wideBiasWeight[0];

		std::shared_ptr<ModelOutputData> wideData = std::make_shared<ModelOutputData>();
		wideData->vDataId.push_back(inputRawData->dataId);
		wideData->realNum = 1;
		float *tmpWideResult = new float[1];
		*tmpWideResult = wideResult;
		wideData->modelOutputData["feature"].buf.reset((uint8_t*)tmpWideResult);
		wideData->modelOutputData["feature"].len = 4;

		outputQueuePtr_->Push(wideData); // wide process does not need infer

		// perf info
		perfInfo_->throughputRate = inputRawData->dataId / (1.0 * (GetCurentTimeStamp() - initialTimeStamp) / TIMES_SECOND_MICROSECOND);
		perfInfo_->moduleLantency = 1.0 / perfInfo_->throughputRate * 1000; // ms
	}

}

std::shared_ptr<PerfInfo> WidePreProcess::GetPerfInfo()
{
	return perfInfo_;
}

void WidePreProcess::Process(BlockingQueue<std::shared_ptr<RawData>>* inputQueuePtr, 
	BlockingQueue<std::shared_ptr<ModelOutputData>>* outputQueuePtr)
{
	inputQueuePtr_ = inputQueuePtr;
	outputQueuePtr_ = outputQueuePtr;
	processThr_ = std::thread(&WidePreProcess::ProcessThread, this);
}

