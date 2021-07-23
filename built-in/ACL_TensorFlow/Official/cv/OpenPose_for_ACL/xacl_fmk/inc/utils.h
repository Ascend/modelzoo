/*
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/*
Created by wang-bain on 2021/3/18.
*/


#include "acl/acl.h"

#include <algorithm>
#include <dirent.h>
#include <map>
#include <memory.h>
#include <string>
#include <vector>
#include <fstream>
#include <getopt.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

#ifndef RUN_ACL_MODEL_UTILS_H
#define RUN_ACL_MODEL_UTILS_H

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

void Help();

string GetTime();

string GetInputName(string inputFile);

string FormatInt(string Idx, size_t formatSize);

void Split(const string &tokens, vector<string> &v_tokens, const string &delimiters);

void ScanFiles(vector <string> &v_fileList, string inputDirectory);

void MergeInputFile(vector <string> &fileList, string &outputFile);

Result InitAndCheckParams(int argc, char *argv[], map<char, string> &params);

Result CreateDir(string &pathName);

Result InitResource(vector<aclrtContext> &v_context, map<char, string> &params);

Result DestroyResource(uint32_t modelId, aclmdlDesc *modelDesc, vector<aclrtContext> &v_context, map<char, string> &params);

Result LoadModelFromFile(uint32_t * modelId, map<char, string> & params);

Result CreateModelDesc(aclmdlDesc *&modelDesc, uint32_t modelId);

Result CreateInputDesc(aclmdlDataset *input, vector<string> v_inputFiles, aclTensorDesc *inputDesc[], map<char, string> &params);

Result CreateInput(aclmdlDesc *modelDesc, aclmdlDataset *&input, string &outputPrefix, vector<string> v_inputFiles, vector<vector<string>> saveOriginInput, map<char, string> &params);

Result CreateOutput(aclmdlDesc *modelDesc, aclmdlDataset *&output, aclmdlDataset *input, map<char, string> &params);

Result Execute(aclmdlDataset *input, aclmdlDataset *output, aclmdlDesc *modelDesc, uint32_t modelId, vector<aclrtContext> &v_context, float &totalSamplesTime, string outputPrefix, vector<vector<string>> saveOriginInput, uint32_t fileIdx, map<char, string> &params);

Result WriteResult(uint32_t stepNumber, float totalSamplesTime, map<char, string> &params);

#endif //RUN_ACL_MODEL_UTILS_H
